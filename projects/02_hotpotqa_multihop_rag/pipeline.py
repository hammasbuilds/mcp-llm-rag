"""Multi-hop RAG audit pipeline over HotpotQA (distractor, validation).

Real components only:
- `datasets` loads the actual HotpotQA validation split (distractor config) from
  the Hugging Face Hub.
- `httpx` calls a local Ollama server's `/api/embed` endpoint (model:
  nomic-embed-text) to embed every context sentence and the query used at
  each retrieval hop.
- `langchain_ollama.ChatOllama` (structured output via `.with_structured_output`)
  drives a local chat model (qwen2.5:7b-instruct if present) to produce a
  final answer plus a structured list of (title, sentence_index) citations.

The point of the pipeline is to keep two things clearly separate for every
example:
  (a) the sentences the retrieval step actually fed into the answer prompt
      ("fed_facts" -- ground truth for "what the pipeline used"), and
  (b) the (title, sentence_index) pairs the model *claims* it used
      ("claimed_facts").
Both are then compared against HotpotQA's real gold `supporting_facts`.
"""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass, field
from typing import Any

import httpx
from pydantic import BaseModel, Field

# langchain_ollama is imported inside build_chat_model(), not here.
#
# It is needed by exactly one function - the one that talks to a model server -
# and importing it at module scope made the whole module unimportable without it.
# That defeated the `live` marker this repo runs on: the scoring, normalisation
# and similarity functions have no model in them and are meant to be testable
# anywhere, but `pytest -m "not live"` still died at collection with
# ModuleNotFoundError, so the tests that were supposed to always run never ran.

OLLAMA_URL = "http://127.0.0.1:11434"
EMBED_MODEL = "nomic-embed-text"
CHAT_MODEL_PREFERRED = "qwen2.5:7b-instruct"
CHAT_MODEL_FALLBACKS = ["llama3.2:3b", "qwen2.5:3b-instruct", "qwen2.5-coder:3b", "granite3.3:2b"]

HOP1_K = 4
HOP2_K = 4


# --------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------


@dataclass
class Sentence:
    title: str
    sent_idx: int
    text: str

    @property
    def key(self) -> tuple[str, int]:
        return (self.title, self.sent_idx)


@dataclass
class ExampleRecord:
    qid: str
    question: str
    gold_answer: str
    sentences: list[Sentence]
    gold_facts: set[tuple[str, int]]


class SupportingFact(BaseModel):
    title: str = Field(description="Exact paragraph title the sentence came from.")
    sentence_index: int = Field(description="0-based sentence index within that paragraph.")


class CitedAnswer(BaseModel):
    answer: str = Field(description="Short, direct answer to the question.")
    supporting_facts: list[SupportingFact] = Field(
        min_length=1,
        description=(
            "REQUIRED. The (title, sentence_index) pairs you actually used to derive the "
            "answer. Must contain at least one entry copied exactly from the numbered context."
        ),
    )


@dataclass
class HopTrace:
    query: str
    retrieved: list[Sentence]


@dataclass
class ExampleResult:
    qid: str
    question: str
    gold_answer: str
    model_answer: str
    gold_facts: list[list[Any]]
    fed_facts: list[list[Any]]
    claimed_facts: list[list[Any]]
    hop1_query: str
    hop2_query: str
    answer_em: bool
    answer_f1: float
    retrieval_recall_of_gold: float
    claim_vs_gold_precision: float
    claim_vs_gold_recall: float
    claim_vs_gold_f1: float
    claim_vs_fed_precision: float
    claim_vs_fed_recall: float
    fabricated_citations: list[list[Any]] = field(default_factory=list)


# --------------------------------------------------------------------------
# Dataset loading
# --------------------------------------------------------------------------


def load_hotpotqa_sample(n: int, seed: int) -> list[ExampleRecord]:
    """Load a fixed, seeded sample of the real HotpotQA distractor validation split."""
    from datasets import load_dataset

    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation")
    ds = ds.shuffle(seed=seed).select(range(n))

    records: list[ExampleRecord] = []
    for ex in ds:
        titles = ex["context"]["title"]
        para_sents = ex["context"]["sentences"]
        sentences: list[Sentence] = []
        for title, sents in zip(titles, para_sents, strict=True):
            for idx, text in enumerate(sents):
                text = text.strip()
                if text:
                    sentences.append(Sentence(title=title, sent_idx=idx, text=text))

        sf_titles = ex["supporting_facts"]["title"]
        sf_ids = ex["supporting_facts"]["sent_id"]
        gold_facts = {(t, i) for t, i in zip(sf_titles, sf_ids, strict=True)}

        records.append(
            ExampleRecord(
                qid=ex["id"],
                question=ex["question"],
                gold_answer=ex["answer"],
                sentences=sentences,
                gold_facts=gold_facts,
            )
        )
    return records


# --------------------------------------------------------------------------
# Embedding + retrieval
# --------------------------------------------------------------------------


def embed_texts(texts: list[str], client: httpx.Client, model: str = EMBED_MODEL) -> list[list[float]]:
    """Batch-embed via Ollama's /api/embed endpoint."""
    if not texts:
        return []
    resp = client.post(f"{OLLAMA_URL}/api/embed", json={"model": model, "input": texts}, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return data["embeddings"]


def cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def two_hop_retrieve(
    question: str,
    sentences: list[Sentence],
    client: httpx.Client,
    hop1_k: int = HOP1_K,
    hop2_k: int = HOP2_K,
) -> tuple[list[Sentence], HopTrace, HopTrace]:
    """Two-hop dense retrieval.

    Hop 1: embed the raw question, take the top-k1 most similar sentences.
    Hop 2: embed the question concatenated with hop-1's retrieved sentence
    text (so the query now carries whatever bridge entity/fact hop 1
    surfaced), then take the top-k2 additional sentences not already picked.
    """
    if not sentences:
        empty = HopTrace(query=question, retrieved=[])
        return [], empty, empty

    all_texts = [s.text for s in sentences]
    sent_embs = embed_texts(all_texts, client)

    q_emb = embed_texts([question], client)[0]
    sims1 = [cosine(q_emb, e) for e in sent_embs]
    order1 = sorted(range(len(sentences)), key=lambda i: sims1[i], reverse=True)
    hop1_idx = order1[:hop1_k]
    hop1_sents = [sentences[i] for i in hop1_idx]
    hop1_trace = HopTrace(query=question, retrieved=hop1_sents)

    hop2_query = question + " " + " ".join(s.text for s in hop1_sents)
    q2_emb = embed_texts([hop2_query], client)[0]
    sims2 = [cosine(q2_emb, e) for e in sent_embs]
    remaining = [i for i in range(len(sentences)) if i not in set(hop1_idx)]
    order2 = sorted(remaining, key=lambda i: sims2[i], reverse=True)
    hop2_idx = order2[:hop2_k]
    hop2_sents = [sentences[i] for i in hop2_idx]
    hop2_trace = HopTrace(query=hop2_query, retrieved=hop2_sents)

    fed = hop1_sents + hop2_sents
    return fed, hop1_trace, hop2_trace


# --------------------------------------------------------------------------
# Answer generation
# --------------------------------------------------------------------------


def build_chat_model(model_name: str) -> Any:
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model_name, temperature=0)
    return llm.with_structured_output(CitedAnswer)


def answer_with_citations(structured_llm: Any, question: str, fed: list[Sentence]) -> CitedAnswer:
    lines = [f"[{i}] (title=\"{s.title}\", sentence_index={s.sent_idx}): {s.text}" for i, s in enumerate(fed)]
    context_block = "\n".join(lines)
    prompt = (
        "You are answering a multi-hop question using ONLY the numbered context sentences below. "
        "Each sentence is labeled with its exact (title, sentence_index) pair.\n\n"
        f"Context sentences:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Give a short, direct answer. You MUST also populate supporting_facts with at least one "
        "(title, sentence_index) pair copied exactly from the numbered context above -- these are "
        "the sentences you actually relied on. Do not invent titles or indices that are not listed "
        "above, and do not leave supporting_facts empty."
    )
    result = structured_llm.invoke(prompt)
    if not isinstance(result, CitedAnswer):
        # Some structured-output paths return a dict-like object.
        result = CitedAnswer.model_validate(result)
    return result


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------

_ARTICLES = {"a", "an", "the"}


def normalize_answer(s: str) -> str:
    s = s.lower()
    s = "".join(ch for ch in s if ch not in string.punctuation)
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def exact_match(pred: str, gold: str) -> bool:
    return normalize_answer(pred) == normalize_answer(gold)


def f1_score(pred: str, gold: str) -> float:
    pred_toks = normalize_answer(pred).split()
    gold_toks = normalize_answer(gold).split()
    if not pred_toks or not gold_toks:
        return float(pred_toks == gold_toks)
    common: dict[str, int] = {}
    for t in pred_toks:
        common[t] = common.get(t, 0) + 1
    num_same = 0
    gold_counts: dict[str, int] = {}
    for t in gold_toks:
        gold_counts[t] = gold_counts.get(t, 0) + 1
    for t, c in gold_counts.items():
        num_same += min(c, common.get(t, 0))
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def prf(predicted: set, gold: set) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted:
        return 0.0, 0.0, 0.0
    if not gold:
        return 0.0, 0.0, 0.0
    tp = len(predicted & gold)
    precision = tp / len(predicted)
    recall = tp / len(gold)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def score_example(
    record: ExampleRecord,
    fed: list[Sentence],
    result: CitedAnswer,
) -> ExampleResult:
    fed_keys = {s.key for s in fed}
    claimed_keys = {(f.title, f.sentence_index) for f in result.supporting_facts}
    gold_keys = record.gold_facts

    em = exact_match(result.answer, record.gold_answer)
    f1 = f1_score(result.answer, record.gold_answer)

    retrieval_recall = (
        len(fed_keys & gold_keys) / len(gold_keys) if gold_keys else 1.0
    )

    cg_p, cg_r, cg_f1 = prf(claimed_keys, gold_keys)
    cf_p, cf_r, _ = prf(claimed_keys, fed_keys)

    fabricated = sorted(claimed_keys - fed_keys)

    return ExampleResult(
        qid=record.qid,
        question=record.question,
        gold_answer=record.gold_answer,
        model_answer=result.answer,
        gold_facts=[list(k) for k in sorted(gold_keys)],
        fed_facts=[list(k) for k in sorted(fed_keys)],
        claimed_facts=[list(k) for k in sorted(claimed_keys)],
        hop1_query=record.question,
        hop2_query="",
        answer_em=em,
        answer_f1=f1,
        retrieval_recall_of_gold=retrieval_recall,
        claim_vs_gold_precision=cg_p,
        claim_vs_gold_recall=cg_r,
        claim_vs_gold_f1=cg_f1,
        claim_vs_fed_precision=cf_p,
        claim_vs_fed_recall=cf_r,
        fabricated_citations=[list(k) for k in fabricated],
    )


def pick_chat_model(available: list[str]) -> str:
    if CHAT_MODEL_PREFERRED in available:
        return CHAT_MODEL_PREFERRED
    for m in CHAT_MODEL_FALLBACKS:
        if m in available:
            return m
    if available:
        return available[0]
    raise RuntimeError("No Ollama chat models available.")


def list_ollama_models(client: httpx.Client) -> list[str]:
    resp = client.get(f"{OLLAMA_URL}/api/tags", timeout=10)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
