"""Tests for projects/02_hotpotqa_multihop_rag.

Pure-logic tests (scoring, normalization, similarity math) run always.
Tests that need a live Ollama server or a live download from the Hugging
Face Hub are marked `@pytest.mark.live` (deselect with `-m "not live"`),
matching the project-wide convention in pyproject.toml.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

PROJECT_DIR = Path(__file__).parent.parent / "projects" / "02_hotpotqa_multihop_rag"
sys.path.insert(0, str(PROJECT_DIR))

from pipeline import (  # noqa: E402
    CitedAnswer,
    ExampleRecord,
    Sentence,
    SupportingFact,
    cosine,
    exact_match,
    f1_score,
    normalize_answer,
    pick_chat_model,
    prf,
    score_example,
)


# --------------------------------------------------------------------------
# Pure logic -- no network, no Ollama
# --------------------------------------------------------------------------


def test_normalize_answer_strips_articles_and_punctuation():
    assert normalize_answer("The Eiffel Tower.") == "eiffel tower"
    assert normalize_answer("A dog") == "dog"


def test_exact_match_is_normalization_insensitive():
    assert exact_match("The Beatles", "beatles")
    assert not exact_match("The Beatles", "The Rolling Stones")


def test_f1_score_partial_overlap():
    score = f1_score("New York City", "New York")
    assert 0.0 < score < 1.0
    assert f1_score("Paris", "Paris") == 1.0
    assert f1_score("Paris", "London") == 0.0


def test_cosine_similarity_basic():
    assert cosine([1, 0], [1, 0]) == pytest.approx(1.0)
    assert cosine([1, 0], [0, 1]) == pytest.approx(0.0)
    assert cosine([1, 0], [-1, 0]) == pytest.approx(-1.0)


def test_cosine_similarity_zero_vector_is_safe():
    assert cosine([0, 0], [1, 1]) == 0.0


def test_prf_perfect_match():
    p, r, f1 = prf({("A", 0), ("B", 1)}, {("A", 0), ("B", 1)})
    assert (p, r, f1) == (1.0, 1.0, 1.0)


def test_prf_partial_and_empty_cases():
    p, r, f1 = prf({("A", 0)}, {("A", 0), ("B", 1)})
    assert p == 1.0
    assert r == 0.5
    # predicted nonempty, gold empty -> 0s (can't have precision against nothing)
    assert prf({("A", 0)}, set()) == (0.0, 0.0, 0.0)
    # both empty -> perfect trivial agreement
    assert prf(set(), set()) == (1.0, 1.0, 1.0)


def test_pick_chat_model_prefers_qwen_7b_instruct():
    assert pick_chat_model(["llama3.2:3b", "qwen2.5:7b-instruct"]) == "qwen2.5:7b-instruct"


def test_pick_chat_model_falls_back_when_preferred_missing():
    assert pick_chat_model(["granite3.3:2b", "llama3.2:3b"]) == "llama3.2:3b"


def test_score_example_flags_fabricated_citation():
    record = ExampleRecord(
        qid="q1",
        question="Where is the tower?",
        gold_answer="Paris",
        sentences=[Sentence("Eiffel Tower", 0, "The Eiffel Tower is in Paris.")],
        gold_facts={("Eiffel Tower", 0)},
    )
    fed = [Sentence("Eiffel Tower", 0, "The Eiffel Tower is in Paris.")]
    # Model cites a sentence index that was never fed to it.
    cited = CitedAnswer(
        answer="Paris",
        supporting_facts=[
            SupportingFact(title="Eiffel Tower", sentence_index=0),
            SupportingFact(title="Eiffel Tower", sentence_index=7),
        ],
    )
    result = score_example(record, fed, cited)
    assert result.answer_em is True
    assert result.fabricated_citations == [["Eiffel Tower", 7]]
    assert result.claim_vs_gold_recall == 1.0  # gold fact (idx 0) was cited
    assert result.claim_vs_fed_precision == 0.5  # only 1 of 2 claims was actually fed


def test_score_example_perfect_retrieval_and_citation():
    record = ExampleRecord(
        qid="q2",
        question="What color is the sky?",
        gold_answer="blue",
        sentences=[Sentence("Sky", 0, "The sky is blue.")],
        gold_facts={("Sky", 0)},
    )
    fed = [Sentence("Sky", 0, "The sky is blue.")]
    cited = CitedAnswer(answer="Blue", supporting_facts=[SupportingFact(title="Sky", sentence_index=0)])
    result = score_example(record, fed, cited)
    assert result.answer_em is True
    assert result.retrieval_recall_of_gold == 1.0
    assert result.claim_vs_gold_f1 == 1.0
    assert result.claim_vs_fed_precision == 1.0
    assert result.fabricated_citations == []


# --------------------------------------------------------------------------
# Live: requires a running local Ollama server (nomic-embed-text pulled)
# --------------------------------------------------------------------------


@pytest.mark.live
def test_embed_texts_returns_correct_shape():
    import httpx

    from pipeline import embed_texts

    with httpx.Client() as client:
        embs = embed_texts(["hello world", "goodbye world"], client)
    assert len(embs) == 2
    assert len(embs[0]) > 0
    assert len(embs[0]) == len(embs[1])


@pytest.mark.live
def test_two_hop_retrieve_returns_requested_counts():
    import httpx

    from pipeline import two_hop_retrieve

    sentences = [
        Sentence("A", 0, "Paris is the capital of France."),
        Sentence("A", 1, "France is in Europe."),
        Sentence("B", 0, "The Eiffel Tower is located in Paris."),
        Sentence("B", 1, "It was completed in 1889."),
        Sentence("C", 0, "Berlin is the capital of Germany."),
    ]
    with httpx.Client() as client:
        fed, hop1, hop2 = two_hop_retrieve(
            "In which country is the Eiffel Tower?", sentences, client, hop1_k=2, hop2_k=2
        )
    assert len(hop1.retrieved) == 2
    assert len(hop2.retrieved) <= 2
    assert len(fed) == len(hop1.retrieved) + len(hop2.retrieved)
    # hop1 and hop2 selections must not overlap
    assert {s.key for s in hop1.retrieved}.isdisjoint({s.key for s in hop2.retrieved})


@pytest.mark.live
def test_list_ollama_models_reachable():
    import httpx

    from pipeline import list_ollama_models

    with httpx.Client() as client:
        models = list_ollama_models(client)
    assert isinstance(models, list)
    assert len(models) > 0


@pytest.mark.live
def test_answer_with_citations_end_to_end():
    import httpx

    # The `live` marker covers the server being up; it does not cover the client
    # library being installed, and this is the only test in the file that needs
    # one. A machine with Ollama running but no langchain got ModuleNotFoundError
    # instead of a skip.
    pytest.importorskip("langchain_ollama", reason="langchain-ollama is not installed")

    from pipeline import answer_with_citations, build_chat_model, list_ollama_models, pick_chat_model

    with httpx.Client() as client:
        model_name = pick_chat_model(list_ollama_models(client))
    structured_llm = build_chat_model(model_name)
    fed = [Sentence("Sky", 0, "The sky is blue on a clear day.")]
    result = answer_with_citations(structured_llm, "What color is the sky?", fed)
    assert isinstance(result, CitedAnswer)
    assert result.answer


@pytest.mark.live
def test_load_hotpotqa_sample_real_dataset():
    from pipeline import load_hotpotqa_sample

    records = load_hotpotqa_sample(2, seed=42)
    assert len(records) == 2
    for r in records:
        assert r.question
        assert r.gold_answer
        assert len(r.sentences) > 0
        assert len(r.gold_facts) > 0
