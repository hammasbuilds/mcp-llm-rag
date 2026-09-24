# Future work

[<- back to README](../README.md)

## 1. A larger-model comparison on the remaining three

`qwen2.5-coder:14b` has since been run on **03, 04 and 05**; the
[README section dated 2026-09-16](../README.md#2026-09-16--qwen25-coder14b-added-to-three-projects)
carries those results. **01, 02 and 06 still top out at 7B**, and each is written so a
larger model adds a **comparison row** rather than requiring a rewrite.

The two questions this section used to pose have answers, and they are worth stating
because both turned out to be more specific than the question:

- **Project 04** was "capability or prompting?". Neither, exactly. The 3B emitted diffs
  `git` could not parse; the 14B emits diffs that parse perfectly and cite line 1234 for
  code at line 1042. Scaling fixed the syntax and not the grounding.
- **Project 05** was "does chain-of-thought reverse with scale?". It does not. The 14B
  gains 20 points over the 3B on truthfulness and tops the MC2 table, and chain-of-thought
  still costs accuracy.

**Project 02 first** of the remainder: its gap is retrieval finding the gold facts 83.6% of
the time while exact-match sits at 0.20, which is the shape most likely to move with a
model that can use what retrieval already handed it.

## 2. More instances for project 04

Two instances is enough to show the pipeline works and that these two patches were
malformed. It is not enough to state a rate. Twenty would be.

## 3. Repeats and variance

Every number is a single run at a fixed seed. Three to five repeats per cell would turn
point estimates into intervals, and would show whether the smaller gaps in project 03 are
real.

## 4. Prompt sensitivity as a measured variable

Project 05 shows a 35-point swing from one prompting change. That means every other
project's numbers are prompt-dependent in ways currently unquantified - and the same
generate-once/score-many design used elsewhere would measure it.

## 5. Fix the diff formatting in project 04

The model targeted the right files and produced plausible patches. Constrained decoding, a
diff-validating retry loop, or asking for whole files instead of diffs would each test a
different hypothesis about *why* the formatting failed.

## 6. Push the red-team modules further

Project 01 found that delivery vector matters more than payload phrasing. That suggests
enumerating vectors systematically - tool descriptions, tool results, document content,
filenames, error messages - rather than enumerating payloads.

## 7. Official scoring for project 05's MC2

Currently a heuristic. Using TruthfulQA's published scoring would make the MC2 numbers
comparable to the literature, rather than only internally consistent.
