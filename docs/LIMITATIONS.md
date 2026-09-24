# Limitations

[<- back to README](../README.md)

## Small samples throughout

| Project | n |
|---|---|
| 02 HotpotQA | 50 questions |
| 03 BFCL | 140 cases per model |
| 04 SWE-bench | **2 instances** |
| 05 TruthfulQA | 100 questions per cell |
| 06 FEVER | 60 claims |

Project 04's two instances are the weakest. Two failures establish that the pipeline runs
end to end and that *these* patches were malformed - **not** a rate at which the model
produces invalid diffs.

## Single seed, single run

Results come from one run at a fixed seed. No variance estimates, no confidence intervals.
A difference of a few points between two models in project 03 should not be read as
meaningful without repeats.

## Half the projects top out at 7B

Projects **01, 02 and 06** use models of 7B parameters or fewer, and nothing in them says
how a larger model behaves.

**03, 04 and 05 have since been run against `qwen2.5-coder:14b`** - see the
[README section dated 2026-09-16](../README.md#2026-09-16--qwen25-coder14b-added-to-three-projects).
This section used to name project 05's chain-of-thought result as the one most likely to
reverse with capability. It does not reverse: the 14B gains 20 points on truthfulness over
the 3B and tops the MC2 table, and chain-of-thought still costs it accuracy. The scale
caveat was real and is now answered for that finding specifically, which is a different
thing from it being answered generally - 4.8x is one step, and these are all one family.

## Project 06 uses live Wikipedia

Evidence is retrieved from the live MediaWiki API, not a frozen dump. That makes the
pipeline realistic and the result **not exactly reproducible** - Wikipedia changes. The
project's README discusses this trade explicitly.

## Project 05's MC2 is an approximation

MC2 is computed by a heuristic rather than TruthfulQA's official scoring. The direction of
the effect matches MC1, but the absolute MC2 values should not be compared against published
numbers.

## Prompt sensitivity is not measured

Each project uses one prompt formulation. Project 05 demonstrates how much prompting
matters - a 35-point swing from one change - which means every other project's numbers are
also prompt-dependent in ways not quantified here.

## Project 01's red-team modules are self-built

The five audit modules are written for this project, not drawn from a published red-team
suite. The before/after comparison is internally valid - same tools, same payloads, hardening
toggled - but the coverage is not benchmarked against anything.
