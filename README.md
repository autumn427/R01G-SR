# R01G-SR

Research code for **R01G-SR: Provenance-Preserving Selection over Complementary Evidence Views for Crisis Situation Report Generation**, LT4CPR at AACL-IJCNLP 2026.

Two report candidates share a generator: R0 presents chronological evidence, while R1 filters and compresses evidence. A frozen logistic selector chooses one whole report; deterministic normalization, section-ownership repair, and evidence sanitation enforce structural constraints. Selection does not synthesize new claims or combine reports.

## Release scope

This initial source release contains candidate generation, feature extraction, grouped selector training and application, structure repair, and submission sanitation. It is an extraction of existing research code, not a new implementation or a newly evaluated model. `SOURCE_MANIFEST.json` records the original source-file SHA-256 values.

Data, social-media messages, gold reports, generated reports, model weights, experiment logs, credentials, and historical evaluation environments are **not redistributed**. Frozen selector coefficients are not included in this initial release; train a selector using permitted training data and cached training scores. This is not yet a one-command reproduction of the submitted ZIP. Historical scoring helpers are retained as dependencies of selector training; for current benchmark results use the organizer's released evaluator, not the historical wrapper as a substitute.

## Setup

Python 3.11 or newer is recommended. Install `requirements.txt` in a separate environment. The listed dependency ranges are compatibility requirements, not a lockfile of the historical environment.

Obtain task data and evaluation tools from the [official task repository](https://github.com/LT4CPR/2026_aacl_workshop), respecting its terms. The recorded test-release snapshot is `78dbff586d391d97ec4d11310cf587f43c57ca2a`; earlier research runs used earlier snapshots.

Obtain the [Qwen3-4B-Instruct-2507 model](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) separately and serve it with an OpenAI-compatible local endpoint. Principal experiments used a UD-Q4_K_XL GGUF conversion. No model download, server startup, training, or GPU work is performed by the smoke tests.

## Candidate generation

From the repository root, with released cell files under `data/dev` and an already running local inference server:

```bash
python -m src.run --data data/dev --out outputs/dev --runs R0,R1 --base-url http://127.0.0.1:8080/v1 --model Qwen3-4B-Instruct-2507
python -m src.postprocess_outputs --source outputs/dev --out outputs/normalized --runs R0,R1
python -m src.build_selector_metadata --data data/dev --source outputs/normalized --r0-name R0 --r1-name R1 --out outputs/metadata
```

`src.run` also retains historical R2-R4 options; they are not the submitted primary method. Do not use its default route list when intending only R0/R1.

## Selector

See CLI help for the training-score payload and metadata inputs:

```bash
python -m src.train_grouped_selector --help
python -m src.apply_grouped_selector --help
```

Train only on permitted training scores, grouping all windows and replicates from each crisis together. Never use hidden-test references to fit or select a model. Apply a frozen selector with:

```bash
python -m src.apply_grouped_selector --data data/dev --model artifacts/selector.json --selector-metadata outputs/metadata --source outputs/normalized --out outputs/selected
```

`src.repair_section_ownership.repair(report)` returns a repaired copy and a move log. It moves only subsection titles with exactly one legal owner; ambiguous titles stay unchanged. Apply it before final sanitation. Sanitation and validation are available through:

```bash
python -m src.prepare_submission sanitize --data data/dev --reports outputs/repaired --out outputs/clean
python -m src.prepare_submission validate --data data/dev --reports outputs/clean
```

The historical packaging subcommand is not a substitute for the latest official team-ZIP validator. Follow the organizer's current submission specification.

## Checks

```bash
python -m unittest discover -s tests -v
python -m compileall -q src tests
```

These checks cover source compilation and synthetic structural invariants only, not model quality or end-to-end benchmark reproduction.

## Official result

Team **gogogo**, primary system **r01g-sr**: official PRIMARY 0.5273, BERTScore F1 0.5018, BLEURT 0.5528; sixth of nine teams. PRIMARY is the mean of BERTScore F1 and BLEURT. These are organizer-reported historical scores, not scores produced by running this release's checks. See the [official leaderboard](https://lt4cpr.github.io/aacl2026-workshop-LT4CPR/leaderboard.html).

## Responsible use

Evidence IDs support traceability, not truth verification. This is research software, not an operational crisis-response system. Human review is required before consequential use. Third-party data and model licenses remain applicable. No additional software license is selected in this initial publication.
