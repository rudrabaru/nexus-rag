"""
Synthetic test sets: questions an LLM writes from chunks of an ingested corpus, so each one has
exact ground truth (the chunk it was written from), reviewed by a person and frozen as an
evaluation dataset (src/evaluation/dataset.py).

    python -m src.testsets generate --tenant T --count 50     # chunks -> a draft of questions
    python -m src.testsets review DRAFT.json                  # accept, edit or reject each question
    python -m src.testsets finalize DRAFT.json DATASET.json   # accepted questions -> the dataset
    python -m src.testsets verify DATASET.json --tenant T     # is the ground truth still in the index?

See docs/phases/phase6_evaluation.md.
"""
