# data/

Local datasets used by the pipeline. Not checked in — download/prepare these yourself and place
them under the paths the scripts expect (see `scripts/*.sh` for exact paths used in each run).

Expected content:

- **BeaverTails** (`tool_dataset/beavertails/...`): harmful/benign prompt-response pairs used to
  train the safety probe (`src/representations.py`, `src/probe.py`) and as the fine-tuning corpus
  that gets poisoned at varying harmful-data ratios (`src/sft.py`). See
  [PKU-Alignment/BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails).
- **HEx-PHI** (`evaluation_datasets/HEx-PHI-full/category_{1..11}.csv`): 11-category harmful
  instruction benchmark used to evaluate fine-tuned models' safety (`src/inference.py`,
  `scripts/inference*.sh`, `scripts/gpt_eval*.sh`).

Extracted activation vectors, generations, and other derived artifacts produced from this data
are written to `output_data/`, `evaluation_results/`, and `evaluation_outputs/` respectively.
