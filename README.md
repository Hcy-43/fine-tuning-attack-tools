# Fine-Tuning Attack Tools

Research toolkit for studying **harmful fine-tuning attacks** on aligned LLMs, and for building a
**representation-based defense** that filters poisoned examples out of a fine-tuning set before
training ever happens.

It is well documented that mixing even a small fraction of harmful examples into an otherwise
benign fine-tuning dataset can strip away a model's RLHF safety alignment (Qi et al., 2023,
*"Fine-tuning aligned language models compromises safety, even when users do not intend to!"*).
This repo implements the full experimental loop around that problem: poison a fine-tuning set at
increasing ratios, fine-tune, measure how harmful the resulting model becomes, and compare that
against models fine-tuned on data that has been filtered by a lightweight safety classifier
trained on the base model's own hidden-state activations.

## Headline result

Probing a single mid-network layer of Llama-2-7B-chat's hidden states and training a
PCA + logistic-regression classifier on it is enough to catch most harmful fine-tuning examples
before training. Judged by GPT-4 on the HEx-PHI harmful-instruction benchmark (1–5 harm score,
higher = worse), at a 30% harmful-data poisoning ratio:

| Filtering strategy         | Avg. harm score | % responses scored maximally harmful (5/5) |
|-----------------------------|:---------------:|:---------:|
| No filtering (baseline)     | 4.43 / 5         | 77.9%     |
| Random filtering            | 3.90 / 5         | 64.2%     |
| Neural-net probe filter     | 2.28 / 5         | 25.2%     |
| **PCA + LogReg probe filter** | **1.82 / 5**   | **14.2%** |

Full per-ratio breakdowns are in `evaluation_outputs/*/GPT_result/summary.md` and are plotted in
`evaluation_outputs/read.ipynb`.

A layer sweep (`results/layer_sweep_results.csv`, visualized in `results/read.ipynb`) shows probe
accuracy for separating harmful vs. benign examples peaks around the middle-to-late decoder
layers (~82% accuracy, ~84% F1 at layer -14 to -16 of Llama-2-7B-chat), and degrades sharply in
the final few layers as representations specialize for next-token prediction.

## How it works

```
BeaverTails (harmful/benign pairs)
        │
        ▼
1. Extract hidden-state representations at a chosen layer   (src/representations.py)
        │
        ▼
2. Train a probe/classifier to separate harmful vs. benign   (src/probe.py,
   representations, sweep layers to find the best one         notebooks/probing_classifier.ipynb)
        │
        ▼
3. Use the probe to filter a poisoned fine-tuning set,        (streamlit/frontend.py,
   at varying harmful-data ratios                              notebooks/data_filtering.ipynb)
        │
        ▼
4. LoRA fine-tune Llama-2-7B-chat on filtered / unfiltered /  (src/sft.py, scripts/sft.sh)
   randomly-filtered data at each ratio
        │
        ▼
5. Generate responses to the HEx-PHI harmful-instruction      (src/inference.py,
   benchmark (11 categories) with each fine-tuned model        scripts/inference*.sh)
        │
        ▼
6. Score every response for harmfulness with a GPT-4 judge    (src/gpt4_eval.py,
   and aggregate into per-category summaries                   src/score_summary.py)
        │
        ▼
7. Compare harmfulness vs. poison ratio across filtering       (evaluation_outputs/read.ipynb,
   strategies                                                   src/plot_layer_sweep.py)
```

## Repo structure

```
src/
  representations.py   Extract per-example hidden-state activations from a local LLaMA2/Vicuna model
  probe.py              Train + cross-validate a logistic-regression probe on extracted activations
  sft.py                LoRA fine-tuning (via TRL's SFTTrainer) on a prompt/response CSV
  inference.py           Batch generation from a base model + optional LoRA adapter
  gpt4_eval.py            GPT-4 "duo judge" harmfulness scoring of generated responses
  score_summary.py        Aggregate per-category GPT-4 scores into a markdown summary table
  plot_layer_sweep.py    Plot probe accuracy/precision/recall/F1 across layers

scripts/                Shell entry points that wire the src/ modules together for the
                         representation → SFT → inference → GPT-4-eval pipeline

streamlit/frontend.py   Upload a CSV of candidate fine-tuning data, extract representations,
                         and flag likely-harmful rows with a saved classifier before you train

notebooks/              Exploratory analysis: probing-classifier comparisons (logistic regression,
                         PCA+LR, random forest, gradient boosting, XGBoost), t-SNE/PCA
                         visualization of harmful vs. benign activation clusters, ROC/AUC probe
                         evaluation, and a data-filtering walkthrough.
                         quickstart_peft_finetuning.ipynb is Meta's unmodified llama-cookbook
                         reference notebook, kept for context.

results/                Layer-sweep probe metrics (CSV) for Llama-2-7B-chat and Vicuna-7B-1.5,
                         plus a notebook that plots them

evaluation_outputs/     GPT-4 judge summaries (avg. harm score, % max-harm) per filtering
                         strategy and poison ratio, plus a notebook comparing them

classifiers/, ft_models/, models/, data/, evaluation_results/, output_data/
                         Local working directories for trained classifiers, fine-tuned adapters,
                         base models, datasets, raw generations, and extracted activation vectors.
                         Not checked in (see each folder's README) — everything here is
                         regenerated by the scripts above.
```

## Setup

Requires Python 3.11 and a CUDA GPU for anything that loads a 7B model (representation
extraction, fine-tuning, inference). Dependencies are managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

For the GPT-4 judging step, set an OpenAI API key:

```bash
export OPENAI_KEY=sk-...
```

## Usage

Download base models (e.g. `meta-llama/Llama-2-7b-chat-hf`) into `models/`, and prepare a
BeaverTails-style CSV (`prompt`, `response` columns, split into harmful/benign) under `data/`.

**1. Extract activations and sweep layers to find the best probe layer:**

```bash
bash scripts/representation.sh
```

**2. Fine-tune (LoRA) at a range of harmful-data ratios:**

```bash
bash scripts/sft.sh
```

**3. Generate responses to the safety-evaluation benchmark:**

```bash
bash scripts/inference.sh          # models trained on randomly-filtered data
bash scripts/inference_pcalr.sh    # models trained on probe-filtered data
```

**4. Judge responses with GPT-4 and summarize:**

```bash
bash scripts/gpt_eval.sh
bash scripts/gpt_eval_nn.sh
```

**5. Compare strategies:** open `evaluation_outputs/read.ipynb`.

**6. Try the interactive filter:**

```bash
streamlit run streamlit/frontend.py
```

## Known limitations

- `src/gpt4_eval.py` imports an `eval_utils.openai_gpt4_judge` module (the GPT-4 harm-scoring
  rubric/prompt) that isn't included in this repo — it needs to be supplied separately (see the
  `safety_evaluation` harness in Qi et al.'s reference implementation).
- Scripts hardcode `CUDA_VISIBLE_DEVICES` for a specific multi-GPU dev box; adjust for your setup.
- Raw model generations and per-example GPT-4 transcripts are intentionally excluded from version
  control, since at high poison ratios they contain genuinely harmful model outputs — only
  aggregated scores are kept.

## License

MIT — see [LICENSE](LICENSE).
