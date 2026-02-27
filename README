# Multilingual Uncertainty & Language Specialization Analysis

Code for the paper *"[Paper Title]"*.

This repository provides a pipeline for evaluating large language models across 14 languages, measuring tokenization quality, neuron-level language specialization (LAPE), and multiple families of uncertainty metrics.

## Languages

| Turkic (Latin) | Turkic (Cyrillic) | High-resource |
|---|---|---|
| Azerbaijani | Crimean Tatar (Cyrillic) | English |
| Crimean Tatar | Kazakh | Russian |
| Karakalpak | Tatar | Turkish |
| Kazakh (Latin) | Uzbek (Cyrillic) | |
| Uyghur (Latin) | Uyghur | |
| Uzbek | | |

## Metrics

### Tokenization (`tokenizer.py`)

| Metric | Description |
|---|---|
| **Fertility** | Tokens per word — measures tokenizer efficiency for a given language |
| **Unique token fraction** | Share of the full vocabulary used by a language |
| **Shared token fraction** | Token overlap with English, Russian, and Turkish |
| **Chars per token** | Mean/std of token string length |

### Language-Specific Neurons (`lape.py`)

Identifies neurons in MLP layers that activate preferentially for specific languages. For each neuron the activation probability is computed per language, normalized, and scored by Shannon entropy. Low-entropy neurons with high activation probability are assigned to the corresponding language.

**Hyperparameters:** `TOP_RATE=0.01`, `FILTER_RATE=0.95`, `ACTIVATION_BAR_RATIO=0.95`.

### Logit-Based Uncertainty (`logit.py`)

| Metric | Formula | Description |
|---|---|---|
| **MeanTokenNLL** | $-\frac{1}{T}\sum_i \log p(y_i \mid y_{<i}, x)$ | Mean negative log-likelihood per generated token |
| **SequenceNLL** | $-\sum_i \log p(y_i \mid y_{<i}, x)$ | Total sequence negative log-likelihood |
| **MeanTokenEntropy** | $\frac{1}{T}\sum_i H\bigl(p(\cdot \mid y_{<i}, x)\bigr)$ | Mean entropy of the predictive distribution |

### Attention-Based Uncertainty (`attention.py`)

| Metric | Description |
|---|---|
| **RAUQ** | Recurrent Attention-based Uncertainty Quantification. Selects the best attention head per layer (middle third), computes recurrent confidence $c_i = \alpha\, p_i + (1-\alpha)\, a_{i,i-1}\, c_{i-1}$, and returns $\max_l\bigl(-\overline{\log c}\bigr)$ |
| **Focus** | IDF-weighted probability correction with attention-based penalty propagation between keyword tokens (top IDF quantile). Sentence-level score is the mean over keywords |

**Hyperparameters:** `RAUQ_ALPHA=0.2`, `FOCUS_GAMMA=0.9`, `FOCUS_RHO=0.01`, `FOCUS_KW_IDF_QUANTILE=0.75`.

### Sampling-Based Uncertainty (`divercity.py`)

All metrics are computed over `N=10` stochastic samples per question (`temperature=0.9`, `top_p=0.95`).

| Metric | Description |
|---|---|
| **LexicalSimilarity** | Mean pairwise (1 − BLEU) distance across samples |
| **DegMat** | Degree-matrix uncertainty on a Jaccard similarity graph |
| **EigValLaplacian** | Frobenius norm of the normalized Laplacian eigenvalues |
| **Eccentricity** | Mean graph eccentricity via Floyd–Warshall on spectral distances |

### Accuracy (`evaluate.py`)

Standard multiple-choice accuracy with deterministically shuffled answer options and JSON-structured model output.

## Data Format

Each language directory contains `all_shuffled.jsonl` with one JSON object per line:

```json
{"question": "...", "choices": ["A_text", "B_text", "C_text", "D_text"], "answer": "B", "subject": "geography"}
```

## Token Budget Equalization

To ensure fair cross-lingual comparison, every script:
1. **Pass 1** — tokenizes the full corpus for each language and records the total token count.
2. Computes `common_tokens = min(total_tokens)` across all languages.
3. **Pass 2** — processes questions sequentially until the accumulated token count reaches the budget.

This guarantees that each language is evaluated on an equal amount of textual material.

## Output Structure

```
{output_base}/{model_id}/
├── lape/
│   ├── activation_stats.pt
│   ├── lang_specific_neurons.pt
│   └── lape_summary.tsv
├── tokenizer/
│   └── tokenizer_summary.tsv
├── nll_entropy/
│   ├── {lang}_nll_entropy.jsonl
│   └── nll_entropy_summary.tsv
├── rauq_focus/
│   ├── {lang}_rauq_focus.jsonl
│   ├── rauq_focus_summary.tsv
│   └── idf_cache/
├── graph_metrics/
│   ├── {lang}_graph_metrics.jsonl
│   └── graph_metrics_summary.tsv
└── evaluate/
    ├── {lang}_eval_1.jsonl
    └── evaluate_summary.tsv
```

All per-question results are saved incrementally in JSONL format, enabling checkpoint-based resumption.

## Supported Architectures

| Architecture | Models |
|---|---|
| LLaMA / Mistral | Llama-3.x, Mistral |
| Qwen2 | Qwen2.5-xB-Instruct |
| Gemma 2 / 3 | gemma-2-9b-it, gemma-3-{4,12,27}b-it |
| GPT-2 | GPT-2 family |
| BLOOM | BLOOM family |

## Requirements

```
torch
transformers
numpy
```

## Usage

```bash
./run_scripts.sh <model_id>
```

For example:

```bash
./run_scripts.sh google/gemma-3-12b-it
```

This runs all six scripts sequentially: `lape.py` → `tokenizer.py` → `logit.py` → `attention.py` → `divercity.py` → `evaluate.py`.

To run a single metric:

```bash
export MODEL_ID=google/gemma-3-12b-it
python3 logit.py
```

Environment variables `HF_HUB_OFFLINE=1` and `TRANSFORMERS_OFFLINE=1` should be set if the model is pre-downloaded.
