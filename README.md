# [paper title]

A pipeline for probing large language models across 14 languages, measuring tokenization quality, neuron-level language specialization, and multiple families of uncertainty metrics.

## Data

Questions are drawn from two benchmarks:

- **[TUMLU](https://arxiv.org/abs/2502.11020)** (Isbarov et al., 2025) — a natively developed (non-translated) multiple-choice benchmark for Turkic languages covering 8 languages, 11 academic domains, and 38 139 questions. Accepted to ACL 2025.
- **[Global MMLU](https://arxiv.org/abs/2412.03304)** (Singh et al., 2024) — a culturally aware extension of MMLU spanning 42 languages with professional translation verification. Used here for English and Russian subsets.

Four TUMLU languages appear in two scripts, giving 14 language–script combinations in total:

| Turkic (Latin) | Turkic (Cyrillic) | Turkic (Arabic) | High-resource |
|---|---|---|---|
| Azerbaijani | Crimean Tatar (Cyrillic) | Uyghur | English |
| Crimean Tatar | Kazakh | | Russian |
| Karakalpak | Tatar | | Turkish |
| Kazakh (Latin) | Uzbek (Cyrillic) | | |
| Uyghur (Latin) | | | |
| Uzbek | | | |

### Data format

Each language directory contains `all_shuffled.jsonl` with one JSON object per line:

```json
{"question": "...", "choices": ["A_text", "B_text", "C_text", "D_text"], "answer": "B", "subject": "geography"}
```

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

For MoE architectures (e.g. DeepSeek V3), shared experts are tracked since they are active for every token.

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

## Cross-Lingual Budget Equalization

To ensure fair comparison across languages, each script equalizes the amount of data processed. Two strategies are used depending on the script:

### Token budget (`lape.py`, `tokenizer.py`)

These scripts operate on **raw prompts without a chat template** to measure intrinsic tokenizer and neuron properties unaffected by template overhead.

1. Tokenize the full corpus for each language → `total_tokens` per language.
2. Set `common_tokens = min(total_tokens)` across all languages.
3. Process questions sequentially until the accumulated token count reaches the budget.

### Question budget (`logit.py`, `attention.py`, `divercity.py`)

These scripts use the **chat template** (if available) for model inference, so the budget is equalized by number of questions to guarantee the same sample size per language.

1. Tokenize the full corpus (with chat template) → `total_tokens` per language.
2. Set `common_tokens = min(total_tokens)` across all languages.
3. For each language, count how many questions fit within `common_tokens` → `questions_per_lang`.
4. Set `common_questions = min(questions_per_lang)` across all languages.
5. Process exactly `common_questions` questions per language.

### Accuracy (`evaluate.py`)

Evaluates on the full dataset for each language (no truncation).

## Checkpoint & Resume

All per-question results are saved incrementally in JSONL format. If a run is interrupted, re-running the same script will skip already-completed items automatically. `lape.py` and `tokenizer.py` save binary/JSON checkpoints after each language and remove them upon successful completion.

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
    ├── {lang}_eval.jsonl
    └── evaluate_summary.tsv
```

## Supported Architectures

| Architecture | `model_type` | Models |
|---|---|---|
| LLaMA / Mistral | `llama`, `mistral` | Llama-3.x, Mistral |
| Qwen 2 | `qwen2` | Qwen2.5-xB-Instruct |
| Gemma 2 / 3 | `gemma2`, `gemma3`, `gemma3_text` | gemma-2-9b-it, gemma-3-{4,12,27}b-it |
| DeepSeek V3 (MoE) | `deepseek_v3` | GigaChat3-10B-A1.8B |
| Qwen 3.5 MoE | `qwen3_5_moe` | Qwen3.5-35B-A3B |
| GPT-2 | `gpt2` | GPT-2 family |
| BLOOM | `bloom` | BLOOM family |

Architecture-specific logic is only required in `lape.py` (layer access, MLP hook registration). All other scripts use `AutoModelForCausalLM` without architecture-specific code.

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
