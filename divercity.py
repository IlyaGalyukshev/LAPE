import warnings

warnings.filterwarnings("ignore")

import os
import json
import csv
import math
import gc
from datetime import datetime
from typing import List, Dict, Tuple, Any

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# CONFIG
# -----------------------------
MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-3-12b-it")
MODEL_PATH = os.environ.get("MODEL_PATH", "/hf_models")

DATA_ROOT = "/work/benchmarks/TUMLU"
OUTPUT_BASE = "/work/benchmarks/uncertainty_metrics"

# Generation parameters for sampling
SAMPLES_N = 10
MAX_NEW_TOKENS = 64
TEMPERATURE = 0.9
TOP_P = 0.95
TOP_K = 0  # 0 => disabled
REPETITION_PENALTY = 1.0

# Optional: greedy answer for logging (doesn't affect the metrics)
LOG_GREEDY_ANSWER = True
GREEDY_MAX_NEW_TOKENS = 64

# LexicalSimilarity mode:
LEXICAL_SIM_METRIC = "BLEU"


LANGS = [
    "azerbaijani",
    "crimean-tatar",
    "crimean-tatar-cyrillic",
    "en",
    "karakalpak",
    "kazakh",
    "kazakh-latin",
    "ru",
    "tatar",
    "turkish",
    "uyghur",
    "uyghur-latin",
    "uzbek",
    "uzbek-cyrillic",
]

PROMPTS = {
    "azerbaijani": """Sual: {question}\n{choices}\n\nCavabı bir cümlə ilə yazın.\nCavab: """,
    "crimean-tatar": """Sual: {question}\n{choices}\n\nCevapnı bir cümle ile yazıñ.\nCevap: """,
    "crimean-tatar-cyrillic": """Суал: {question}\n{choices}\n\nДжевапны бир джумле иле языныз.\nДжевап: """,
    "en": """Question: {question}\n{choices}\n\nAnswer in one sentence.\nAnswer: """,
    "karakalpak": """Soraw: {question}\n{choices}\n\nJuwaptı bir sózlem menen jazıń.\nJuwap: """,
    "kazakh": """Сұрақ: {question}\n{choices}\n\nЖауапты бір сөйлеммен жазыңыз.\nЖауап: """,
    "kazakh-latin": """Suraq: {question}\n{choices}\n\nJawapty bir sóılemmen jazyńyz.\nJawap: """,
    "ru": """Вопрос: {question}\n{choices}\n\nОтветьте одним предложением.\nОтвет: """,
    "tatar": """Сорау: {question}\n{choices}\n\nҖавапны бер җөмлә белән языгыз.\nҖавап: """,
    "turkish": """Soru: {question}\n{choices}\n\nCevabı tek bir cümleyle yazın.\nCevap: """,
    "uyghur": """سوئال: {question}\n{choices}\n\nجاۋابنى بىر جۈملە بىلەن يېزىڭ.\nجاۋاب: """,
    "uyghur-latin": """Soal: {question}\n{choices}\n\nJawabni bir jumle bilen yezing.\nJawab: """,
    "uzbek": """Savol: {question}\n{choices}\n\nJavobni bitta gap bilan yozing.\nJavob: """,
    "uzbek-cyrillic": """Савол: {question}\n{choices}\n\nЖавобни битта гап билан ёзинг.\nЖавоб: """,
}


# -----------------------------
# Logging helpers
# -----------------------------
def log(msg: str) -> None:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def safe_model_id(model_name: str) -> str:
    return model_name.replace("/", "__").replace(":", "_").replace(" ", "_")


def format_choices(choices: list) -> str:
    labels = ["A", "B", "C", "D"]
    out = []
    for i, c in enumerate(choices):
        if i < len(labels):
            out.append(f"{labels[i]}. {c}")
    return "\n".join(out)


# -----------------------------
# Tokenization / chat formatting
# -----------------------------
def make_user_prompt(lang: str, obj: Dict[str, Any]) -> str:
    formatted_choices = format_choices(obj["choices"])
    prompt_template = PROMPTS[lang]
    return prompt_template.format(question=obj["question"], choices=formatted_choices)


def apply_chat_if_available(tokenizer: AutoTokenizer, user_text: str) -> str:
    """
    If tokenizer has a chat template (Qwen Instruct does), wrap prompt into chat format.
    This significantly reduces "roleplay garbage" and helps EOS behavior.
    """
    try:
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            messages = [
                {"role": "user", "content": user_text},
            ]
            kwargs = dict(tokenize=False, add_generation_prompt=True)
            if "enable_thinking" in (tokenizer.chat_template or ""):
                kwargs["enable_thinking"] = False
            return tokenizer.apply_chat_template(messages, **kwargs)
    except Exception:
        pass
    return user_text


# -----------------------------
# Generation (sampling)
# -----------------------------
@torch.inference_mode()
def generate_grouped_samples(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_inputs_text: List[str],
    samples_n: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
    repetition_penalty: float,
) -> List[List[str]]:
    """
    Returns: list over batch, each is list[str] of length samples_n
    """
    enc = tokenizer(
        model_inputs_text,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
        model.device
    )

    # true prompt lengths per sample (avoid padding-length bug)
    prompt_lens = attention_mask.sum(dim=1).tolist()

    gen_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=1,
        num_return_sequences=samples_n,
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    if top_k and top_k > 0:
        gen_kwargs["top_k"] = int(top_k)

    # generate: (batch*samples_n, seq_len_total)
    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    # regroup outputs
    batch_size = len(model_inputs_text)
    grouped: List[List[str]] = [[] for _ in range(batch_size)]

    # out is stacked by prompt: for each input, it returns `samples_n` sequences
    # In HF generation, ordering is: [x0_s0, x0_s1, ... x0_sN, x1_s0, ...]
    for i in range(batch_size):
        base = i * samples_n
        p_len = int(prompt_lens[i])

        for j in range(samples_n):
            seq = out[base + j]
            gen_tokens = seq[p_len:]  # tokens after prompt

            # stop at first EOS
            if tokenizer.eos_token_id is not None:
                eos_positions = (gen_tokens == tokenizer.eos_token_id).nonzero(
                    as_tuple=False
                )
                if eos_positions.numel() > 0:
                    gen_tokens = gen_tokens[: int(eos_positions[0].item())]

            txt = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            grouped[i].append(txt)

    return grouped


@torch.inference_mode()
def generate_greedy_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_text: str,
    max_new_tokens: int,
) -> str:
    enc = tokenizer(
        model_input_text,
        return_tensors="pt",
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
        model.device
    )
    p_len = int(attention_mask.sum().item())

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        num_beams=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )[0]

    gen_tokens = out[p_len:]
    if tokenizer.eos_token_id is not None:
        eos_positions = (gen_tokens == tokenizer.eos_token_id).nonzero(as_tuple=False)
        if eos_positions.numel() > 0:
            gen_tokens = gen_tokens[: int(eos_positions[0].item())]

    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


# -----------------------------
# Similarity score
# -----------------------------
def compute_sim_score_jaccard(text1: str, text2: str) -> float:
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    union = tokens1.union(tokens2)
    if len(union) == 0:
        return 0.0
    inter = tokens1.intersection(tokens2)
    return float(len(inter) / len(union))


# -----------------------------
# LexicalSimilarity (BLEU)
# -----------------------------
def _count_ngrams(tokens: List[str], n: int) -> Dict[Tuple[str, ...], int]:
    from collections import Counter

    if n <= 0:
        return {}
    grams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
    return dict(Counter(grams))


def _modified_precision(reference: List[str], candidate: List[str], n: int) -> float:
    ref_counts = _count_ngrams(reference, n)
    cand_counts = _count_ngrams(candidate, n)
    if not cand_counts:
        return 0.0
    # clipped counts
    clip = 0
    total = 0
    for g, c in cand_counts.items():
        total += c
        clip += min(c, ref_counts.get(g, 0))
    if total == 0:
        return 0.0
    return clip / total


def _brevity_penalty(ref_len: int, cand_len: int) -> float:
    if cand_len == 0:
        return 0.0
    if cand_len > ref_len:
        return 1.0
    return math.exp(1.0 - (ref_len / cand_len))


def sentence_bleu_like_lm_polygraph(
    reference: List[str], candidate: List[str]
) -> float:
    """
    Matches the weight-selection logic used in lm-polygraph LexicalSimilarity for BLEU:
      min_len==1: [1,0,0,0]
      ==2: [0.5,0.5,0,0]
      ==3: [1/3,1/3,1/3,0]
      else: [0.25,0.25,0.25,0.25]
    (lm-polygraph calls nltk.sentence_bleu with these weights.)
    """
    ref_len = len(reference)
    cand_len = len(candidate)
    if cand_len == 0:
        return 0.0

    min_len = min(ref_len, cand_len)
    if min_len <= 0:
        return 0.0

    if min_len == 1:
        weights = [1.0, 0.0, 0.0, 0.0]
        max_n = 1
    elif min_len == 2:
        weights = [0.5, 0.5, 0.0, 0.0]
        max_n = 2
    elif min_len == 3:
        weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.0]
        max_n = 3
    else:
        weights = [0.25, 0.25, 0.25, 0.25]
        max_n = 4

    # geometric mean of modified precisions
    precisions = []
    for n in range(1, max_n + 1):
        p_n = _modified_precision(reference, candidate, n)
        precisions.append(p_n)

    # If any precision is 0, BLEU becomes 0
    if any(p == 0.0 for p in precisions):
        return 0.0

    s = 0.0
    for w, p in zip(weights, precisions + [1.0] * (4 - len(precisions))):
        if w > 0:
            s += w * math.log(p)

    bp = _brevity_penalty(ref_len, cand_len)
    return float(bp * math.exp(s))


def lexical_similarity_uncertainty(
    sample_texts: List[str], metric: str = "BLEU"
) -> float:
    n = len(sample_texts)
    if n < 2:
        return float("nan")

    dists = []
    if metric.upper() != "BLEU":
        raise ValueError(
            "This standalone script implements LexicalSimilarity only for metric='BLEU'."
        )

    for i in range(n):
        for j in range(i + 1, n):
            ref = sample_texts[i].split()
            cand = sample_texts[j].split()
            score = sentence_bleu_like_lm_polygraph(ref, cand)
            dists.append(1.0 - score)

    if not dists:
        return float("nan")
    return float(-np.mean(dists))


# -----------------------------
# Graph-based metrics
# DegMat / EigValLaplacian / Eccentricity using Jaccard_score path.
# -----------------------------
def build_W_jaccard(answers: List[str]) -> np.ndarray:
    n = len(answers)
    W = np.ones((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            s = compute_sim_score_jaccard(answers[i], answers[j])
            W[i, j] = s
            W[j, i] = s
    return W


def degmat_uncertainty(sample_texts: List[str]) -> float:
    n = len(sample_texts)
    if n == 0:
        return float("nan")
    W = build_W_jaccard(sample_texts)
    D = np.diag(np.sum(W, axis=1))
    return float(np.trace(n - D) / (n**2))


def eigvallaplacian_uncertainty(sample_texts: List[str]) -> float:
    n = len(sample_texts)
    if n == 0:
        return float("nan")
    W = build_W_jaccard(sample_texts)
    D = np.diag(np.sum(W, axis=1))
    D_sqrt_inv = np.linalg.inv(np.sqrt(D))
    L = np.eye(n) - D_sqrt_inv @ W @ D_sqrt_inv
    eigvals = np.linalg.eigvalsh(L)
    return float(np.sqrt(np.sum(eigvals**2)))


def floyd_warshall_all_pairs(dist: np.ndarray) -> np.ndarray:
    # dist: (n,n) with np.inf allowed
    n = dist.shape[0]
    d = dist.copy()
    for k in range(n):
        d = np.minimum(d, d[:, [k]] + d[[k], :])
    return d


def eccentricity_uncertainty(sample_texts: List[str], thres: float = 0.9) -> float:
    n = len(sample_texts)
    if n == 0:
        return float("nan")

    W = build_W_jaccard(sample_texts)
    D = np.diag(np.sum(W, axis=1))
    D_sqrt_inv = np.linalg.inv(np.sqrt(D))
    L = np.eye(n) - D_sqrt_inv @ W @ D_sqrt_inv

    eigvals, eigvecs = np.linalg.eigh(L)
    keep_mask = eigvals < thres
    eigvals_kept = eigvals[keep_mask]
    eigvecs_kept = eigvecs[:, keep_mask]

    # C_Ecc_s_j = smallest_eigenvectors.T @ smallest_eigenvectors
    # Then C_sim = max(0, C) and D_ecc = -log(C_sim)
    if eigvecs_kept.size == 0:
        return float("nan")

    C = eigvecs_kept @ eigvecs_kept.T
    C_sim = np.maximum(C, 0.0)

    D_ecc = np.where(C_sim > 0.0, -np.log(C_sim), np.inf)
    np.fill_diagonal(D_ecc, 0.0)

    sp = floyd_warshall_all_pairs(D_ecc)

    # eccentricity per node: max shortest-path distance from node
    ecc_per_node = np.max(sp, axis=1)
    return float(np.mean(ecc_per_node))


# -----------------------------
# Checkpoint helpers
# -----------------------------
def load_existing_results(path: str) -> Dict[int, Dict[str, Any]]:
    results: Dict[int, Dict[str, Any]] = {}
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    idx = rec.get("index")
                    if idx is not None:
                        results[idx] = rec
        except Exception as e:
            log(f"Warning: Could not load existing results from {path}: {e}")
    return results


# -----------------------------
# Aggregation helpers
# -----------------------------
def mean_and_std(values: List[float]) -> Tuple[float, float]:
    vals = [
        v
        for v in values
        if v is not None and not (isinstance(v, float) and math.isnan(v))
    ]
    if not vals:
        return float("nan"), float("nan")
    arr = np.array(vals, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return mean, std


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    model_id_safe = safe_model_id(MODEL_ID)
    OUTPUT_DIR = os.path.join(OUTPUT_BASE, model_id_safe, "graph_metrics")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    log("=" * 80)
    log(f"Processing model: {MODEL_ID}")
    log(f"Local model path: {MODEL_PATH}")
    log("=" * 80)
    log(f"Data root: {DATA_ROOT}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"Languages to process: {len(LANGS)}")
    log("")

    log("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        local_files_only=True,
    )

    if tokenizer.eos_token_id is None and tokenizer.eos_token is not None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    vocab_size = len(tokenizer)
    log(f"Vocab size: {vocab_size}")
    log(
        f"EOS token id: {tokenizer.eos_token_id}, PAD token id: {tokenizer.pad_token_id}"
    )

    log("Loading model (device_map='auto' to use 2xA100)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
    )
    model.eval()
    log(f"Model loaded. torch.cuda.device_count() = {torch.cuda.device_count()}")
    log("")

    # ---------- First pass: count tokens and questions per language
    log("First pass: counting tokens and questions per language...")
    total_tokens_raw: Dict[str, int] = {}
    question_token_counts: Dict[str, List[int]] = {}

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log(f"[{lang}] scanning {path}")

        total_tokens = 0
        q_tokens: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                user_prompt = make_user_prompt(lang, obj)
                model_input = apply_chat_if_available(tokenizer, user_prompt)
                token_ids = tokenizer.encode(model_input, add_special_tokens=False)
                total_tokens += len(token_ids)
                q_tokens.append(len(token_ids))

        total_tokens_raw[lang] = total_tokens
        question_token_counts[lang] = q_tokens
        log(f"  total_tokens_raw = {total_tokens}, total_questions = {len(q_tokens)}")

    common_tokens = min(total_tokens_raw.values()) if total_tokens_raw else 0
    log(f"\nCommon token budget (min over langs): {common_tokens}")

    # Count how many questions fit within common_tokens per language
    questions_per_lang: Dict[str, int] = {}
    for lang in LANGS:
        used = 0
        count = 0
        for t in question_token_counts[lang]:
            if used + t > common_tokens:
                break
            used += t
            count += 1
        questions_per_lang[lang] = count

    common_questions = min(questions_per_lang.values()) if questions_per_lang else 0
    log(f"Common question budget (min over langs): {common_questions}")
    log("")

    # ---------- Second pass: compute metrics for common_questions per language
    log("Second pass: estimating metrics (common_questions per language)...")

    stats_per_lang: Dict[str, Dict[str, Any]] = {
        lang: {
            "n_examples": 0,
            "used_tokens": 0,
            "lexical_similarity_values": [],
            "degmat_values": [],
            "eigvallaplacian_values": [],
            "eccentricity_values": [],
        }
        for lang in LANGS
    }

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log(f"[{lang}] processing {path}")
        log(f"[{lang}] target question budget: {common_questions}")

        checkpoint_path = os.path.join(OUTPUT_DIR, f"{lang}_graph_metrics.jsonl")
        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
        existing_results = load_existing_results(checkpoint_path)
        log(f"[{lang}] Found {len(existing_results)} existing results, will skip those")

        used_tokens = 0
        question_idx = 0

        checkpoint_f = open(checkpoint_path, "a", encoding="utf-8")

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if question_idx >= common_questions:
                    break

                line = line.strip()
                if not line:
                    continue

                question_idx += 1

                obj = json.loads(line)
                user_prompt = make_user_prompt(lang, obj)
                model_input = apply_chat_if_available(tokenizer, user_prompt)
                text_tokens = int(len(tokenizer.encode(model_input, add_special_tokens=False)))

                # Resume: restore from checkpoint
                if question_idx in existing_results:
                    rec = existing_results[question_idx]
                    stats = stats_per_lang[lang]
                    stats["n_examples"] += 1
                    stats["lexical_similarity_values"].append(
                        float(rec["lexical_similarity"])
                    )
                    stats["degmat_values"].append(float(rec["degmat"]))
                    stats["eigvallaplacian_values"].append(
                        float(rec["eigvallaplacian"])
                    )
                    stats["eccentricity_values"].append(float(rec["eccentricity"]))
                    used_tokens += text_tokens
                    stats["used_tokens"] = used_tokens
                    continue

                log(
                    f"[{lang}] [{question_idx}/{common_questions}] processing example: {text_tokens} tokens"
                )

                # optional greedy answer for logging
                if LOG_GREEDY_ANSWER:
                    log(f"[{lang}] PROMPT:\n{user_prompt}")
                    ans = generate_greedy_answer(
                        model, tokenizer, model_input, GREEDY_MAX_NEW_TOKENS
                    )
                    log(f"[{lang}] ANSWER:\n{ans}")

                # sampling for metrics
                grouped_samples = generate_grouped_samples(
                    model=model,
                    tokenizer=tokenizer,
                    model_inputs_text=[model_input],
                    samples_n=SAMPLES_N,
                    max_new_tokens=MAX_NEW_TOKENS,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    repetition_penalty=REPETITION_PENALTY,
                )
                sample_texts = grouped_samples[0]

                # compute metrics
                lex_u = lexical_similarity_uncertainty(
                    sample_texts, metric=LEXICAL_SIM_METRIC
                )
                deg_u = degmat_uncertainty(sample_texts)
                eig_u = eigvallaplacian_uncertainty(sample_texts)
                ecc_u = eccentricity_uncertainty(sample_texts, thres=0.9)

                stats = stats_per_lang[lang]
                stats["n_examples"] += 1
                stats["lexical_similarity_values"].append(float(lex_u))
                stats["degmat_values"].append(float(deg_u))
                stats["eigvallaplacian_values"].append(float(eig_u))
                stats["eccentricity_values"].append(float(ecc_u))

                used_tokens += text_tokens
                stats["used_tokens"] = used_tokens

                # Save checkpoint
                rec = {
                    "index": question_idx,
                    "language": lang,
                    "text_tokens": text_tokens,
                    "lexical_similarity": float(lex_u),
                    "degmat": float(deg_u),
                    "eigvallaplacian": float(eig_u),
                    "eccentricity": float(ecc_u),
                }
                checkpoint_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                checkpoint_f.flush()

                log(
                    f"[{lang}] METRICS: LexSim={lex_u:.6f}, DegMat={deg_u:.6f}, "
                    f"EigValLaplacian={eig_u:.6f}, Eccentricity={ecc_u:.6f} "
                    f"(samples_n={SAMPLES_N})"
                )

                # Periodic memory cleanup every 10 questions
                if question_idx % 10 == 0:
                    gc.collect()
                    torch.cuda.empty_cache()

        checkpoint_f.close()

        stats = stats_per_lang[lang]
        lex_avg = (
            float(np.mean(stats["lexical_similarity_values"]))
            if stats["lexical_similarity_values"]
            else float("nan")
        )
        deg_avg = (
            float(np.mean(stats["degmat_values"]))
            if stats["degmat_values"]
            else float("nan")
        )
        eig_avg = (
            float(np.mean(stats["eigvallaplacian_values"]))
            if stats["eigvallaplacian_values"]
            else float("nan")
        )
        ecc_avg = (
            float(np.mean(stats["eccentricity_values"]))
            if stats["eccentricity_values"]
            else float("nan")
        )

        log(
            f"[{lang}] done: n_examples={stats['n_examples']}, used_tokens={stats['used_tokens']}"
        )
        log(
            f"[{lang}] avg LexSim={lex_avg:.6f}, avg DegMat={deg_avg:.6f}, "
            f"avg EigValLaplacian={eig_avg:.6f}, avg Eccentricity={ecc_avg:.6f}"
        )
        log("")

        # Clean up memory after each language
        gc.collect()
        torch.cuda.empty_cache()

    # ---------- Save summary
    output_path = os.path.join(OUTPUT_DIR, "graph_metrics_summary.tsv")
    log(f"Saving aggregated metrics to {output_path}")

    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            [
                "language",
                "n_examples",
                "used_tokens",
                "lexical_similarity_mean",
                "lexical_similarity_std",
                "degmat_mean",
                "degmat_std",
                "eigvallaplacian_mean",
                "eigvallaplacian_std",
                "eccentricity_mean",
                "eccentricity_std",
                "eccentricity_infs_cnt",
            ]
        )

        for lang in LANGS:
            st = stats_per_lang[lang]
            lex_m, lex_s = mean_and_std(st["lexical_similarity_values"])
            deg_m, deg_s = mean_and_std(st["degmat_values"])
            eig_m, eig_s = mean_and_std(st["eigvallaplacian_values"])

            ecc_vals = st["eccentricity_values"]
            ecc_infs = sum(1 for v in ecc_vals if v is not None and math.isinf(v))
            ecc_finite = [v for v in ecc_vals if v is not None and math.isfinite(v)]
            ecc_m, ecc_s = mean_and_std(ecc_finite)

            writer.writerow(
                [
                    lang,
                    st["n_examples"],
                    st["used_tokens"],
                    f"{lex_m:.6f}" if not math.isnan(lex_m) else "nan",
                    f"{lex_s:.6f}" if not math.isnan(lex_s) else "nan",
                    f"{deg_m:.6f}" if not math.isnan(deg_m) else "nan",
                    f"{deg_s:.6f}" if not math.isnan(deg_s) else "nan",
                    f"{eig_m:.6f}" if not math.isnan(eig_m) else "nan",
                    f"{eig_s:.6f}" if not math.isnan(eig_s) else "nan",
                    f"{ecc_m:.6f}" if not math.isnan(ecc_m) else "nan",
                    f"{ecc_s:.6f}" if not math.isnan(ecc_s) else "nan",
                    ecc_infs,
                ]
            )

    stats_json_path = os.path.join(OUTPUT_DIR, "graph_metrics_stats_per_lang.json")
    log(f"Saving raw stats_per_lang to: {stats_json_path}")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_per_lang, f, indent=2, ensure_ascii=False)

    log(f"Saved stats for {len(stats_per_lang)} languages")
    log("Done.")


if __name__ == "__main__":
    main()
