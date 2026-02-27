import warnings

warnings.filterwarnings("ignore")

import os
import json
import csv
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple, Set, Optional, Union

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


# ============================================================================
# 1) GENERAL
# ============================================================================
MODEL_ID = "Qwen/Qwen2.5-3B-Instruct"
MODEL_PATH = "/hf_models"

DATA_ROOT = "/work/benchmarks/TUMLU"
OUTPUT_ROOT = "/work/benchmarks/uncertainty_metrics"

# Shared generation constants
NEW_TOKENS = 64
BATCH_SIZE = 2  # used where batching improves throughput (sampling-based metrics)

# If you want quieter runs, set to False
LOG_EACH_EXAMPLE = True

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

MAJOR_LANGS = ["en", "ru", "turkish"]


# ============================================================================
# 2) LAPE
# ============================================================================
LAPE_CHUNK_SIZE = 128
LAPE_TOP_RATE = 0.01
LAPE_FILTER_RATE = 0.95
LAPE_ACTIVATION_BAR_RATIO = 0.95


# ============================================================================
# 3) TOKENIZER_ANALYSIS
# ============================================================================
TOKENIZER_ADD_PREFIX_SPACE = True


# ============================================================================
# 4) LOGIT_UNCERTAINTY
# ============================================================================
# (no extra hyperparameters besides NEW_TOKENS)


# ============================================================================
# 5) DIVERCITY_UNCERTAINTY
# ============================================================================
DIV_SAMPLES_N = 10
DIV_TEMPERATURE = 0.9
DIV_TOP_P = 0.95
DIV_TOP_K = 0  # 0 => disabled
DIV_REPETITION_PENALTY = 1.0
DIV_LEXICAL_SIM_METRIC = "BLEU"  # standalone implementation supports BLEU
DIV_ECC_THRES = 0.9


# ============================================================================
# 6) ATTENTION_UNCERTAINTY
# ============================================================================
FOCUS_GAMMA = 0.9
FOCUS_RHO = 0.01
FOCUS_KW_IDF_QUANTILE = 0.75
IDF_MAX_DOCS = -1
RAUQ_ALPHA = 0.2


# ============================================================================
# Shared helpers
# ============================================================================


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


def make_user_prompt(lang: str, obj: Dict[str, Any]) -> str:
    formatted_choices = format_choices(obj["choices"])
    prompt_template = PROMPTS[lang]
    return prompt_template.format(question=obj["question"], choices=formatted_choices)


def apply_chat_if_available(tokenizer: AutoTokenizer, user_text: str) -> str:
    try:
        if hasattr(tokenizer, "apply_chat_template") and getattr(
            tokenizer, "chat_template", None
        ):
            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_text},
            ]
            return tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
    except Exception:
        pass
    return user_text


def fmt_float(x: Any, nd: int = 6) -> str:
    if x is None:
        return "nan"
    xx = float(x)
    if math.isnan(xx):
        return "nan"
    if math.isinf(xx):
        return "inf" if xx > 0 else "-inf"
    return f"{xx:.{nd}f}"


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


def mean_std_simple(values: List[float]) -> Tuple[float, float]:
    vals = [v for v in values if v is not None]
    if not vals:
        return 0.0, 0.0
    n = len(vals)
    mean = sum(vals) / n
    if n < 2:
        return float(mean), 0.0
    var = sum((x - mean) ** 2 for x in vals) / (n - 1)
    return float(mean), float(var**0.5)


def mean_and_std_excluding_inf(values: List[float]) -> Tuple[float, float, int]:
    inf_cnt = 0
    clean = []
    for v in values:
        if v is None:
            continue
        vv = float(v)
        if math.isnan(vv):
            continue
        if math.isinf(vv):
            inf_cnt += 1
            continue
        clean.append(vv)
    if not clean:
        return float("nan"), float("nan"), inf_cnt
    arr = np.array(clean, dtype=float)
    mean = float(arr.mean())
    std = float(arr.std(ddof=1)) if arr.size >= 2 else 0.0
    return mean, std, inf_cnt


# ============================================================================
# Pass 1: compute minimal token budgets
# ============================================================================


def compute_common_token_budgets(
    tokenizer_main: AutoTokenizer, tokenizer_tok: AutoTokenizer
) -> Dict[str, Any]:
    log("Pass 1: counting total tokens per language (full corpora)...")

    totals_tok: Dict[str, int] = {}
    totals_lape: Dict[str, int] = {}
    totals_unc: Dict[str, int] = {}

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log(f"[{lang}] scanning {path}")

        total_tok = 0
        total_lape = 0
        total_unc = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                user_prompt = make_user_prompt(lang, obj)

                # tokenizer_analysis notion: raw prompt, no special tokens
                total_tok += len(
                    tokenizer_tok.encode(user_prompt, add_special_tokens=False)
                )

                # LAPE notion: raw prompt, with special tokens
                total_lape += len(
                    tokenizer_main(user_prompt, add_special_tokens=True)["input_ids"]
                )

                # uncertainty notion: chat-formatted input, no special tokens
                model_input = apply_chat_if_available(tokenizer_main, user_prompt)
                total_unc += len(
                    tokenizer_main.encode(model_input, add_special_tokens=False)
                )

        totals_tok[lang] = total_tok
        totals_lape[lang] = total_lape
        totals_unc[lang] = total_unc

        log(f"  tokenizer_analysis total_tokens = {total_tok}")
        log(f"  LAPE total_tokens = {total_lape}")
        log(f"  uncertainty total_tokens = {total_unc}")

    common_tok = min(totals_tok.values()) if totals_tok else 0
    common_lape = min(totals_lape.values()) if totals_lape else 0
    common_unc = min(totals_unc.values()) if totals_unc else 0

    log("\nToken budgets (min over langs):")
    log(f"  tokenizer_analysis common_tokens = {int(common_tok)}")
    log(f"  LAPE common_tokens = {int(common_lape)}")
    log(f"  uncertainty common_tokens = {int(common_unc)}")
    log("")

    return {
        "totals_tok": totals_tok,
        "totals_lape": totals_lape,
        "totals_unc": totals_unc,
        "common_tok": int(common_tok),
        "common_lape": int(common_lape),
        "common_unc": int(common_unc),
    }


# ============================================================================
# TOKENIZER_ANALYSIS
# ============================================================================


def compute_tokenizer_analysis(
    tokenizer_tok: AutoTokenizer,
    model_out_dir: str,
    common_tokens: int,
) -> Dict[str, Dict[str, Any]]:
    """Identical metric logic to tokenizer (2).py; adds n_questions_used for summary."""

    vocab_size = len(tokenizer_tok)

    tokens_per_lang: Dict[str, int] = {}
    words_per_lang: Dict[str, int] = {}
    unique_tokens_per_lang: Dict[str, Set[int]] = {}
    chars_per_token_values_per_lang: Dict[str, List[float]] = {}
    questions_used_per_lang: Dict[str, int] = {}

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")

        used_tokens = 0
        words_seen = set()
        token_ids_set: Set[int] = set()
        word_offset = 0
        cpt_values: List[float] = []
        n_questions_used = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if used_tokens >= common_tokens:
                    break
                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                text = make_user_prompt(lang, obj)

                words = text.strip().split()
                if not words:
                    continue

                encoded = tokenizer_tok(
                    words,
                    is_split_into_words=True,
                    add_special_tokens=False,
                    return_attention_mask=False,
                )

                ids = encoded["input_ids"]
                word_ids = encoded.word_ids()
                tok_strs = tokenizer_tok.convert_ids_to_tokens(ids)

                tokens_added_this_question = 0
                for tid, wid, tstr in zip(ids, word_ids, tok_strs):
                    if used_tokens >= common_tokens:
                        break

                    token_ids_set.add(int(tid))
                    used_tokens += 1
                    tokens_added_this_question += 1

                    # Characters per Token (token string length)
                    cpt_values.append(float(len(tstr)))

                    if wid is not None:
                        global_wid = word_offset + int(wid)
                        words_seen.add(global_wid)

                if tokens_added_this_question > 0:
                    n_questions_used += 1

                word_offset += len(words)

        tokens_per_lang[lang] = used_tokens
        words_per_lang[lang] = len(words_seen)
        unique_tokens_per_lang[lang] = token_ids_set
        chars_per_token_values_per_lang[lang] = cpt_values
        questions_used_per_lang[lang] = n_questions_used

    major_sets: Dict[str, Set[int]] = {
        ml: unique_tokens_per_lang.get(ml, set()) for ml in MAJOR_LANGS
    }

    rows: List[Dict[str, Any]] = []
    per_lang: Dict[str, Dict[str, Any]] = {}

    for lang in LANGS:
        total_tokens = tokens_per_lang.get(lang, 0)
        total_words = words_per_lang.get(lang, 0)
        token_ids_set = unique_tokens_per_lang.get(lang, set())

        fertility = (
            (float(total_tokens) / float(total_words))
            if (total_tokens > 0 and total_words > 0)
            else 0.0
        )

        unique_tokens = len(token_ids_set)
        unique_token_fraction = (
            unique_tokens / float(vocab_size) if vocab_size > 0 else 0.0
        )

        if unique_tokens > 0:
            shared_en_count = len(token_ids_set & major_sets["en"])
            shared_en_fraction = shared_en_count / float(unique_tokens)

            shared_ru_count = len(token_ids_set & major_sets["ru"])
            shared_ru_fraction = shared_ru_count / float(unique_tokens)

            shared_tk_count = len(token_ids_set & major_sets["turkish"])
            shared_tk_fraction = shared_tk_count / float(unique_tokens)
        else:
            shared_en_count = 0
            shared_en_fraction = 0.0
            shared_ru_count = 0
            shared_ru_fraction = 0.0
            shared_tk_count = 0
            shared_tk_fraction = 0.0

        cpt_vals = chars_per_token_values_per_lang.get(lang, [])
        cpt_mean, cpt_std = mean_std_simple(cpt_vals)

        row = {
            "model": MODEL_ID,
            "model_path": MODEL_PATH,
            "language": lang,
            "vocab_size": vocab_size,
            "n_questions_used": questions_used_per_lang.get(lang, 0),
            "total_tokens_used": total_tokens,
            "total_words_used": total_words,
            "common_tokens_budget": common_tokens,
            "fertility_tokens_per_word": fertility,
            "unique_tokens": unique_tokens,
            "unique_token_fraction": unique_token_fraction,
            "shared_en_token_count": shared_en_count,
            "shared_en_token_fraction": shared_en_fraction,
            "shared_ru_token_count": shared_ru_count,
            "shared_ru_token_fraction": shared_ru_fraction,
            "shared_turkish_token_count": shared_tk_count,
            "shared_turkish_token_fraction": shared_tk_fraction,
            "chars_per_token_mean": cpt_mean,
            "chars_per_token_std": cpt_std,
        }
        rows.append(row)

        per_lang[lang] = {
            "n_questions": int(questions_used_per_lang.get(lang, 0)),
            "n_tokens": int(total_tokens),
            "total_words": int(total_words),
            "fertility": float(fertility),
            "unique_tokens": int(unique_tokens),
            "unique_token_fraction": float(unique_token_fraction),
            "chars_per_token_mean": float(cpt_mean),
            "chars_per_token_std": float(cpt_std),
            "shared_en_token_count": int(shared_en_count),
            "shared_en_token_fraction": float(shared_en_fraction),
            "shared_ru_token_count": int(shared_ru_count),
            "shared_ru_token_fraction": float(shared_ru_fraction),
            "shared_turkish_token_count": int(shared_tk_count),
            "shared_turkish_token_fraction": float(shared_tk_fraction),
        }

    safe_name = safe_model_id(MODEL_ID)
    out_csv = os.path.join(
        model_out_dir, f"{safe_name}_tokenization_fertility_metrics.csv"
    )
    fieldnames = list(rows[0].keys()) if rows else []
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    log(f"tokenizer_analysis details saved to: {out_csv}")
    return per_lang


# ============================================================================
# ATTENTION_UNCERTAINTY helpers (IDF + Focus/RAUQ metric)
# ============================================================================


def compute_or_load_idf_for_lang(
    tokenizer: AutoTokenizer,
    lang: str,
    lang_jsonl_path: str,
    cache_dir: str,
    max_docs: int,
) -> np.ndarray:
    """Identical logic to attention_uncertainty.py."""

    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{lang}_idf.npy")
    meta_path = os.path.join(cache_dir, f"{lang}_idf_meta.json")

    if os.path.exists(cache_path) and os.path.exists(meta_path):
        try:
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            if (
                meta.get("tokenizer_len") == len(tokenizer)
                and meta.get("max_docs") == max_docs
            ):
                return np.load(cache_path)
        except Exception:
            pass

    vocab_size = len(tokenizer)
    df = np.zeros((vocab_size,), dtype=np.int64)

    N = 0
    with open(lang_jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if max_docs > 0 and N >= max_docs:
                break
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)

            doc_text = obj["question"] + "\n" + "\n".join(obj["choices"])
            ids = tokenizer.encode(doc_text, add_special_tokens=False)
            if not ids:
                continue
            N += 1
            for tid in set(ids):
                if 0 <= tid < vocab_size:
                    df[tid] += 1

    if N <= 0:
        idf = np.ones((vocab_size,), dtype=np.float32)
    else:
        idf = np.log((N + 1.0) / (df.astype(np.float64) + 1.0)).astype(np.float32)

    np.save(cache_path, idf)
    with open(meta_path, "w", encoding="utf-8") as fmeta:
        json.dump(
            {"tokenizer_len": len(tokenizer), "max_docs": max_docs, "N": int(N)},
            fmeta,
            ensure_ascii=False,
            indent=2,
        )

    return idf


@torch.inference_mode()
def compute_focus_and_rauq_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_seq: torch.Tensor,
    prompt_len: int,
    gen_len: int,
    idf_t: torch.Tensor,
    focus_gamma: float,
    focus_rho: float,
    focus_kw_idf_quantile: float,
    rauq_alpha: float,
) -> Dict[str, float]:
    """Identical metric logic to attention_uncertainty.py."""

    if gen_len <= 0 or prompt_len <= 0:
        return {"Focus": float("nan"), "RAUQ": float("nan")}

    eps = 1e-12

    input_ids = full_seq.unsqueeze(0)
    attention_mask = torch.ones_like(
        input_ids, dtype=torch.long, device=input_ids.device
    )

    out = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=True,
        return_dict=True,
    )

    logits = out.logits.to(dtype=torch.float32)

    attentions = out.attentions
    if attentions is None or len(attentions) == 0:
        return {"Focus": float("nan"), "RAUQ": float("nan")}

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    L = logits.shape[1]
    V = logits.shape[2]

    gen_start = prompt_len
    gen_end = prompt_len + gen_len
    pred_start = gen_start - 1
    pred_end = gen_end - 1

    if pred_start < 0 or pred_end <= pred_start:
        return {"Focus": float("nan"), "RAUQ": float("nan")}

    logits_pred = logits[0, pred_start:pred_end, :]
    target = input_ids[0, gen_start:gen_end].to(dtype=torch.long)

    logp_all = torch.log_softmax(logits_pred, dim=-1)
    logp_t = logp_all.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    p_tok = torch.exp(logp_t)

    gen_token_ids = target
    idf_vals = idf_t[gen_token_ids].detach().to(dtype=torch.float32)

    if idf_vals.numel() == 0:
        kw_mask = torch.zeros((gen_len,), device=input_ids.device, dtype=torch.bool)
    else:
        thresh = torch.quantile(idf_vals, float(focus_kw_idf_quantile))
        kw_mask = idf_vals >= thresh

    focus_layer_scores = []

    for layer_idx in range(num_layers):
        attn = attentions[layer_idx][0]
        if attn.shape[0] != num_heads:
            continue

        attn_mean = attn.mean(dim=0)

        a_gen = attn_mean[gen_start:gen_end, :]
        if a_gen.shape[0] != gen_len:
            continue

        a_prompt = a_gen[:, :prompt_len]

        if a_prompt.numel() == 0:
            focus_layer_scores.append(float("nan"))
            continue

        if kw_mask.any():
            kw_idx = kw_mask.nonzero(as_tuple=False).view(-1)
            a_kw = a_prompt[kw_idx]
        else:
            a_kw = a_prompt

        if a_kw.numel() == 0:
            focus_layer_scores.append(float("nan"))
            continue

        gamma = float(focus_gamma)
        rho = float(focus_rho)

        a_pow = torch.pow(torch.clamp(a_kw, min=eps), gamma)
        denom = torch.sum(a_pow, dim=-1, keepdim=True) + eps
        pi = a_pow / denom

        max_pi = torch.max(pi, dim=-1).values
        s = (max_pi - rho) / (1.0 - rho)
        s = torch.clamp(s, min=0.0, max=1.0)

        focus_layer_scores.append(float(s.mean().item()))

    focus_score = (
        float(np.max(focus_layer_scores)) if focus_layer_scores else float("nan")
    )

    rauq_layer_scores = []

    for layer_idx in range(num_layers):
        attn = attentions[layer_idx][0]
        if attn.shape[0] != num_heads:
            continue

        attn_mean = attn.mean(dim=0)

        a_gen = attn_mean[gen_start:gen_end, :]
        if a_gen.shape[0] != gen_len:
            continue

        a_prev = a_gen[:, : L - 1]
        if a_prev.shape[1] < gen_start:
            rauq_layer_scores.append(float("nan"))
            continue

        a_prev = a_prev[:, gen_start - 1 : gen_end - 1]

        if a_prev.shape[0] != gen_len or a_prev.shape[1] != gen_len:
            rauq_layer_scores.append(float("nan"))
            continue

        a_prev = torch.diagonal(a_prev, offset=0, dim1=0, dim2=1)

        alpha = float(rauq_alpha)
        c = torch.zeros((gen_len,), device=p_tok.device, dtype=torch.float32)
        c[0] = p_tok[0].to(dtype=torch.float32)

        if gen_len >= 2:
            for i in range(1, gen_len):
                cur_p = p_tok[i].to(dtype=torch.float32)
                att_w = a_prev[i - 1].to(dtype=torch.float32)
                c[i] = alpha * cur_p + (1.0 - alpha) * att_w * c[i - 1]

        c = torch.clamp(c, min=eps)
        u_l = -torch.log(c).mean()
        rauq_layer_scores.append(float(u_l.item()))

    rauq_score = float(np.max(rauq_layer_scores)) if rauq_layer_scores else float("nan")
    return {"Focus": focus_score, "RAUQ": rauq_score}


# ============================================================================
# LOGIT_UNCERTAINTY metric (Perplexity/MSP/Entropy)
# ============================================================================


@torch.inference_mode()
def compute_logit_metrics_single(
    model: AutoModelForCausalLM,
    full_seq: torch.Tensor,
    prompt_len: int,
    gen_len: int,
) -> Dict[str, float]:
    """Identical metric logic to logit_uncertainty.py (compute_metrics_single)."""

    if gen_len <= 0 or prompt_len <= 0:
        return {
            "Perplexity": float("nan"),
            "MaximumSequenceProbability": float("nan"),
            "MeanTokenEntropy": float("nan"),
        }

    input_ids = full_seq.unsqueeze(0)
    attention_mask = torch.ones_like(
        input_ids, dtype=torch.long, device=input_ids.device
    )

    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
    )
    logits = outputs.logits.to(dtype=torch.float32)

    gen_start = prompt_len
    gen_end = prompt_len + gen_len
    pred_start = gen_start - 1
    pred_end = gen_end - 1

    if pred_start < 0 or pred_end <= pred_start:
        return {
            "Perplexity": float("nan"),
            "MaximumSequenceProbability": float("nan"),
            "MeanTokenEntropy": float("nan"),
        }

    logits_pred = logits[0, pred_start:pred_end, :]
    target = input_ids[0, gen_start:gen_end].to(dtype=torch.long)

    logp_all = torch.log_softmax(logits_pred, dim=-1)
    logp_t = logp_all.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)

    logp_np = logp_t.detach().cpu().numpy().astype(np.float64)

    perplexity_val = float(-np.mean(logp_np))
    msp_val = float(-np.sum(logp_np))

    p_all = torch.exp(logp_all)
    entropy_t = -(p_all * logp_all).sum(dim=-1)
    entropy_np = entropy_t.detach().cpu().numpy().astype(np.float64)
    mean_entropy_val = float(np.mean(entropy_np))

    return {
        "Perplexity": perplexity_val,
        "MaximumSequenceProbability": msp_val,
        "MeanTokenEntropy": mean_entropy_val,
    }


# ============================================================================
# DIVERCITY_UNCERTAINTY metric functions
# ============================================================================


def compute_sim_score_jaccard(text1: str, text2: str) -> float:
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    union = tokens1.union(tokens2)
    if len(union) == 0:
        return 0.0
    inter = tokens1.intersection(tokens2)
    return float(len(inter) / len(union))


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

    precisions = []
    for n in range(1, max_n + 1):
        p_n = _modified_precision(reference, candidate, n)
        precisions.append(p_n)

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
    # lm-polygraph returns -mean(1 - similarity) over all unordered pairs
    n = len(sample_texts)
    if n < 2:
        return float("nan")

    dists = []
    if metric.upper() != "BLEU":
        raise ValueError(
            "This standalone runner implements LexicalSimilarity only for metric='BLEU'."
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
    eigvecs_kept = eigvecs[:, keep_mask]

    if eigvecs_kept.size == 0:
        return float("nan")

    C = eigvecs_kept.T @ eigvecs_kept
    C_sim = np.maximum(C, 0.0)

    D_ecc = np.where(C_sim > 0.0, -np.log(C_sim), np.inf)
    np.fill_diagonal(D_ecc, 0.0)

    sp = floyd_warshall_all_pairs(D_ecc)

    ecc_per_node = np.max(sp, axis=1)
    return float(np.mean(ecc_per_node))


# ============================================================================
# Unified generation function
# ============================================================================


@torch.inference_mode()
def generate_answer(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_inputs_text: Union[str, List[str]],
    max_new_tokens: int,
    do_sample: bool,
    num_return_sequences: int = 1,
    temperature: Optional[float] = None,
    top_p: Optional[float] = None,
    top_k: Optional[int] = None,
    repetition_penalty: Optional[float] = None,
) -> Any:
    """Single entry point for generation.

    Deterministic (do_sample=False, num_return_sequences=1, single prompt):
      returns (full_seq, prompt_len, gen_len, answer_text)

    Sampling (do_sample=True, num_return_sequences>=1, list of prompts):
      returns grouped samples: List[List[str]] with shape [batch][num_return_sequences]
    """

    if isinstance(model_inputs_text, str):
        model_inputs = [model_inputs_text]
    else:
        model_inputs = list(model_inputs_text)

    if not do_sample:
        if len(model_inputs) != 1 or num_return_sequences != 1:
            raise ValueError(
                "Deterministic generation expects a single prompt and num_return_sequences=1"
            )

        enc = tokenizer(
            model_inputs[0],
            return_tensors="pt",
            padding=False,
            truncation=False,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
            model.device
        )

        prompt_len = int(attention_mask.sum().item())

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

        gen_part = out[prompt_len:]
        gen_len = int(gen_part.shape[0])

        if tokenizer.eos_token_id is not None and gen_len > 0:
            eos_pos = (gen_part == tokenizer.eos_token_id).nonzero(as_tuple=False)
            if eos_pos.numel() > 0:
                gen_len = int(eos_pos[0].item()) + 1  # include EOS

        full_seq = out[: prompt_len + gen_len]

        gen_tokens = full_seq[prompt_len:]
        if (
            tokenizer.eos_token_id is not None
            and gen_tokens.numel() > 0
            and int(gen_tokens[-1].item()) == int(tokenizer.eos_token_id)
        ):
            gen_tokens = gen_tokens[:-1]
        answer_text = tokenizer.decode(gen_tokens, skip_special_tokens=True)

        return full_seq, prompt_len, gen_len, answer_text

    # sampling path (batch + regroup) — identical to divercity_uncertainty.py generation_grouped_samples
    enc = tokenizer(
        model_inputs,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = enc["input_ids"].to(model.device)
    attention_mask = enc.get("attention_mask", torch.ones_like(input_ids)).to(
        model.device
    )

    prompt_lens = attention_mask.sum(dim=1).tolist()

    gen_kwargs: Dict[str, Any] = dict(
        max_new_tokens=max_new_tokens,
        do_sample=True,
        num_beams=1,
        num_return_sequences=num_return_sequences,
        temperature=float(temperature) if temperature is not None else 1.0,
        top_p=float(top_p) if top_p is not None else 1.0,
        repetition_penalty=(
            float(repetition_penalty) if repetition_penalty is not None else 1.0
        ),
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        use_cache=True,
    )

    if top_k is not None and int(top_k) > 0:
        gen_kwargs["top_k"] = int(top_k)

    out = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        **gen_kwargs,
    )

    batch_size = len(model_inputs)
    grouped: List[List[str]] = [[] for _ in range(batch_size)]

    for i in range(batch_size):
        base = i * num_return_sequences
        p_len = int(prompt_lens[i])

        for j in range(num_return_sequences):
            seq = out[base + j]
            gen_tokens = seq[p_len:]

            if tokenizer.eos_token_id is not None:
                eos_positions = (gen_tokens == tokenizer.eos_token_id).nonzero(
                    as_tuple=False
                )
                if eos_positions.numel() > 0:
                    gen_tokens = gen_tokens[: int(eos_positions[0].item())]

            txt = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            grouped[i].append(txt)

    return grouped


# ============================================================================
# Uncertainty metrics: unified pass over dataset
# ============================================================================


def compute_uncertainty_metrics_one_pass(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_out_dir: str,
    common_tokens: int,
) -> Tuple[
    Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]
]:
    """One pass over the dataset (per language, truncated by common_tokens).

    Returns per-language summaries for:
      - logit metrics
      - attention metrics
      - diversity metrics

    Also writes detailed outputs to model_out_dir.
    """

    safe_name = safe_model_id(MODEL_ID)

    # Detailed per-example outputs
    logit_detail_path = os.path.join(
        model_out_dir, f"{safe_name}_logit_metrics_detailed.jsonl"
    )
    attn_detail_path = os.path.join(
        model_out_dir, f"{safe_name}_attention_metrics_detailed.jsonl"
    )
    div_detail_path = os.path.join(
        model_out_dir, f"{safe_name}_divercity_metrics_detailed.jsonl"
    )

    # Per-language stats containers (mirrors standalone scripts)
    logit_stats: Dict[str, Dict[str, Any]] = {
        lang: {
            "n_examples": 0,
            "used_tokens": 0,
            "Perplexity_values": [],
            "MaximumSequenceProbability_values": [],
            "MeanTokenEntropy_values": [],
        }
        for lang in LANGS
    }

    attn_stats: Dict[str, Dict[str, Any]] = {
        lang: {
            "n_examples": 0,
            "used_tokens": 0,
            "RAUQ_values": [],
            "Focus_values": [],
        }
        for lang in LANGS
    }

    div_stats: Dict[str, Dict[str, Any]] = {
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

    idf_cache_dir = os.path.join(model_out_dir, "idf_cache")

    # Stream JSONL to avoid huge RAM
    os.makedirs(model_out_dir, exist_ok=True)
    lf = open(logit_detail_path, "w", encoding="utf-8")
    af = open(attn_detail_path, "w", encoding="utf-8")
    df = open(div_detail_path, "w", encoding="utf-8")

    try:
        log("Pass 2: unified uncertainty pass (deterministic + sampling metrics)...")
        log(f"Common token budget (uncertainty): {common_tokens}")

        for lang in LANGS:
            path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
            log(f"[{lang}] processing {path}")

            # IDF for this language (attention metrics)
            log(f"[{lang}] IDF: computing/loading (max_docs={IDF_MAX_DOCS})")
            idf_np = compute_or_load_idf_for_lang(
                tokenizer=tokenizer,
                lang=lang,
                lang_jsonl_path=path,
                cache_dir=idf_cache_dir,
                max_docs=IDF_MAX_DOCS,
            )
            idf_t = torch.tensor(idf_np, device=model.device, dtype=torch.float32)

            used_tokens = 0
            q_idx = 0

            # buffer for sampling-based metrics
            sample_buffer_inputs: List[str] = []
            sample_buffer_meta: List[Dict[str, Any]] = []  # lang, q_idx

            def flush_sampling_buffer() -> None:
                if not sample_buffer_inputs:
                    return

                grouped_samples: List[List[str]] = generate_answer(
                    model=model,
                    tokenizer=tokenizer,
                    model_inputs_text=sample_buffer_inputs,
                    max_new_tokens=NEW_TOKENS,
                    do_sample=True,
                    num_return_sequences=DIV_SAMPLES_N,
                    temperature=DIV_TEMPERATURE,
                    top_p=DIV_TOP_P,
                    top_k=DIV_TOP_K,
                    repetition_penalty=DIV_REPETITION_PENALTY,
                )

                for meta, sample_texts in zip(sample_buffer_meta, grouped_samples):
                    lex_u = lexical_similarity_uncertainty(
                        sample_texts, metric=DIV_LEXICAL_SIM_METRIC
                    )
                    deg_u = degmat_uncertainty(sample_texts)
                    eig_u = eigvallaplacian_uncertainty(sample_texts)
                    ecc_u = eccentricity_uncertainty(sample_texts, thres=DIV_ECC_THRES)

                    st = div_stats[lang]
                    st["n_examples"] += 1
                    st["lexical_similarity_values"].append(float(lex_u))
                    st["degmat_values"].append(float(deg_u))
                    st["eigvallaplacian_values"].append(float(eig_u))
                    st["eccentricity_values"].append(float(ecc_u))
                    st["used_tokens"] = used_tokens

                    df.write(
                        json.dumps(
                            {
                                "language": lang,
                                "q_idx": int(meta["q_idx"]),
                                "LexicalSimilarity": float(lex_u),
                                "DegMat": float(deg_u),
                                "EigValLaplacian": float(eig_u),
                                "Eccentricity": float(ecc_u),
                                "samples_n": int(DIV_SAMPLES_N),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    if LOG_EACH_EXAMPLE:
                        log(
                            f"[{lang}][{meta['q_idx']}] METRICS(samples_n={DIV_SAMPLES_N}): "
                            f"LexSim={lex_u:.6f}, DegMat={deg_u:.6f}, EigValLaplacian={eig_u:.6f}, Eccentricity={ecc_u:.6f}"
                        )

                sample_buffer_inputs.clear()
                sample_buffer_meta.clear()

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue

                    if used_tokens >= common_tokens:
                        break

                    obj = json.loads(line)
                    user_prompt = make_user_prompt(lang, obj)
                    model_input = apply_chat_if_available(tokenizer, user_prompt)

                    prompt_tokens = len(
                        tokenizer.encode(model_input, add_special_tokens=False)
                    )
                    if used_tokens + prompt_tokens > common_tokens:
                        break

                    q_idx += 1

                    if LOG_EACH_EXAMPLE:
                        log(
                            f"[{lang}][{q_idx}] prompt_tokens={prompt_tokens}, used={used_tokens}/{common_tokens}"
                        )
                        log(f"[{lang}][{q_idx}] QUESTION:\n{user_prompt}")

                    full_seq, prompt_len, gen_len, answer_text = generate_answer(
                        model=model,
                        tokenizer=tokenizer,
                        model_inputs_text=model_input,
                        max_new_tokens=NEW_TOKENS,
                        do_sample=False,
                        num_return_sequences=1,
                    )

                    if LOG_EACH_EXAMPLE:
                        log(f"[{lang}][{q_idx}] ANSWER:\n{answer_text}")

                    # logit metrics
                    lm = compute_logit_metrics_single(
                        model=model,
                        full_seq=full_seq,
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                    )

                    lstat = logit_stats[lang]
                    lstat["n_examples"] += 1
                    lstat["Perplexity_values"].append(float(lm["Perplexity"]))
                    lstat["MaximumSequenceProbability_values"].append(
                        float(lm["MaximumSequenceProbability"])
                    )
                    lstat["MeanTokenEntropy_values"].append(
                        float(lm["MeanTokenEntropy"])
                    )
                    lstat["used_tokens"] = used_tokens + prompt_tokens

                    lf.write(
                        json.dumps(
                            {
                                "language": lang,
                                "q_idx": int(q_idx),
                                "Perplexity": float(lm["Perplexity"]),
                                "MaximumSequenceProbability": float(
                                    lm["MaximumSequenceProbability"]
                                ),
                                "MeanTokenEntropy": float(lm["MeanTokenEntropy"]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    # attention metrics
                    am = compute_focus_and_rauq_single(
                        model=model,
                        tokenizer=tokenizer,
                        full_seq=full_seq,
                        prompt_len=prompt_len,
                        gen_len=gen_len,
                        idf_t=idf_t,
                        focus_gamma=FOCUS_GAMMA,
                        focus_rho=FOCUS_RHO,
                        focus_kw_idf_quantile=FOCUS_KW_IDF_QUANTILE,
                        rauq_alpha=RAUQ_ALPHA,
                    )

                    astat = attn_stats[lang]
                    astat["n_examples"] += 1
                    astat["RAUQ_values"].append(float(am["RAUQ"]))
                    astat["Focus_values"].append(float(am["Focus"]))
                    astat["used_tokens"] = used_tokens + prompt_tokens

                    af.write(
                        json.dumps(
                            {
                                "language": lang,
                                "q_idx": int(q_idx),
                                "RAUQ": float(am["RAUQ"]),
                                "Focus": float(am["Focus"]),
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )

                    if LOG_EACH_EXAMPLE:
                        log(
                            f"[{lang}][{q_idx}] METRICS(single): "
                            f"Perplexity={lm['Perplexity']:.6f}, "
                            f"MSP={lm['MaximumSequenceProbability']:.6f}, "
                            f"MeanTokenEntropy={lm['MeanTokenEntropy']:.6f}, "
                            f"RAUQ={am['RAUQ']:.6f}, Focus={am['Focus']:.6f}"
                        )

                    # budget accounting and sampling buffer
                    used_tokens += prompt_tokens

                    sample_buffer_inputs.append(model_input)
                    sample_buffer_meta.append({"q_idx": q_idx})

                    if len(sample_buffer_inputs) >= max(1, int(BATCH_SIZE)):
                        flush_sampling_buffer()

            # flush remaining sampling buffer for this language
            flush_sampling_buffer()

            log(
                f"[{lang}] done: used_tokens={used_tokens}/{common_tokens}, "
                f"n_examples_single={logit_stats[lang]['n_examples']}, n_examples_samples={div_stats[lang]['n_examples']}"
            )
            log("")

    finally:
        lf.close()
        af.close()
        df.close()

    # --- Write per-language TSV/JSON outputs (compatible naming)

    ppl_tsv = os.path.join(model_out_dir, f"{safe_name}_ppl_msp_entropy_summary.tsv")
    ppl_json = os.path.join(
        model_out_dir, f"{safe_name}_ppl_msp_entropy_stats_per_lang.json"
    )

    with open(ppl_tsv, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            [
                "language",
                "n_examples",
                "used_tokens",
                "Perplexity_mean",
                "Perplexity_std",
                "MaximumSequenceProbability_mean",
                "MaximumSequenceProbability_std",
                "MeanTokenEntropy_mean",
                "MeanTokenEntropy_std",
            ]
        )
        for lang in LANGS:
            st = logit_stats[lang]
            ppl_m, ppl_s = mean_and_std(st["Perplexity_values"])
            msp_m, msp_s = mean_and_std(st["MaximumSequenceProbability_values"])
            ent_m, ent_s = mean_and_std(st["MeanTokenEntropy_values"])
            writer.writerow(
                [
                    lang,
                    st["n_examples"],
                    st["used_tokens"],
                    fmt_float(ppl_m),
                    fmt_float(ppl_s),
                    fmt_float(msp_m),
                    fmt_float(msp_s),
                    fmt_float(ent_m),
                    fmt_float(ent_s),
                ]
            )

    with open(ppl_json, "w", encoding="utf-8") as f:
        json.dump(logit_stats, f, indent=2, ensure_ascii=False)

    rauq_tsv = os.path.join(model_out_dir, f"{safe_name}_rauq_focus_summary.tsv")
    rauq_json = os.path.join(
        model_out_dir, f"{safe_name}_rauq_focus_stats_per_lang.json"
    )

    with open(rauq_tsv, "w", encoding="utf-8", newline="") as out_f:
        writer = csv.writer(out_f, delimiter="\t")
        writer.writerow(
            [
                "language",
                "n_examples",
                "used_tokens",
                "RAUQ_mean",
                "RAUQ_std",
                "Focus_mean",
                "Focus_std",
            ]
        )
        for lang in LANGS:
            st = attn_stats[lang]
            rauq_m, rauq_s = mean_and_std(st["RAUQ_values"])
            foc_m, foc_s = mean_and_std(st["Focus_values"])
            writer.writerow(
                [
                    lang,
                    st["n_examples"],
                    st["used_tokens"],
                    fmt_float(rauq_m),
                    fmt_float(rauq_s),
                    fmt_float(foc_m),
                    fmt_float(foc_s),
                ]
            )

    with open(rauq_json, "w", encoding="utf-8") as f:
        json.dump(attn_stats, f, indent=2, ensure_ascii=False)

    graph_tsv = os.path.join(model_out_dir, f"{safe_name}_graph_metrics_summary.tsv")
    graph_json = os.path.join(
        model_out_dir, f"{safe_name}_graph_metrics_stats_per_lang.json"
    )

    with open(graph_tsv, "w", encoding="utf-8", newline="") as out_f:
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
                "eccentricity_mean_excluding_inf",
                "eccentricity_std_excluding_inf",
                "eccentricity_infs_cnt",
            ]
        )

        for lang in LANGS:
            st = div_stats[lang]
            lex_m, lex_s = mean_and_std(st["lexical_similarity_values"])
            deg_m, deg_s = mean_and_std(st["degmat_values"])
            eig_m, eig_s = mean_and_std(st["eigvallaplacian_values"])
            ecc_m, ecc_s, ecc_inf_cnt = mean_and_std_excluding_inf(
                st["eccentricity_values"]
            )

            writer.writerow(
                [
                    lang,
                    st["n_examples"],
                    st["used_tokens"],
                    fmt_float(lex_m),
                    fmt_float(lex_s),
                    fmt_float(deg_m),
                    fmt_float(deg_s),
                    fmt_float(eig_m),
                    fmt_float(eig_s),
                    fmt_float(ecc_m),
                    fmt_float(ecc_s),
                    int(ecc_inf_cnt),
                ]
            )

    with open(graph_json, "w", encoding="utf-8") as f:
        json.dump(div_stats, f, indent=2, ensure_ascii=False)

    log(
        f"logit uncertainty details saved to: {ppl_tsv}, {ppl_json}, {logit_detail_path}"
    )
    log(
        f"attention uncertainty details saved to: {rauq_tsv}, {rauq_json}, {attn_detail_path}"
    )
    log(
        f"divercity uncertainty details saved to: {graph_tsv}, {graph_json}, {div_detail_path}"
    )

    # --- build compact per-lang dicts for combined summary
    logit_res: Dict[str, Dict[str, Any]] = {}
    attn_res: Dict[str, Dict[str, Any]] = {}
    div_res: Dict[str, Dict[str, Any]] = {}

    for lang in LANGS:
        stl = logit_stats[lang]
        ppl_m, ppl_s = mean_and_std(stl["Perplexity_values"])
        msp_m, msp_s = mean_and_std(stl["MaximumSequenceProbability_values"])
        ent_m, ent_s = mean_and_std(stl["MeanTokenEntropy_values"])
        logit_res[lang] = {
            "perplexity_mean": float(ppl_m),
            "perplexity_std": float(ppl_s),
            "msp_mean": float(msp_m),
            "msp_std": float(msp_s),
            "entropy_mean": float(ent_m),
            "entropy_std": float(ent_s),
            "n_examples": int(stl["n_examples"]),
            "used_tokens": int(stl["used_tokens"]),
        }

        sta = attn_stats[lang]
        rauq_m, rauq_s = mean_and_std(sta["RAUQ_values"])
        foc_m, foc_s = mean_and_std(sta["Focus_values"])
        attn_res[lang] = {
            "rauq_mean": float(rauq_m),
            "rauq_std": float(rauq_s),
            "focus_mean": float(foc_m),
            "focus_std": float(foc_s),
            "n_examples": int(sta["n_examples"]),
            "used_tokens": int(sta["used_tokens"]),
        }

        std = div_stats[lang]
        lex_m, lex_s = mean_and_std(std["lexical_similarity_values"])
        deg_m, deg_s = mean_and_std(std["degmat_values"])
        eig_m, eig_s = mean_and_std(std["eigvallaplacian_values"])
        ecc_m, ecc_s, ecc_inf_cnt = mean_and_std_excluding_inf(
            std["eccentricity_values"]
        )
        div_res[lang] = {
            "lexsim_mean": float(lex_m),
            "lexsim_std": float(lex_s),
            "degmat_mean": float(deg_m),
            "degmat_std": float(deg_s),
            "eigval_mean": float(eig_m),
            "eigval_std": float(eig_s),
            "ecc_mean": float(ecc_m),
            "ecc_std": float(ecc_s),
            "ecc_inf_cnt": int(ecc_inf_cnt),
            "n_examples": int(std["n_examples"]),
            "used_tokens": int(std["used_tokens"]),
        }

    return logit_res, attn_res, div_res


# ============================================================================
# LAPE
# ============================================================================


def compute_lape(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_out_dir: str,
    common_tokens: int,
) -> Dict[str, Dict[str, Any]]:
    """Identical LAPE logic to lape (2).py; takes common_tokens computed in pass 1."""

    torch.set_grad_enabled(False)

    has_cuda = torch.cuda.is_available()
    input_device = (
        model.get_input_embeddings().weight.device if has_cuda else torch.device("cpu")
    )

    model_type = model.config.model_type

    if model_type in ("llama", "mistral", "qwen2", "gemma2"):
        layers = model.model.layers
    elif model_type == "gpt2":
        layers = model.transformer.h
    elif model_type == "bloom":
        layers = model.transformer.h
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    num_layers = len(layers)

    if model_type in ("llama", "mistral", "qwen2", "gemma2"):
        sample_mlp = layers[0].mlp
        intermediate_size = sample_mlp.gate_proj.out_features
    elif model_type == "gpt2":
        sample_mlp = layers[0].mlp
        intermediate_size = sample_mlp.c_fc.weight.shape[1]
    elif model_type == "bloom":
        sample_mlp = layers[0].mlp
        intermediate_size = sample_mlp.dense_h_to_4h.out_features
    else:
        raise ValueError(
            f"Unsupported model_type (for intermediate_size): {model_type}"
        )

    over_zero = torch.zeros(
        num_layers, intermediate_size, len(LANGS), dtype=torch.long, device="cpu"
    )
    token_counts = torch.zeros(len(LANGS), dtype=torch.long, device="cpu")
    current_lang_index = 0

    log("Registering LAPE forward hooks...")

    if model_type in ("llama", "mistral", "qwen2", "gemma2"):

        def make_gate_hook(layer_idx, act_fn):
            def hook(module, input, output):
                nonlocal current_lang_index
                activation = act_fn(output.to(torch.float32))
                active = (activation > 0).sum(dim=(0, 1))
                over_zero[layer_idx, :, current_lang_index] += active.to(
                    dtype=torch.long
                ).cpu()

            return hook

        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            mlp.gate_proj.register_forward_hook(make_gate_hook(layer_idx, mlp.act_fn))

    elif model_type == "gpt2":

        def make_fc_hook(layer_idx, act_fn):
            def hook(module, input, output):
                nonlocal current_lang_index
                activation = act_fn(output.to(torch.float32))
                active = (activation > 0).sum(dim=(0, 1))
                over_zero[layer_idx, :, current_lang_index] += active.to(
                    dtype=torch.long
                ).cpu()

            return hook

        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            mlp.c_fc.register_forward_hook(make_fc_hook(layer_idx, mlp.act))

    elif model_type == "bloom":

        def make_gelu_hook(layer_idx):
            def hook(module, input, output):
                nonlocal current_lang_index
                activation = output.to(torch.float32)
                active = (activation > 0).sum(dim=(0, 1))
                over_zero[layer_idx, :, current_lang_index] += active.to(
                    dtype=torch.long
                ).cpu()

            return hook

        for layer_idx, layer in enumerate(layers):
            mlp = layer.mlp
            mlp.gelu_impl.register_forward_hook(make_gelu_hook(layer_idx))

    log("Collecting neuron activations (LAPE)...")

    with torch.no_grad():
        for lang_idx, lang in enumerate(LANGS):
            current_lang_index = lang_idx
            path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")

            tokens_used = 0
            buffer: List[int] = []

            def flush_full_chunks() -> None:
                nonlocal tokens_used, buffer
                while len(buffer) >= LAPE_CHUNK_SIZE and tokens_used < common_tokens:
                    take = LAPE_CHUNK_SIZE
                    chunk = buffer[:take]
                    buffer = buffer[take:]

                    input_ids = (
                        torch.tensor(chunk, dtype=torch.long)
                        .unsqueeze(0)
                        .to(input_device)
                    )
                    _ = model(input_ids=input_ids, use_cache=False)

                    tokens_used += take

            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    if tokens_used >= common_tokens:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    text = make_user_prompt(lang, obj)
                    ids = tokenizer(text, add_special_tokens=True)["input_ids"]

                    remaining = common_tokens - (tokens_used + len(buffer))
                    if remaining <= 0:
                        break
                    if len(ids) > remaining:
                        ids = ids[:remaining]

                    buffer.extend(ids)
                    flush_full_chunks()

            if tokens_used < common_tokens and len(buffer) > 0:
                remaining = common_tokens - tokens_used
                if remaining > 0:
                    chunk = buffer[:remaining]
                    input_ids = (
                        torch.tensor(chunk, dtype=torch.long)
                        .unsqueeze(0)
                        .to(input_device)
                    )
                    _ = model(input_ids=input_ids, use_cache=False)
                    tokens_used += len(chunk)
                buffer = []

            token_counts[lang_idx] = tokens_used

            if has_cuda:
                torch.cuda.empty_cache()

    lang_num = len(LANGS)
    n = token_counts.to(torch.float32)

    activation_probs = over_zero.to(torch.float32) / n.view(1, 1, lang_num)

    normed_activation_probs = activation_probs / activation_probs.sum(
        dim=-1, keepdim=True
    )
    normed_activation_probs[torch.isnan(normed_activation_probs)] = 0.0

    log_probs = torch.where(
        normed_activation_probs > 0,
        normed_activation_probs.log(),
        torch.zeros_like(normed_activation_probs),
    )
    entropy = -(normed_activation_probs * log_probs).sum(dim=-1)

    flattened_probs = activation_probs.flatten()
    k_filter = int(round(len(flattened_probs) * LAPE_FILTER_RATE))
    if k_filter < 1:
        k_filter = 1
    top_prob_value = flattened_probs.kthvalue(k_filter).values.item()

    top_position = (activation_probs > top_prob_value).sum(dim=-1)
    entropy = entropy.clone()
    entropy[top_position == 0] = float("inf")

    flattened_entropy = entropy.flatten()
    k_top = int(round(len(flattened_entropy) * LAPE_TOP_RATE))
    if k_top < 1:
        k_top = 1

    _, top_index = flattened_entropy.topk(k_top, largest=False)

    row_index = top_index // intermediate_size
    col_index = top_index % intermediate_size

    selected_probs = activation_probs[row_index, col_index]
    selected_probs_t = selected_probs.transpose(0, 1)

    k_bar = int(round(len(flattened_probs) * LAPE_ACTIVATION_BAR_RATIO))
    if k_bar < 1:
        k_bar = 1
    activation_bar = flattened_probs.kthvalue(k_bar).values.item()

    lang_index_tensor, selected_idx_in_selected = torch.where(
        selected_probs_t > activation_bar
    )

    final_lang_neurons = []
    for lang_id in range(lang_num):
        mask = lang_index_tensor == lang_id
        idx_for_lang = selected_idx_in_selected[mask]
        layer_to_neurons = [[] for _ in range(num_layers)]
        for pos in idx_for_lang.tolist():
            layer = int(row_index[pos])
            neuron = int(col_index[pos])
            layer_to_neurons[layer].append(neuron)

        lang_layer_tensors = []
        for layer_neurons in layer_to_neurons:
            if len(layer_neurons) == 0:
                lang_layer_tensors.append(torch.empty(0, dtype=torch.long))
            else:
                layer_neurons_sorted = sorted(set(layer_neurons))
                lang_layer_tensors.append(
                    torch.tensor(layer_neurons_sorted, dtype=torch.long)
                )
        final_lang_neurons.append(lang_layer_tensors)

    total_neurons = num_layers * intermediate_size

    lape_per_lang: Dict[str, Dict[str, Any]] = {}
    for lang_id, lang in enumerate(LANGS):
        neuron_count = sum(
            len(layer_neurons) for layer_neurons in final_lang_neurons[lang_id]
        )
        percentage = (neuron_count / total_neurons) * 100.0
        lape_per_lang[lang] = {
            "lape_quantity": int(neuron_count),
            "lape_percentage": float(percentage),
        }

    param_count = 0
    for p in model.parameters():
        param_count += p.numel()

    model_info = {
        "model_name": MODEL_ID,
        "model_path": MODEL_PATH,
        "model_type": model_type,
        "num_layers": int(num_layers),
        "hidden_size": int(model.config.hidden_size),
        "intermediate_size": int(intermediate_size),
        "vocab_size": int(model.config.vocab_size),
        "num_parameters": int(param_count),
    }

    activation_data = {
        "model": model_info,
        "langs": LANGS,
        "tokens_per_lang": token_counts.clone(),
        "over_zero": over_zero.clone(),
        "activation_probs": activation_probs.clone(),
        "entropy": entropy.clone(),
        "top_rate": LAPE_TOP_RATE,
        "filter_rate": LAPE_FILTER_RATE,
        "activation_bar_ratio": LAPE_ACTIVATION_BAR_RATIO,
    }

    mask_data = {
        "model": model_info,
        "langs": LANGS,
        "tokens_per_lang": token_counts.clone(),
        "lang_neurons": final_lang_neurons,
        "top_rate": LAPE_TOP_RATE,
        "filter_rate": LAPE_FILTER_RATE,
        "activation_bar_ratio": LAPE_ACTIVATION_BAR_RATIO,
        "top_prob_value": float(top_prob_value),
        "activation_bar": float(activation_bar),
    }

    safe_name = safe_model_id(MODEL_ID)
    stats_path = os.path.join(model_out_dir, safe_name + "_activation_stats.pt")
    mask_path = os.path.join(model_out_dir, safe_name + "_lang_specific_neurons.pt")

    torch.save(activation_data, stats_path)
    torch.save(mask_data, mask_path)

    log(f"LAPE activation stats saved to: {stats_path}")
    log(f"LAPE language-specific neurons saved to: {mask_path}")

    return lape_per_lang


# ============================================================================
# Combined summary
# ============================================================================


def write_combined_summary(
    model_out_dir: str,
    tok_res: Dict[str, Dict[str, Any]],
    lape_res: Dict[str, Dict[str, Any]],
    logit_res: Dict[str, Dict[str, Any]],
    attn_res: Dict[str, Dict[str, Any]],
    div_res: Dict[str, Dict[str, Any]],
) -> str:
    summary_path = os.path.join(model_out_dir, "summary.csv")

    fields = [
        "LANGUAGE",
        "n questions",
        "n tokens",
        "LAPE quantity",
        "LAPE percentage",
        "total_words",
        "fertility (total_tokens / total_words)",
        "unique_tokens",
        "unique_token_fraction",
        "Characters per Token mean",
        "Characters per Token std",
        "shared_en_token_count",
        "shared_en_token_fraction",
        "shared_ru_token_count",
        "shared_ru_token_fraction",
        "shared_turkish_token_count",
        "shared_turkish_token_fraction",
        "perplexity mean",
        "perplexity std",
        "MaximumSequenceProbability mean",
        "MaximumSequenceProbability std",
        "MeanTokenEntropy mean",
        "MeanTokenEntropy std",
        "Recurrent Attention-based Uncertainty Quantification mean",
        "Recurrent Attention-based Uncertainty Quantification std",
        "Focus mean",
        "Focus std",
        "LexicalSimilarity mean",
        "LexicalSimilarity std",
        "DegMat mean",
        "DegMat std",
        "EigValLaplacian mean",
        "EigValLaplacian std",
        "Eccentricity mean",
        "Eccentricity std",
        "Eccentricity infs cnt",
    ]

    with open(summary_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()

        for lang in LANGS:
            t = tok_res.get(lang, {})
            lp = lape_res.get(lang, {})
            lg = logit_res.get(lang, {})
            at = attn_res.get(lang, {})
            dv = div_res.get(lang, {})

            row = {
                "LANGUAGE": lang,
                "n questions": t.get("n_questions", 0),
                "n tokens": t.get("n_tokens", 0),
                "LAPE quantity": lp.get("lape_quantity", "nan"),
                "LAPE percentage": fmt_float(lp.get("lape_percentage", float("nan"))),
                "total_words": t.get("total_words", 0),
                "fertility (total_tokens / total_words)": fmt_float(
                    t.get("fertility", float("nan"))
                ),
                "unique_tokens": t.get("unique_tokens", 0),
                "unique_token_fraction": fmt_float(
                    t.get("unique_token_fraction", float("nan"))
                ),
                "Characters per Token mean": fmt_float(
                    t.get("chars_per_token_mean", float("nan"))
                ),
                "Characters per Token std": fmt_float(
                    t.get("chars_per_token_std", float("nan"))
                ),
                "shared_en_token_count": t.get("shared_en_token_count", 0),
                "shared_en_token_fraction": fmt_float(
                    t.get("shared_en_token_fraction", float("nan"))
                ),
                "shared_ru_token_count": t.get("shared_ru_token_count", 0),
                "shared_ru_token_fraction": fmt_float(
                    t.get("shared_ru_token_fraction", float("nan"))
                ),
                "shared_turkish_token_count": t.get("shared_turkish_token_count", 0),
                "shared_turkish_token_fraction": fmt_float(
                    t.get("shared_turkish_token_fraction", float("nan"))
                ),
                "perplexity mean": fmt_float(lg.get("perplexity_mean", float("nan"))),
                "perplexity std": fmt_float(lg.get("perplexity_std", float("nan"))),
                "MaximumSequenceProbability mean": fmt_float(
                    lg.get("msp_mean", float("nan"))
                ),
                "MaximumSequenceProbability std": fmt_float(
                    lg.get("msp_std", float("nan"))
                ),
                "MeanTokenEntropy mean": fmt_float(
                    lg.get("entropy_mean", float("nan"))
                ),
                "MeanTokenEntropy std": fmt_float(lg.get("entropy_std", float("nan"))),
                "Recurrent Attention-based Uncertainty Quantification mean": fmt_float(
                    at.get("rauq_mean", float("nan"))
                ),
                "Recurrent Attention-based Uncertainty Quantification std": fmt_float(
                    at.get("rauq_std", float("nan"))
                ),
                "Focus mean": fmt_float(at.get("focus_mean", float("nan"))),
                "Focus std": fmt_float(at.get("focus_std", float("nan"))),
                "LexicalSimilarity mean": fmt_float(
                    dv.get("lexsim_mean", float("nan"))
                ),
                "LexicalSimilarity std": fmt_float(dv.get("lexsim_std", float("nan"))),
                "DegMat mean": fmt_float(dv.get("degmat_mean", float("nan"))),
                "DegMat std": fmt_float(dv.get("degmat_std", float("nan"))),
                "EigValLaplacian mean": fmt_float(dv.get("eigval_mean", float("nan"))),
                "EigValLaplacian std": fmt_float(dv.get("eigval_std", float("nan"))),
                "Eccentricity mean": fmt_float(dv.get("ecc_mean", float("nan"))),
                "Eccentricity std": fmt_float(dv.get("ecc_std", float("nan"))),
                "Eccentricity infs cnt": dv.get("ecc_inf_cnt", 0),
            }

            w.writerow(row)

    log(f"Combined summary saved to: {summary_path}")
    return summary_path


# ============================================================================
# MAIN
# ============================================================================


def main() -> None:
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    model_safe = safe_model_id(MODEL_ID)
    model_out_dir = os.path.join(OUTPUT_ROOT, model_safe)
    os.makedirs(model_out_dir, exist_ok=True)

    log("=" * 80)
    log(f"Processing model: {MODEL_ID}")
    log(f"MODEL_PATH: {MODEL_PATH}")
    log(f"DATA_ROOT: {DATA_ROOT}")
    log(f"MODEL_OUT_DIR: {model_out_dir}")
    log(f"torch.cuda.device_count() = {torch.cuda.device_count()}")
    log("=" * 80)

    # Ensure attentions are returned
    try:
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
    except Exception:
        pass

    torch.backends.cuda.matmul.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # --- Load tokenizers
    log("Loading tokenizer for generation/uncertainty/LAPE...")
    tokenizer_main = AutoTokenizer.from_pretrained(
        MODEL_PATH,
        use_fast=True,
        local_files_only=True,
    )

    tokenizer_main.padding_side = "left"

    if tokenizer_main.eos_token_id is None and tokenizer_main.eos_token is not None:
        tokenizer_main.eos_token_id = tokenizer_main.convert_tokens_to_ids(
            tokenizer_main.eos_token
        )

    if tokenizer_main.pad_token_id is None:
        if tokenizer_main.eos_token_id is not None:
            tokenizer_main.pad_token_id = tokenizer_main.eos_token_id
        elif tokenizer_main.eos_token is not None:
            tokenizer_main.pad_token = tokenizer_main.eos_token

    log(f"Main tokenizer vocab size: {len(tokenizer_main)}")
    log(
        f"EOS token id: {tokenizer_main.eos_token_id}, PAD token id: {tokenizer_main.pad_token_id}"
    )

    log("Loading tokenizer for tokenizer_analysis...")
    if TOKENIZER_ADD_PREFIX_SPACE:
        try:
            tokenizer_tok = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                use_fast=True,
                add_prefix_space=True,
                local_files_only=True,
            )
        except TypeError:
            tokenizer_tok = AutoTokenizer.from_pretrained(
                MODEL_PATH,
                use_fast=True,
                local_files_only=True,
            )
    else:
        tokenizer_tok = AutoTokenizer.from_pretrained(
            MODEL_PATH,
            use_fast=True,
            local_files_only=True,
        )

    if tokenizer_tok.pad_token is None and tokenizer_tok.eos_token is not None:
        tokenizer_tok.pad_token = tokenizer_tok.eos_token

    # --- Pass 1: compute budgets
    budgets = compute_common_token_budgets(
        tokenizer_main=tokenizer_main, tokenizer_tok=tokenizer_tok
    )

    # --- Tokenizer analysis
    tok_res = compute_tokenizer_analysis(
        tokenizer_tok=tokenizer_tok,
        model_out_dir=model_out_dir,
        common_tokens=budgets["common_tok"],
    )

    # --- Load model once
    log("Loading model (device_map='auto' for 2xA100, attn_implementation='eager')...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto" if torch.cuda.is_available() else None,
        torch_dtype="auto" if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",
    )

    if not torch.cuda.is_available():
        model.to(torch.device("cpu"))

    model.eval()

    # --- Unified uncertainty pass
    logit_res, attn_res, div_res = compute_uncertainty_metrics_one_pass(
        model=model,
        tokenizer=tokenizer_main,
        model_out_dir=model_out_dir,
        common_tokens=budgets["common_unc"],
    )

    # --- LAPE (hook-heavy)
    lape_res = compute_lape(
        model=model,
        tokenizer=tokenizer_main,
        model_out_dir=model_out_dir,
        common_tokens=budgets["common_lape"],
    )

    # --- Combined summary
    write_combined_summary(
        model_out_dir=model_out_dir,
        tok_res=tok_res,
        lape_res=lape_res,
        logit_res=logit_res,
        attn_res=attn_res,
        div_res=div_res,
    )

    log("Done.")


if __name__ == "__main__":
    main()
