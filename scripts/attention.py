import warnings

warnings.filterwarnings("ignore")

import os
import json
import csv
import math
from datetime import datetime
from typing import Dict, Any, List, Tuple

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.utils import logging as hf_logging

hf_logging.set_verbosity_error()


# -----------------------------
# CONFIG
# -----------------------------
MODEL_ID = "google/gemma-3-12b-it"
MODEL_PATH = "/hf_models"  # mounted from model_registry

DATA_ROOT = "/work/benchmarks/TUMLU"
OUTPUT_DIR = "/work/benchmarks/uncertainty_metrics"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Greedy generation
MAX_NEW_TOKENS = 64
LOG_GREEDY_ANSWER = True

# Focus hyperparams
FOCUS_GAMMA = 0.9
FOCUS_RHO = 0.01

# mark tokens whose IDF is >= quantile among generated tokens
FOCUS_KW_IDF_QUANTILE = 0.75

# IDF computed per-language, only from that language file (independent across languages)
IDF_MAX_DOCS = (
    -1
)  # -1 => all docs of language file (can be heavy but faithful & independent)

# RAUQ hyperparams:
RAUQ_ALPHA = 0.2  # in RAUQ paper: alpha=0.2 for QA-type tasks; robust single hyperparam

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


# -----------------------------
# Stats helpers
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
# IDF (per language, independent)
# -----------------------------
def compute_or_load_idf_for_lang(
    tokenizer: AutoTokenizer,
    lang: str,
    lang_jsonl_path: str,
    cache_dir: str,
    max_docs: int,
) -> np.ndarray:
    """
    IDF over *token ids* for this tokenizer, computed only from this language file.
    Document = question + choices (no prompt template, no chat template) to reduce template artifacts.
    IDF(t) = log((N+1)/(df+1))  (non-negative, stable).
    """
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
        # fallback: all ones (no effect)
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


# -----------------------------
# Greedy generation (single)
# -----------------------------
@torch.inference_mode()
def generate_greedy_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    model_input_text: str,
    max_new_tokens: int,
) -> Tuple[torch.Tensor, int, int]:
    """
    Returns:
      full_seq: (L,) token ids = prompt + generated (cut at first EOS, includes EOS if generated)
      prompt_len: int
      gen_len: int (includes EOS if it was generated)
    """
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
    return full_seq, prompt_len, gen_len


# -----------------------------
# Focus (language-agnostic keywords via IDF quantile) + RAUQ
# -----------------------------
@torch.inference_mode()
def compute_focus_and_rauq_single(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    full_seq: torch.Tensor,  # (L,)
    prompt_len: int,
    gen_len: int,
    idf_t: torch.Tensor,  # (V,) float32 on model.device
    focus_gamma: float,
    focus_rho: float,
    focus_kw_idf_quantile: float,
    rauq_alpha: float,
) -> Dict[str, float]:
    """
    Returns dict with:
      Focus, RAUQ

    NOTE: Both metrics require attentions. If attentions are not returned (Flash/SDPA),
    we'll return NaNs (and you should force eager attention implementation).
    """
    if gen_len <= 0 or prompt_len <= 0:
        return {"Focus": float("nan"), "RAUQ": float("nan")}

    eps = 1e-12

    input_ids = full_seq.unsqueeze(0)  # (1, L)
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

    logits = out.logits.to(dtype=torch.float32)  # (1, L, V)

    attentions = out.attentions
    if attentions is None or len(attentions) == 0:
        # This happens with FlashAttention / SDPA fast paths; must force eager attention.
        return {"Focus": float("nan"), "RAUQ": float("nan")}

    num_layers = len(attentions)
    num_heads = attentions[0].shape[1]
    L = logits.shape[1]
    V = logits.shape[2]

    # indices in full sequence for generated tokens
    gen_start = prompt_len
    gen_end = prompt_len + gen_len
    if gen_end > L:
        gen_end = L
        gen_len = gen_end - gen_start
    if gen_len <= 0:
        return {"Focus": float("nan"), "RAUQ": float("nan")}

    # probabilities for each generated token: positions predicted by logits at [gen_start-1 .. gen_end-2]
    pred_start = gen_start - 1
    pred_end = gen_end - 1
    if pred_start < 0 or pred_end <= pred_start:
        return {"Focus": float("nan"), "RAUQ": float("nan")}

    logits_pred = logits[0, pred_start:pred_end, :]  # (gen_len, V)
    target = input_ids[0, gen_start:gen_end].to(dtype=torch.long)  # (gen_len,)

    logp_all = torch.log_softmax(logits_pred, dim=-1)  # (gen_len, V)
    p_all = torch.exp(logp_all)  # (gen_len, V)
    p_tok = p_all.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)  # (gen_len,)
    p_tok = torch.clamp(p_tok, min=eps)

    # -------------------------
    # Focus probability correction: candidate set via rho, then IDF reweighting
    # -------------------------
    p_tilde = p_all.clone()
    p_tilde = torch.where(p_tilde >= focus_rho, p_tilde, torch.zeros_like(p_tilde))
    denom = p_tilde.sum(dim=-1, keepdim=True)
    fallback_mask = denom.squeeze(-1) <= eps
    if fallback_mask.any():
        p_tilde[fallback_mask] = p_all[fallback_mask]
        denom = p_tilde.sum(dim=-1, keepdim=True)

    p_tilde = p_tilde / torch.clamp(denom, min=eps)

    # apply IDF (eq 8)
    if idf_t.shape[0] != V:
        p_hat = p_tilde
    else:
        p_idf = p_tilde * idf_t.view(1, -1)
        denom2 = p_idf.sum(dim=-1, keepdim=True)
        fallback2 = denom2.squeeze(-1) <= eps
        if fallback2.any():
            p_idf[fallback2] = p_tilde[fallback2]
            denom2 = p_idf.sum(dim=-1, keepdim=True)
        p_hat = p_idf / torch.clamp(denom2, min=eps)

    # Focus token score hi = -log p_hat(t_i) + H_i, with H_i = 2^{entropy_base2}
    p_hat_tok = p_hat.gather(dim=-1, index=target.unsqueeze(-1)).squeeze(-1)
    p_hat_tok = torch.clamp(p_hat_tok, min=eps)
    nll = -torch.log(p_hat_tok)  # natural log

    ent2 = -(p_hat * torch.log2(torch.clamp(p_hat, min=eps))).sum(
        dim=-1
    )  # base-2 entropy
    H = torch.pow(2.0, ent2)  # 2^{entropy}
    h = nll + H  # (gen_len,)

    # attention max-pooling across layers and heads
    att_stack = torch.stack(
        [a[0].to(dtype=torch.float32) for a in attentions], dim=0
    )  # (layers, heads, L, L)

    # safety against any NaN/inf in attentions
    if torch.isnan(att_stack).any() or torch.isinf(att_stack).any():
        att_stack = torch.nan_to_num(att_stack, nan=0.0, posinf=0.0, neginf=0.0)

    att_pool = torch.amax(att_stack, dim=(0, 1))  # (L, L)
    # restrict to generated tokens only (response r)
    att_r = att_pool[
        gen_start:gen_end, gen_start:gen_end
    ].contiguous()  # (gen_len, gen_len)

    # -------------------------
    # Focus keywords (no spaCy): top-IDF quantile among generated tokens
    # -------------------------
    special_ids = set()
    for tid in [
        getattr(tokenizer, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
        getattr(tokenizer, "bos_token_id", None),
    ]:
        if tid is not None:
            special_ids.add(int(tid))

    gen_token_ids = target.detach().cpu().numpy().tolist()
    gen_idf_vals = []
    for tid in gen_token_ids:
        if tid in special_ids or tid < 0 or tid >= idf_t.shape[0]:
            gen_idf_vals.append(float("-inf"))
        else:
            gen_idf_vals.append(float(idf_t[tid].item()))

    idf_arr = np.array(gen_idf_vals, dtype=np.float32)
    finite = np.isfinite(idf_arr)
    kw_mask = np.zeros((gen_len,), dtype=bool)
    if finite.any():
        thr = float(np.quantile(idf_arr[finite], focus_kw_idf_quantile))
        kw_mask = (idf_arr >= thr) & finite
        if not kw_mask.any():
            kw_mask[int(np.argmax(idf_arr))] = True

    # -------------------------
    # Focus propagation (eq 4-6): only between keywords
    # -------------------------
    h_hat = torch.zeros_like(h)
    kw_idx = np.where(kw_mask)[0].tolist()
    kw_set = set(kw_idx)

    for i in range(gen_len):
        base = h[i]
        if i not in kw_set:
            h_hat[i] = base
            continue

        prev_kw = [j for j in kw_idx if j < i]
        if not prev_kw:
            h_hat[i] = base
            continue

        att_row = att_r[i]  # (gen_len,)
        att_prev = att_row[
            torch.tensor(prev_kw, device=att_row.device, dtype=torch.long)
        ]
        denom_w = torch.sum(att_prev)
        if float(denom_w.item()) <= eps:
            h_hat[i] = base
            continue

        w = att_prev / torch.clamp(denom_w, min=eps)
        prev_scores = h_hat[
            torch.tensor(prev_kw, device=h_hat.device, dtype=torch.long)
        ]
        penalty = torch.sum(w * prev_scores)
        h_hat[i] = base + float(focus_gamma) * penalty

    # sentence-level Focus score: average over keywords (eq 3)
    if len(kw_idx) == 0:
        focus_score = float("nan")
    else:
        focus_score = float(
            h_hat[torch.tensor(kw_idx, device=h_hat.device)].mean().item()
        )

    # -------------------------
    # RAUQ (eq 1-4)
    # -------------------------
    mid_start = num_layers // 3
    mid_end = int(math.ceil(num_layers * 2.0 / 3.0))
    layer_set = list(range(mid_start, mid_end))

    rauq_layer_scores: List[float] = []
    gen_positions = list(range(gen_start, gen_end))

    for l in layer_set:
        A_l = attentions[l][0].to(dtype=torch.float32)  # (heads, L, L)
        if torch.isnan(A_l).any() or torch.isinf(A_l).any():
            A_l = torch.nan_to_num(A_l, nan=0.0, posinf=0.0, neginf=0.0)

        # head selection by mean attention to preceding token in the response (eq 1)
        if gen_len < 2:
            head_best = 0
            a_prev = torch.zeros((0,), device=A_l.device, dtype=torch.float32)
        else:
            means = []
            for h_idx in range(num_heads):
                vals = []
                for i in range(1, gen_len):
                    gi = gen_positions[i]
                    gj = gen_positions[i - 1]
                    vals.append(A_l[h_idx, gi, gj])
                means.append(torch.stack(vals).mean())
            head_best = int(torch.argmax(torch.stack(means)).item())

            a_vals = []
            for i in range(1, gen_len):
                gi = gen_positions[i]
                gj = gen_positions[i - 1]
                a_vals.append(A_l[head_best, gi, gj])
            a_prev = torch.stack(a_vals)  # (gen_len-1,)

        # recurrent confidence (eq 2)
        alpha = float(rauq_alpha)
        c = torch.zeros((gen_len,), device=p_tok.device, dtype=torch.float32)
        c[0] = p_tok[0].to(dtype=torch.float32)

        if gen_len >= 2:
            for i in range(1, gen_len):
                cur_p = p_tok[i].to(dtype=torch.float32)
                att_w = a_prev[i - 1].to(dtype=torch.float32)
                c[i] = alpha * cur_p + (1.0 - alpha) * att_w * c[i - 1]

        c = torch.clamp(c, min=eps)
        u_l = -torch.log(c).mean()  # eq 3
        rauq_layer_scores.append(float(u_l.item()))

    rauq_score = float(np.max(rauq_layer_scores)) if rauq_layer_scores else float("nan")
    return {"Focus": focus_score, "RAUQ": rauq_score}


# -----------------------------
# MAIN
# -----------------------------
def main() -> None:
    # Make SDPA fall back to math (attentions are usually not available on flash/mem-efficient paths)
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

    tokenizer.padding_side = "left"

    if tokenizer.eos_token_id is None and tokenizer.eos_token is not None:
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token_id is not None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        elif tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token

    log(f"Vocab size: {len(tokenizer)}")
    log(
        f"EOS token id: {tokenizer.eos_token_id}, PAD token id: {tokenizer.pad_token_id}"
    )
    log(f"tokenizer.padding_side = {tokenizer.padding_side}")
    log("")

    log(
        "Loading model (device_map='auto') with eager attention for output_attentions..."
    )
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        device_map="auto",
        torch_dtype="auto",
        low_cpu_mem_usage=True,
        local_files_only=True,
        attn_implementation="eager",  # IMPORTANT: ensures attentions are returned
    )
    model.eval()
    log(f"Model loaded. torch.cuda.device_count() = {torch.cuda.device_count()}")
    log("")

    # ---------- First pass: total prompt tokens per language (full corpora)
    log("First pass: counting total prompt tokens per language (full corpora)...")
    total_tokens_raw: Dict[str, int] = {}

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log(f"[{lang}] scanning {path}")

        total_tokens = 0
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

        total_tokens_raw[lang] = total_tokens
        log(f"  total_tokens_raw = {total_tokens}")

    common_tokens = min(total_tokens_raw.values()) if total_tokens_raw else 0
    log(f"\nCommon token budget per language (min over langs): {common_tokens}")
    log("")

    # ---------- Second pass: compute RAUQ + Focus
    log("Second pass: estimating RAUQ + Focus (up to common_tokens per language)...")

    stats_per_lang: Dict[str, Dict[str, Any]] = {
        lang: {
            "n_examples": 0,
            "used_tokens": 0,
            "RAUQ_values": [],
            "Focus_values": [],
        }
        for lang in LANGS
    }

    model_id_safe = safe_model_id(MODEL_ID)
    idf_cache_dir = os.path.join(OUTPUT_DIR, f"{model_id_safe}_idf_cache")

    for lang in LANGS:
        path = os.path.join(DATA_ROOT, lang, "all_shuffled.jsonl")
        log(f"[{lang}] processing {path}")
        log(f"[{lang}] target token budget: {common_tokens}")

        # IDF for this language only (independent)
        log(f"[{lang}] computing/loading IDF (max_docs={IDF_MAX_DOCS})...")
        idf_np = compute_or_load_idf_for_lang(
            tokenizer=tokenizer,
            lang=lang,
            lang_jsonl_path=path,
            cache_dir=idf_cache_dir,
            max_docs=IDF_MAX_DOCS,
        )
        idf_t = torch.tensor(idf_np, device=model.device, dtype=torch.float32)

        used_tokens = 0

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if used_tokens >= common_tokens:
                    break

                line = line.strip()
                if not line:
                    continue

                obj = json.loads(line)
                user_prompt = make_user_prompt(lang, obj)
                model_input = apply_chat_if_available(tokenizer, user_prompt)

                token_ids = tokenizer.encode(model_input, add_special_tokens=False)
                text_tokens = int(len(token_ids))

                if used_tokens + text_tokens > common_tokens:
                    break

                log(
                    f"[{lang}] example: {text_tokens} prompt tokens, cumulative: {used_tokens}/{common_tokens}"
                )

                if LOG_GREEDY_ANSWER:
                    log(f"[{lang}] PROMPT:\n{user_prompt}")

                full_seq, prompt_len, gen_len = generate_greedy_single(
                    model=model,
                    tokenizer=tokenizer,
                    model_input_text=model_input,
                    max_new_tokens=MAX_NEW_TOKENS,
                )

                if LOG_GREEDY_ANSWER:
                    gen_tokens = full_seq[prompt_len:]
                    if (
                        tokenizer.eos_token_id is not None
                        and gen_tokens.numel() > 0
                        and int(gen_tokens[-1].item()) == int(tokenizer.eos_token_id)
                    ):
                        gen_tokens = gen_tokens[:-1]
                    ans = tokenizer.decode(gen_tokens, skip_special_tokens=True)
                    log(f"[{lang}] ANSWER:\n{ans}")

                m = compute_focus_and_rauq_single(
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

                st = stats_per_lang[lang]
                st["n_examples"] += 1
                st["RAUQ_values"].append(float(m["RAUQ"]))
                st["Focus_values"].append(float(m["Focus"]))

                used_tokens += text_tokens
                st["used_tokens"] = used_tokens

                log(f"[{lang}] METRICS: RAUQ={m['RAUQ']:.6f}, Focus={m['Focus']:.6f}")
                log("")

        st = stats_per_lang[lang]
        log(
            f"[{lang}] done: n_examples={st['n_examples']}, used_tokens={st['used_tokens']}"
        )
        log("")

    # ---------- Save summary
    output_path = os.path.join(OUTPUT_DIR, f"{model_id_safe}_rauq_focus_summary.tsv")
    log(f"Saving aggregated metrics to {output_path}")

    with open(output_path, "w", encoding="utf-8", newline="") as out_f:
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
            st = stats_per_lang[lang]
            rauq_m, rauq_s = mean_and_std(st["RAUQ_values"])
            foc_m, foc_s = mean_and_std(st["Focus_values"])

            def fmt(x: float) -> str:
                return f"{x:.6f}" if not math.isnan(x) else "nan"

            writer.writerow(
                [
                    lang,
                    st["n_examples"],
                    st["used_tokens"],
                    fmt(rauq_m),
                    fmt(rauq_s),
                    fmt(foc_m),
                    fmt(foc_s),
                ]
            )

    stats_json_path = os.path.join(
        OUTPUT_DIR, f"{model_id_safe}_rauq_focus_stats_per_lang.json"
    )
    log(f"Saving raw stats_per_lang to: {stats_json_path}")
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(stats_per_lang, f, indent=2, ensure_ascii=False)

    log(f"Saved stats for {len(stats_per_lang)} languages")
    log("Done.")


if __name__ == "__main__":
    main()
