import json
import os
import gc
import time
from typing import List, Dict, Any, Tuple

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


CSV_PATH = "data/parallel_corpora/crh_Latn_eng.csv"           
LANG1_COL = "crh_Latn"                        
LANG2_COL = "eng"                        

MODEL_NAME = "Tweeties/tweety-tatar-base-7b-2024-v1"
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 4
MAX_LENGTH = 512
LIMIT = 0  # 0 = без ограничения по числу строк

# LAPE-параметры
FILTER_RATE = 0.95          # Перцентиль по p(активации) для фильтра «существенных» нейронов
TOP_RATE = 0.01             # Доля нейронов с минимальной энтропией
ACTIVATION_BAR_RATIO = 0.95 # Планка p(активации) для приписывания языка нейрону


SAVE_JSON = f"activation_parallel/activation_results/json/{LANG1_COL}_{LANG2_COL}_{MODEL_NAME.replace('/', '_')}.json"
SAVE_MASK_PTH = f"activation_parallel/activation_results/pth/{LANG1_COL}_{LANG2_COL}_{MODEL_NAME.replace('/', '_')}.pth"


def load_texts(
    csv_path: str, c1: str, c2: str, limit: int
) -> Tuple[List[str], List[str]]:
    print(f"[1/8] Загрузка текстов из {csv_path}...")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Не найден CSV: {csv_path}")
    df = pd.read_csv(csv_path)
    if c1 not in df.columns or c2 not in df.columns:
        raise ValueError(
            f"В CSV должны быть колонки '{c1}' и '{c2}'. Найдены: {list(df.columns)}"
        )
    t1 = [
        str(x).strip() for x in df[c1].tolist() if isinstance(x, str) and str(x).strip()
    ]
    t2 = [
        str(x).strip() for x in df[c2].tolist() if isinstance(x, str) and str(x).strip()
    ]
    n = min(len(t1), len(t2))
    t1, t2 = t1[:n], t2[:n]
    if limit and limit > 0:
        t1, t2 = t1[:limit], t2[:limit]
    print(f"  Загружено {len(t1)} пар текстов (lang1: {c1}, lang2: {c2})")
    return t1, t2


def detect_mlp_kind(mlp) -> str:
    """
    Возвращает тип MLP:
      - "gated" для Mistral/LLaMA (gate_proj/up_proj/down_proj, SiLU),
      - "bloom" для BLOOM (dense_h_to_4h, gelu_impl, dense_4h_to_h).
    """
    if (
        hasattr(mlp, "gate_proj")
        and hasattr(mlp, "up_proj")
        and hasattr(mlp, "down_proj")
    ):
        return "gated"
    if hasattr(mlp, "dense_h_to_4h") and hasattr(mlp, "dense_4h_to_h"):
        return "bloom"
    raise RuntimeError("Неизвестный тип MLP-модуля (ожидался gated или bloom).")


def get_layers_and_intermediate(model) -> Tuple[List[Any], int]:
    """
    Возвращает список DecoderLayer и размер intermediate (должен быть постоянен по слоям).
    Поддержка Mistral/LLaMA (model.model.layers) и BLOOM (model.transformer.h).
    """
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
        first_mlp = layers[0].mlp
        kind = detect_mlp_kind(first_mlp)
        if kind == "gated":
            inter = first_mlp.gate_proj.out_features
        else:
            inter = first_mlp.dense_h_to_4h.out_features
        return layers, inter
    elif hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        layers = list(model.transformer.h)
        first_mlp = layers[0].mlp
        kind = detect_mlp_kind(first_mlp)
        if kind == "gated":
            inter = first_mlp.gate_proj.out_features
        else:
            inter = first_mlp.dense_h_to_4h.out_features
        return layers, inter
    else:
        raise RuntimeError(
            "Не удалось найти список слоёв модели (ожидались .model.layers или .transformer.h)."
        )


@torch.no_grad()
def run_pass_and_count(
    texts: List[str],
    tokenizer,
    model,
    device: torch.device,
    batch_size: int,
    max_length: int,
) -> int:
    """
    Прогоняет тексты через модель; в патченных forward слоях увеличивается счётчик over_zero[layer, unit]
    на число токенов, где активация после нелинейности > 0. Возвращает суммарное число обработанных токенов.
    """
    total_tokens = 0
    model.eval()
    num_batches = (len(texts) + batch_size - 1) // batch_size
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        total_tokens += int(attention_mask.sum().item())
        batch_idx = i // batch_size + 1
        if batch_idx % 10 == 0 or batch_idx == num_batches:
            print(
                f"    Обработано батчей: {batch_idx}/{num_batches} ({total_tokens} токенов)"
            )
    return total_tokens


def make_patchers(
    layers: List[Any], over_zero_ref: Dict[str, torch.Tensor]
) -> List[Any]:
    """
    Создаёт списки patched forward для каждого слоя, которые
    (1) воспроизводят исходный MLP-проход и (2) инкрементируют счётчики активаций.
    over_zero_ref["tensor"] должен указывать на текущий счётчик (для языка 1 или 2).
    """
    patched = []
    for li, layer in enumerate(layers):
        mlp = layer.mlp
        kind = detect_mlp_kind(mlp)

        if kind == "gated":
            gate, up, down = mlp.gate_proj, mlp.up_proj, mlp.down_proj
            act = mlp.act_fn  # SiLU

            def fwd(x, gate=gate, up=up, down=down, act=act, li=li):
                g = gate(x)  # (B, L, I)
                a = act(g)  # (B, L, I)
                ctr = over_zero_ref["tensor"]
                if ctr is not None:
                    pos = (a > 0).sum(dim=(0, 1))
                    ctr[li].add_(pos.to(ctr.dtype))
                u = up(x)  # (B, L, I)
                y = down(a * u)  # (B, L, H)
                return y

        else:  # bloom
            d14h, d4h1 = mlp.dense_h_to_4h, mlp.dense_4h_to_h
            gelu = mlp.gelu_impl

            def fwd(x, d14h=d14h, d4h1=d4h1, gelu=gelu, li=li):
                z = d14h(x)  # (B, L, 4H)
                a = gelu(z)  # (B, L, 4H)
                ctr = over_zero_ref["tensor"]
                if ctr is not None:
                    pos = (a > 0).sum(dim=(0, 1))
                    ctr[li].add_(pos.to(ctr.dtype))
                y = d4h1(a)
                return y

        patched.append((mlp, mlp.forward, fwd))
    return patched


def apply_patches(patched: List[Any]) -> None:
    for mlp, orig, fwd in patched:
        mlp.forward = fwd


def restore_patches(patched: List[Any]) -> None:
    for mlp, orig, fwd in patched:
        mlp.forward = orig


def layerwise_indices_from_mask(mask_tensor: torch.Tensor) -> List[List[int]]:
    out: List[List[int]] = []
    for li in range(mask_tensor.size(0)):
        idx = torch.nonzero(mask_tensor[li], as_tuple=False).squeeze(-1)
        out.append([int(i.item()) for i in idx])
    return out


def main():
   
    LANG1_COL_LIST = ["crh_Latn", 'crh_Cyrl']                       
    LANG2_COL_LIST = ["eng", "rus", "ukr"]    
    MODEL_NAME_LIST = ['Tweeties/tweety-tatar-base-7b-2024-v1', 'mistralai/Mistral-7B-Instruct-v0.2']
    for LANG1_COL in LANG1_COL_LIST:
        for LANG2_COL in LANG2_COL_LIST:
            for MODEL_NAME in MODEL_NAME_LIST:
                CSV_PATH = f"data/parallel_corpora/{LANG1_COL}_{LANG2_COL}.csv"  
                SAVE_JSON = f"activation_parallel/activation_results/json/{LANG1_COL}_{LANG2_COL}_{MODEL_NAME.replace('/', '_')}.json"
                SAVE_MASK_PTH = f"activation_parallel/activation_results/pth/{LANG1_COL}_{LANG2_COL}_{MODEL_NAME.replace('/', '_')}.pth"
    
                device = torch.device(DEVICE)

                # 1) Данные
                lang1_texts, lang2_texts = load_texts(
                    CSV_PATH, LANG1_COL, LANG2_COL, LIMIT
                )

                # 2) Модель и токенайзер
                print(f"[2/8] Загрузка токенайзера из {MODEL_NAME}...")
                tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                print(f"[3/8] Загрузка модели {MODEL_NAME}...")
                model = AutoModelForCausalLM.from_pretrained(
                    MODEL_NAME,
                    torch_dtype=(torch.float16 if device.type == "cuda" else torch.float32),
                    low_cpu_mem_usage=True,
                ).to(device)
                model.config.pad_token_id = tokenizer.pad_token_id

                layers, intermediate_size = get_layers_and_intermediate(model)
                num_layers = len(layers)
                total_neurons = num_layers * intermediate_size
                print(
                    f"  Модель загружена: {num_layers} слоёв, {intermediate_size} нейронов/слой, всего {total_neurons} нейронов"
                )

                # 3) Счётчики активаций для обоих языков
                over_zero_1 = torch.zeros(
                    (num_layers, intermediate_size), dtype=torch.int64, device=device
                )
                over_zero_2 = torch.zeros(
                    (num_layers, intermediate_size), dtype=torch.int64, device=device
                )

                # 4) Патчим forward, прогоняем язык 1
                print(f"[4/8] Прогон текстов lang1 ({LANG1_COL}) через модель...")
                over_zero_ref = {"tensor": over_zero_1}
                patched = make_patchers(layers, over_zero_ref)
                apply_patches(patched)
                tokens_1 = run_pass_and_count(
                    lang1_texts,
                    tokenizer,
                    model,
                    device,
                    BATCH_SIZE,
                    MAX_LENGTH,
                )
                print(f"  Lang1 завершён: {tokens_1} токенов обработано")

                # 5) Патч остаётся, меняем счётчик и прогоняем язык 2
                print(f"[5/8] Прогон текстов lang2 ({LANG2_COL}) через модель...")
                over_zero_ref["tensor"] = over_zero_2
                tokens_2 = run_pass_and_count(
                    lang2_texts,
                    tokenizer,
                    model,
                    device,
                    BATCH_SIZE,
                    MAX_LENGTH,
                )
                print(f"  Lang2 завершён: {tokens_2} токенов обработано")

                # (не обязательно) восстановить исходный forward
                # restore_patches(patched)

                # 6) Вероятности активаций p(>0) по языкам
                print(f"[6/8] Вычисление вероятностей активаций и энтропии...")
                eps = 1e-12
                p1 = over_zero_1.float() / max(1, tokens_1)
                p2 = over_zero_2.float() / max(1, tokens_2)

                # 7) Нормировка и энтропия (LAPE-style) для двух языков
                probs = torch.stack([p1, p2], dim=-1)  # (L, I, 2)
                probs_sum = probs.sum(dim=-1, keepdim=True).clamp_min(eps)
                probs_norm = probs / probs_sum
                entropy = -(probs_norm * (probs_norm.clamp_min(eps).log())).sum(dim=-1)  # (L, I)

                # 8) Фильтр «существенных» по максимуму вероятности
                flat_probs = probs.view(-1)
                k_filter = max(1, int(round(len(flat_probs) * FILTER_RATE)))
                top_prob_value = torch.kthvalue(flat_probs, k_filter).values.item()
                max_over_langs = probs.max(dim=-1).values  # (L, I)
                entropy_filtered = entropy.clone()
                mask_keep = max_over_langs > top_prob_value  # True = оставить
                entropy_filtered[~mask_keep] = float("inf")  # исключаем слабые нейроны

                # 9) Отбор нижних top_rate по энтропии
                print(
                    f"[7/8] Отбор языко-специфичных нейронов (top {TOP_RATE*100:.1f}% по энтропии)..."
                )
                flat_entropy = entropy_filtered.view(-1)
                k_top = max(1, int(round(len(flat_entropy) * TOP_RATE)))
                top_vals, top_idx = torch.topk(-flat_entropy, k=k_top, largest=True)  # мин энтропия
                row_index = top_idx // entropy.size(1)  # слои
                col_index = top_idx % entropy.size(1)  # нейроны
                selected_probs = probs[row_index, col_index]  # (k_top, 2) — вероятности по двум языкам

                # 10) Порог для приписывания языка нейрону
                k_bar = max(1, int(round(len(flat_probs) * ACTIVATION_BAR_RATIO)))
                activation_bar = torch.kthvalue(flat_probs, k_bar).values.item()

                # Для каждого из отобранных нейронов отмечаем, где p(lang) > activation_bar
                selected_probs_T = selected_probs.t()  # (2, k_top)
                lang_idx, neu_idx = torch.where(selected_probs_T > activation_bar)  # lang ∈ {0,1}

                # Собираем списки индексов по слоям для каждого языка
                merged = torch.stack((row_index, col_index), dim=-1)  # (k_top, 2)
                by_lang: List[List[torch.LongTensor]] = []
                for LID in [0, 1]:
                    # индексы в selected-массиве, которые принадлежат этому языку
                    sel = merged[neu_idx[lang_idx == LID]]
                    # сортировка и разбиение по слоям
                    layer_lists = [[] for _ in range(num_layers)]
                    for l, h in sel.tolist():
                        layer_lists[l].append(h)
                    layer_tensors = [torch.tensor(sorted(v), dtype=torch.long) for v in layer_lists]
                    by_lang.append(layer_tensors)

                # 11) Дополнительная «наивная» статистика пересечений по факту >0
                active1_any = over_zero_1 > 0
                active2_any = over_zero_2 > 0
                only1 = active1_any & (~active2_any)
                only2 = active2_any & (~active1_any)
                both = active1_any & active2_any

                # 12) Топ «языко-перекошенных» нейронов по нормализованной разнице
                diff_rate = (p1 - p2).view(-1)
                flat_p1 = p1.view(-1)
                flat_p2 = p2.view(-1)
                top_k = diff_rate.numel()
                top_vals2, top_idx2 = torch.topk(diff_rate, k=top_k, largest=True, sorted=True)
                top_list = []
                for val, idx in zip(top_vals2.tolist(), top_idx2.tolist()):
                    layer = idx // intermediate_size
                    neuron = idx % intermediate_size
                    top_list.append(
                        {
                            "layer": int(layer),
                            "neuron": int(neuron),
                            "lang1_rate": float(flat_p1[idx].item()),
                            "lang2_rate": float(flat_p2[idx].item()),
                            "rate_diff": float(val),
                        }
                    )

                # 13) Готовим выводы и сохраняем
                print(f"[8/8] Сохранение результатов...")
                result: Dict[str, Any] = {
                    "model": MODEL_NAME,
                    "csv": CSV_PATH,
                    "columns": {"lang1": LANG1_COL, "lang2": LANG2_COL},
                    "device": str(device),
                    "num_layers": num_layers,
                    "intermediate_size": intermediate_size,
                    "tokens_processed": {"lang1": int(tokens_1), "lang2": int(tokens_2)},
                    "neuron_totals": {
                        "total_neurons": int(total_neurons),
                        "lang1_active_any": int(active1_any.sum().item()),
                        "lang2_active_any": int(active2_any.sum().item()),
                        "both_active": int(both.sum().item()),
                        "only_lang1": int(only1.sum().item()),
                        "only_lang2": int(only2.sum().item()),
                    },
                    "thresholds": {
                        "filter_rate": FILTER_RATE,
                        "top_rate": TOP_RATE,
                        "activation_bar_ratio": ACTIVATION_BAR_RATIO,
                        "activation_bar_value": activation_bar,
                        "top_prob_value": top_prob_value,
                    },
                    "layerwise_indices_any": {
                        "only_lang1": layerwise_indices_from_mask(only1),
                        "only_lang2": layerwise_indices_from_mask(only2),
                        "both": layerwise_indices_from_mask(both),
                    },
                    # маски языко-специфичных нейронов (LAPE-style отобранные, p(lang)>bar)
                    "language_specific_masks": {
                        "lang1": [v.tolist() for v in by_lang[0]],
                        "lang2": [v.tolist() for v in by_lang[1]],
                    },
                    "top_lang1_specific_by_rate_diff": top_list,
                }

                with open(SAVE_JSON, "w", encoding="utf-8") as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)

                # Сохраняем pth в формате как у авторов: List[ per-language ] of List[ per-layer ] of LongTensor
                torch.save([by_lang[0], by_lang[1]], SAVE_MASK_PTH)

                print("\n=== DONE ===")
                print(f"Модель: {MODEL_NAME}")
                print(
                    f"Слои: {num_layers} | intermediate_size: {intermediate_size} | всего нейронов: {total_neurons}"
                )
                print(f"Токенов обработано: lang1={int(tokens_1)}, lang2={int(tokens_2)}")
                print(
                    f"Активные (≥1 раз): lang1={int(active1_any.sum().item())}, lang2={int(active2_any.sum().item())}, оба={int(both.sum().item())}"
                )
                print(
                    f"Языко-специфичные (по порогу {ACTIVATION_BAR_RATIO:.2f} и топ-{TOP_RATE*100:.1f}% низк. энтропии) — "
                    f"lang1={sum(len(v) for v in result['language_specific_masks']['lang1'])}, "
                    f"lang2={sum(len(v) for v in result['language_specific_masks']['lang2'])}"
                )
                print(f"JSON: {SAVE_JSON}")
                print(f"Masks .pth: {SAVE_MASK_PTH}")
                
                # Очистка памяти после обработки модели
                print("\n=== Очистка памяти ===")
                if torch.cuda.is_available():
                    print(f"Память до очистки: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
                
                # Восстанавливаем патчи перед удалением
                restore_patches(patched)
                
                # Удаляем большие тензоры
                del over_zero_1, over_zero_2
                del p1, p2, probs, probs_norm, entropy, entropy_filtered
                del active1_any, active2_any, only1, only2, both
                del flat_probs, flat_entropy, selected_probs
                del diff_rate, flat_p1, flat_p2
                del row_index, col_index, selected_probs_T
                del lang_idx, neu_idx, merged, by_lang
                del top_vals, top_idx, top_vals2, top_idx2
                del max_over_langs, mask_keep
                del patched, layers
                
                # Удаляем результат (уже сохранён)
                del result, top_list
                
                # Переносим модель на CPU перед удалением (освобождает GPU память)
                model.cpu()
                
                # Удаляем модель и токенайзер
                del model
                del tokenizer
                
                # Принудительная очистка кэша CUDA
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                
                # Запуск сборщика мусора несколько раз
                gc.collect()
                gc.collect()
                
                # Ещё одна очистка CUDA кэша
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                # Даём время GPU драйверу освободить память
                time.sleep(2)
                
                if torch.cuda.is_available():
                    print(f"Память после очистки: {torch.cuda.memory_allocated()/1024**3:.2f} GB allocated, {torch.cuda.memory_reserved()/1024**3:.2f} GB reserved")
                
                print("Память очищена\n")


if __name__ == "__main__":
    main()
