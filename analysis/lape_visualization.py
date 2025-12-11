import math
import torch
import matplotlib.pyplot as plt
import os
MASK_PATH = "activation_results/Tweeties_tweety-tatar-base-7b-2024-v1_lang_specific_neurons.pt" 

data = torch.load(MASK_PATH, map_location="cpu")

model_info = data["model"]             # словарь с инфой о модели
langs = data["langs"]                  # список языков
lang_neurons = data["lang_neurons"]    # список длиной n_langs, для каждого языка: [layers] -> tensor(neuron_ids)

num_layers = model_info["num_layers"]
n_langs = len(langs)

# матрица counts[lang_idx, layer_idx] = число языковых нейронов в этом слое для этого языка
counts = []
for lang_idx in range(n_langs):
    layer_counts = []
    for layer_idx in range(num_layers):
        neurons_tensor = lang_neurons[lang_idx][layer_idx]
        layer_counts.append(len(neurons_tensor))
    counts.append(layer_counts)

# параметры сетки сабплотов
ncols = 3
nrows = math.ceil(n_langs / ncols)

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3 * nrows), sharex=True, sharey=True)
if nrows == 1 and ncols == 1:
    axes = [[axes]]
elif nrows == 1:
    axes = [axes]
elif ncols == 1:
    axes = [[ax] for ax in axes]

x = list(range(num_layers))

for idx, lang in enumerate(langs):
    row = idx // ncols
    col = idx % ncols
    ax = axes[row][col]

    ax.bar(x, counts[idx])
    ax.set_title(lang)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Num language-specific neurons")

# скрыть пустые сабплоты, если языков меньше, чем nrows * ncols
for idx in range(n_langs, nrows * ncols):
    row = idx // ncols
    col = idx % ncols
    fig.delaxes(axes[row][col])

fig.suptitle(
    f"Distribution of language-specific neurons per layer\nModel: {model_info['model_name']}",
    fontsize=14
)
plt.tight_layout(rect=[0, 0, 1, 0.95])

base = os.path.splitext(os.path.basename(MASK_PATH))[0]
out_path = "imgs/"+base + "_lang_neurons_per_layer.png"

plt.savefig(out_path, dpi=200)
plt.close()

print("Saved figure to:", out_path)
