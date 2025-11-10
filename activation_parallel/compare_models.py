import json
from typing import Dict, Any, Tuple, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --------- ВИЗУАЛЬНЫЕ НАСТРОЙКИ ---------
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 10

LANG1_COL = "crh_Cyrl"
LANG2_COL = "eng"
NAME1 = "Tweeties/tweety-tatar-base-7b-2024-v1"
NAME2 = "mistralai/Mistral-7B-Instruct-v0.2"
JSON1_PATH = f"activation_parallel/activation_results/json/{LANG1_COL}_{LANG2_COL}_{NAME1.replace('/', '_')}.json"  # первый JSON
JSON2_PATH = f"activation_parallel/activation_results/json/{LANG1_COL}_{LANG2_COL}_{NAME2.replace('/', '_')}.json"  # второй JSON

OUTPUT_PREFIX = f"activation_parallel/activation_results/comparison/{LANG1_COL}_{LANG2_COL}_{NAME1.replace('/', '_')}_vs_{NAME2.replace('/', '_')}"                 # префикс для файлов

def load_json(path: str) -> Dict[str, Any]:
    """Load JSON file."""
    print(f"Загрузка {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def compare_basic_stats(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> str:
    """Generate basic statistics comparison."""
    report = []
    report.append("=" * 80)
    report.append("БАЗОВАЯ СТАТИСТИКА МОДЕЛЕЙ")
    report.append("=" * 80)
    
    # Model info
    report.append(f"\n{'Параметр':<40} {name1:<20} {name2:<20}")
    report.append("-" * 80)
    report.append(f"{'Модель':<40} {data1['model'][:18]:<20} {data2['model'][:18]:<20}")
    report.append(f"{'Количество слоёв':<40} {data1['num_layers']:<20} {data2['num_layers']:<20}")
    report.append(f"{'Размер intermediate':<40} {data1['intermediate_size']:<20} {data2['intermediate_size']:<20}")
    report.append(f"{'Всего нейронов':<40} {data1['neuron_totals']['total_neurons']:<20} {data2['neuron_totals']['total_neurons']:<20}")
    
    # Token processing
    report.append(f"\n{'ТОКЕНЫ ОБРАБОТАНО':<40}")
    report.append("-" * 80)
    t1_lang1 = data1['tokens_processed']['lang1']
    t1_lang2 = data1['tokens_processed']['lang2']
    t2_lang1 = data2['tokens_processed']['lang1']
    t2_lang2 = data2['tokens_processed']['lang2']
    
    report.append(f"{'Lang1 (' + lang1_name + ')':<40} {t1_lang1:<20,} {t2_lang1:<20,}")
    report.append(f"{'Lang2 (' + lang2_name + ')':<40} {t1_lang2:<20,} {t2_lang2:<20,}")
    report.append(f"{'Всего':<40} {t1_lang1+t1_lang2:<20,} {t2_lang1+t2_lang2:<20,}")
    
    # Neuron activation
    report.append(f"\n{'АКТИВАЦИЯ НЕЙРОНОВ':<40}")
    report.append("-" * 80)
    n1 = data1['neuron_totals']
    n2 = data2['neuron_totals']
    
    report.append(f"{lang1_name + ' активных (хотя бы раз)':<40} {n1['lang1_active_any']:<20,} {n2['lang1_active_any']:<20,}")
    report.append(f"{lang2_name + ' активных (хотя бы раз)':<40} {n1['lang2_active_any']:<20,} {n2['lang2_active_any']:<20,}")
    report.append(f"{'Оба языка':<40} {n1['both_active']:<20,} {n2['both_active']:<20,}")
    report.append(f"{'Только ' + lang1_name:<40} {n1['only_lang1']:<20,} {n2['only_lang1']:<20,}")
    report.append(f"{'Только ' + lang2_name:<40} {n1['only_lang2']:<20,} {n2['only_lang2']:<20,}")
    
    # Percentage
    pct1_both = 100 * n1['both_active'] / n1['total_neurons']
    pct2_both = 100 * n2['both_active'] / n2['total_neurons']
    report.append(f"{'% активных в обоих языках':<40} {pct1_both:<20.2f}% {pct2_both:<20.2f}%")
    
    return "\n".join(report)


def compare_language_specific_neurons(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> str:
    """Compare language-specific neurons identified by LAPE method."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("ЯЗЫКО-СПЕЦИФИЧНЫЕ НЕЙРОНЫ (LAPE метод)")
    report.append("=" * 80)
    
    masks1 = data1['language_specific_masks']
    masks2 = data2['language_specific_masks']
    
    # Count neurons per language
    count1_lang1 = sum(len(layer) for layer in masks1['lang1'])
    count1_lang2 = sum(len(layer) for layer in masks1['lang2'])
    count2_lang1 = sum(len(layer) for layer in masks2['lang1'])
    count2_lang2 = sum(len(layer) for layer in masks2['lang2'])
    
    total1 = count1_lang1 + count1_lang2
    total2 = count2_lang1 + count2_lang2
    
    report.append(f"\n{'Метрика':<40} {name1:<20} {name2:<20}")
    report.append("-" * 80)
    report.append(f"{lang1_name + '-специфичные нейроны':<40} {count1_lang1:<20,} {count2_lang1:<20,}")
    report.append(f"{lang2_name + '-специфичные нейроны':<40} {count1_lang2:<20,} {count2_lang2:<20,}")
    report.append(f"{'Всего языко-специфичных':<40} {total1:<20,} {total2:<20,}")
    
    # Percentage of total neurons
    pct1_lang1 = 100 * count1_lang1 / data1['neuron_totals']['total_neurons']
    pct2_lang1 = 100 * count2_lang1 / data2['neuron_totals']['total_neurons']
    pct1_lang2 = 100 * count1_lang2 / data1['neuron_totals']['total_neurons']
    pct2_lang2 = 100 * count2_lang2 / data2['neuron_totals']['total_neurons']
    
    report.append(f"\n{'% от всех нейронов:':<40}")
    report.append(f"{'  ' + lang1_name + '-специфичные':<40} {pct1_lang1:<20.3f}% {pct2_lang1:<20.3f}%")
    report.append(f"{'  ' + lang2_name + '-специфичные':<40} {pct1_lang2:<20.3f}% {pct2_lang2:<20.3f}%")
    report.append(f"{'  Всего специфичных':<40} {100*total1/data1['neuron_totals']['total_neurons']:<20.3f}% {100*total2/data2['neuron_totals']['total_neurons']:<20.3f}%")
    
    # Thresholds used
    report.append(f"\n{'ПОРОГИ ОТБОРА':<40}")
    report.append("-" * 80)
    t1 = data1['thresholds']
    t2 = data2['thresholds']
    report.append(f"{'Filter rate':<40} {t1['filter_rate']:<20} {t2['filter_rate']:<20}")
    report.append(f"{'Top rate (энтропия)':<40} {t1['top_rate']:<20} {t2['top_rate']:<20}")
    report.append(f"{'Activation bar ratio':<40} {t1['activation_bar_ratio']:<20} {t2['activation_bar_ratio']:<20}")
    report.append(f"{'Activation bar value':<40} {t1['activation_bar_value']:<20.4f} {t2['activation_bar_value']:<20.4f}")
    
    return "\n".join(report)


def analyze_layer_distribution(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> Tuple[str, np.ndarray, np.ndarray]:
    """Analyze distribution of language-specific neurons across layers."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("РАСПРЕДЕЛЕНИЕ ПО СЛОЯМ")
    report.append("=" * 80)
    
    masks1 = data1['language_specific_masks']
    masks2 = data2['language_specific_masks']
    
    num_layers1 = data1['num_layers']
    num_layers2 = data2['num_layers']
    
    # Count per layer for model 1
    layer_counts1_lang1 = [len(masks1['lang1'][i]) for i in range(num_layers1)]
    layer_counts1_lang2 = [len(masks1['lang2'][i]) for i in range(num_layers1)]
    
    # Count per layer for model 2
    layer_counts2_lang1 = [len(masks2['lang1'][i]) for i in range(num_layers2)]
    layer_counts2_lang2 = [len(masks2['lang2'][i]) for i in range(num_layers2)]
    
    # Summary statistics
    report.append(f"\n{name1}:")
    report.append(f"  {lang1_name} - Мин: {min(layer_counts1_lang1)}, Макс: {max(layer_counts1_lang1)}, "
                 f"Среднее: {np.mean(layer_counts1_lang1):.1f}, Медиана: {np.median(layer_counts1_lang1):.1f}")
    report.append(f"  {lang2_name} - Мин: {min(layer_counts1_lang2)}, Макс: {max(layer_counts1_lang2)}, "
                 f"Среднее: {np.mean(layer_counts1_lang2):.1f}, Медиана: {np.median(layer_counts1_lang2):.1f}")
    
    report.append(f"\n{name2}:")
    report.append(f"  {lang1_name} - Мин: {min(layer_counts2_lang1)}, Макс: {max(layer_counts2_lang1)}, "
                 f"Среднее: {np.mean(layer_counts2_lang1):.1f}, Медиана: {np.median(layer_counts2_lang1):.1f}")
    report.append(f"  {lang2_name} - Мин: {min(layer_counts2_lang2)}, Макс: {max(layer_counts2_lang2)}, "
                 f"Среднее: {np.mean(layer_counts2_lang2):.1f}, Медиана: {np.median(layer_counts2_lang2):.1f}")
    
    # Find layers with most differences
    report.append(f"\nСлои с наибольшими различиями ({lang1_name}-специфичные):")
    diff_lang1 = np.array(layer_counts1_lang1) - np.array(layer_counts2_lang1[:num_layers1] + [0]*(num_layers1 - min(num_layers1, num_layers2)))
    # Исправим diff корректно под минимальное число слоёв:
    min_layers = min(num_layers1, num_layers2)
    diff_lang1 = np.array(layer_counts1_lang1[:min_layers]) - np.array(layer_counts2_lang1[:min_layers])
    top_diff_idx = np.argsort(np.abs(diff_lang1))[-5:][::-1]
    for idx in top_diff_idx:
        report.append(f"  Слой {idx}: {name1}={layer_counts1_lang1[idx]}, {name2}={layer_counts2_lang1[idx]}, разница={diff_lang1[idx]}")
    
    # Prepare data for plotting
    data_array1 = np.array([layer_counts1_lang1, layer_counts1_lang2])
    data_array2 = np.array([layer_counts2_lang1, layer_counts2_lang2])
    
    return "\n".join(report), data_array1, data_array2


def calculate_overlap(data1: Dict, data2: Dict, name1: str, name2: str, lang1_name: str, lang2_name: str) -> Tuple[Dict, str]:
    """Calculate overlap of language-specific neurons between models."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("ПЕРЕСЕЧЕНИЕ ЯЗЫКО-СПЕЦИФИЧНЫХ НЕЙРОНОВ")
    report.append("=" * 80)
    
    masks1 = data1['language_specific_masks']
    masks2 = data2['language_specific_masks']
    
    num_layers = min(data1['num_layers'], data2['num_layers'])
    
    # Build sets of (layer, neuron) tuples
    set1_lang1 = set()
    set1_lang2 = set()
    set2_lang1 = set()
    set2_lang2 = set()
    
    for layer in range(num_layers):
        for neuron in masks1['lang1'][layer]:
            set1_lang1.add((layer, neuron))
        for neuron in masks1['lang2'][layer]:
            set1_lang2.add((layer, neuron))
        for neuron in masks2['lang1'][layer]:
            set2_lang1.add((layer, neuron))
        for neuron in masks2['lang2'][layer]:
            set2_lang2.add((layer, neuron))
    
    # Calculate overlaps
    overlap_lang1 = set1_lang1 & set2_lang1
    overlap_lang2 = set1_lang2 & set2_lang2
    
    only_model1_lang1 = set1_lang1 - set2_lang1
    only_model2_lang1 = set2_lang1 - set1_lang1
    only_model1_lang2 = set1_lang2 - set2_lang2
    only_model2_lang2 = set2_lang2 - set1_lang2
    
    # Jaccard similarity
    union_lang1 = set1_lang1 | set2_lang1
    union_lang2 = set1_lang2 | set2_lang2
    jaccard_lang1 = len(overlap_lang1) / len(union_lang1) if len(union_lang1) > 0 else 0.0
    jaccard_lang2 = len(overlap_lang2) / len(union_lang2) if len(union_lang2) > 0 else 0.0
    
    report.append(f"\n{lang1_name}-специфичные нейроны:")
    report.append(f"  {NAME1}: {len(set1_lang1):,} нейронов")
    report.append(f"  {NAME2}: {len(set2_lang1):,} нейронов")
    report.append(f"  Пересечение: {len(overlap_lang1):,} нейронов ({(100*len(overlap_lang1)/len(set1_lang1)) if len(set1_lang1) else 0:.2f}% от {NAME1})")
    report.append(f"  Только в {NAME1}: {len(only_model1_lang1):,} нейронов")
    report.append(f"  Только в {NAME2}: {len(only_model2_lang1):,} нейронов")
    report.append(f"  Jaccard similarity: {jaccard_lang1:.4f}")
    
    report.append(f"\n{lang2_name}-специфичные нейроны:")
    report.append(f"  {NAME1}: {len(set1_lang2):,} нейронов")
    report.append(f"  {NAME2}: {len(set2_lang2):,} нейронов")
    report.append(f"  Пересечение: {len(overlap_lang2):,} нейронов ({(100*len(overlap_lang2)/len(set1_lang2)) if len(set1_lang2) else 0:.2f}% от {NAME1})")
    report.append(f"  Только в {NAME1}: {len(only_model1_lang2):,} нейронов")
    report.append(f"  Только в {NAME2}: {len(only_model2_lang2):,} нейронов")
    report.append(f"  Jaccard similarity: {jaccard_lang2:.4f}")
    
    overlap_data = {
        'lang1': {
            'overlap': len(overlap_lang1),
            'only_model1': len(only_model1_lang1),
            'only_model2': len(only_model2_lang1),
            'jaccard': jaccard_lang1
        },
        'lang2': {
            'overlap': len(overlap_lang2),
            'only_model1': len(only_model1_lang2),
            'only_model2': len(only_model2_lang2),
            'jaccard': jaccard_lang2
        }
    }
    
    return overlap_data, "\n".join(report)


def create_visualizations(data1: Dict, data2: Dict, name1: str, name2: str, 
                          layer_dist1: np.ndarray, layer_dist2: np.ndarray,
                          overlap_data: Dict, output_prefix: str, lang1_name: str, lang2_name: str):
    """Create comprehensive visualizations."""
    # ------- Большое полотно: 3x3 -------
    fig = plt.figure(figsize=(20, 12))
    
    # 1. Layer distribution comparison - Model 1
    ax1 = plt.subplot(3, 3, 1)
    layers1 = np.arange(data1['num_layers'])
    width = 0.35
    ax1.bar(layers1 - width/2, layer_dist1[0], width, label=f'Lang1 ({lang1_name})', alpha=0.8, color='#2E86AB')
    ax1.bar(layers1 + width/2, layer_dist1[1], width, label=f'Lang2 ({lang2_name})', alpha=0.8, color='#A23B72')
    ax1.set_xlabel('Слой')
    ax1.set_ylabel('Количество нейронов')
    ax1.set_title(f'{name1}: Распределение по слоям')
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Layer distribution comparison - Model 2
    ax2 = plt.subplot(3, 3, 2)
    layers2 = np.arange(data2['num_layers'])
    ax2.bar(layers2 - width/2, layer_dist2[0], width, label=f'Lang1 ({lang1_name})', alpha=0.8, color='#2E86AB')
    ax2.bar(layers2 + width/2, layer_dist2[1], width, label=f'Lang2 ({lang2_name})', alpha=0.8, color='#A23B72')
    ax2.set_xlabel('Слой')
    ax2.set_ylabel('Количество нейронов')
    ax2.set_title(f'{name2}: Распределение по слоям')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Direct comparison of layer distributions
    ax3 = plt.subplot(3, 3, 3)
    x = np.arange(min(data1['num_layers'], data2['num_layers']))
    ax3.plot(x, layer_dist1[0][:len(x)], 'o-', label=f'{name1} - {lang1_name}', linewidth=2, markersize=4, color='#2E86AB')
    ax3.plot(x, layer_dist2[0][:len(x)], 's-', label=f'{name2} - {lang1_name}', linewidth=2, markersize=4, color='#58A4B0')
    ax3.plot(x, layer_dist1[1][:len(x)], 'o-', label=f'{name1} - {lang2_name}', linewidth=2, markersize=4, color='#A23B72')
    ax3.plot(x, layer_dist2[1][:len(x)], 's-', label=f'{name2} - {lang2_name}', linewidth=2, markersize=4, color='#F18F01')
    ax3.set_xlabel('Слой')
    ax3.set_ylabel('Количество нейронов')
    ax3.set_title('Сравнение моделей по слоям')
    ax3.legend(fontsize=8)
    ax3.grid(alpha=0.3)
    
    # 4. Total neurons comparison
    ax4 = plt.subplot(3, 3, 4)
    categories = [f'{lang1_name}-specific', f'{lang2_name}-specific', 'Total specific']
    model1_vals = [
        sum(len(l) for l in data1['language_specific_masks']['lang1']),
        sum(len(l) for l in data1['language_specific_masks']['lang2']),
        sum(len(l) for l in data1['language_specific_masks']['lang1']) + 
        sum(len(l) for l in data1['language_specific_masks']['lang2'])
    ]
    model2_vals = [
        sum(len(l) for l in data2['language_specific_masks']['lang1']),
        sum(len(l) for l in data2['language_specific_masks']['lang2']),
        sum(len(l) for l in data2['language_specific_masks']['lang1']) + 
        sum(len(l) for l in data2['language_specific_masks']['lang2'])
    ]
    x = np.arange(len(categories))
    ax4.bar(x - width/2, model1_vals, width, label=name1, alpha=0.8, color='#2E86AB')
    ax4.bar(x + width/2, model2_vals, width, label=name2, alpha=0.8, color='#F18F01')
    ax4.set_ylabel('Количество нейронов')
    ax4.set_title('Сравнение общего количества')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Overlap visualization - Lang1
    ax5 = plt.subplot(3, 3, 5)
    overlap_cats = ['Пересечение', f'Только\n{name1}', f'Только\n{name2}']
    overlap_vals_lang1 = [
        overlap_data['lang1']['overlap'],
        overlap_data['lang1']['only_model1'],
        overlap_data['lang1']['only_model2']
    ]
    colors = ['#06A77D', '#2E86AB', '#F18F01']
    ax5.bar(overlap_cats, overlap_vals_lang1, color=colors, alpha=0.8)
    ax5.set_ylabel('Количество нейронов')
    ax5.set_title(f'Пересечение: {lang1_name}-специфичные\n(Jaccard: {overlap_data["lang1"]["jaccard"]:.3f})')
    ax5.grid(axis='y', alpha=0.3)
    for i, v in enumerate(overlap_vals_lang1):
        ax5.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    # 6. Overlap visualization - Lang2
    ax6 = plt.subplot(3, 3, 6)
    overlap_vals_lang2 = [
        overlap_data['lang2']['overlap'],
        overlap_data['lang2']['only_model1'],
        overlap_data['lang2']['only_model2']
    ]
    ax6.bar(overlap_cats, overlap_vals_lang2, color=colors, alpha=0.8)
    ax6.set_ylabel('Количество нейронов')
    ax6.set_title(f'Пересечение: {lang2_name}-специфичные\n(Jaccard: {overlap_data["lang2"]["jaccard"]:.3f})')
    ax6.grid(axis='y', alpha=0.3)
    for i, v in enumerate(overlap_vals_lang2):
        ax6.text(i, v, f'{v:,}', ha='center', va='bottom', fontsize=9)
    
    # 7. Heatmap - Model 1 layer distribution
    ax7 = plt.subplot(3, 3, 7)
    heatmap_data1 = layer_dist1.T  # Transpose for better visualization
    im1 = ax7.imshow(heatmap_data1, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax7.set_xlabel('Язык')
    ax7.set_ylabel('Слой')
    ax7.set_title(f'{name1}: Тепловая карта')
    ax7.set_xticks([0, 1])
    ax7.set_xticklabels([lang1_name, lang2_name])
    plt.colorbar(im1, ax=ax7, label='Количество нейронов')
    
    # 8. Heatmap - Model 2 layer distribution
    ax8 = plt.subplot(3, 3, 8)
    heatmap_data2 = layer_dist2.T
    im2 = ax8.imshow(heatmap_data2, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax8.set_xlabel('Язык')
    ax8.set_ylabel('Слой')
    ax8.set_title(f'{name2}: Тепловая карта')
    ax8.set_xticks([0, 1])
    ax8.set_xticklabels([lang1_name, lang2_name])
    plt.colorbar(im2, ax=ax8, label='Количество нейронов')
    
    # 9. Distribution histogram
    ax9 = plt.subplot(3, 3, 9)
    all_counts1 = np.concatenate([layer_dist1[0], layer_dist1[1]])
    all_counts2 = np.concatenate([layer_dist2[0], layer_dist2[1]])
    bins = np.linspace(0, max(all_counts1.max(), all_counts2.max()), 30)
    ax9.hist(all_counts1, bins=bins, alpha=0.5, label=name1, color='#2E86AB', edgecolor='black')
    ax9.hist(all_counts2, bins=bins, alpha=0.5, label=name2, color='#F18F01', edgecolor='black')
    ax9.set_xlabel('Количество нейронов в слое')
    ax9.set_ylabel('Частота')
    ax9.set_title('Гистограмма распределения')
    ax9.legend()
    ax9.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_comparison.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_prefix}_comparison.png")
    plt.close()
    
    # ------- Детальное полотно 2x2 -------
    fig2, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Difference plots
    num_layers = min(data1['num_layers'], data2['num_layers'])
    diff_lang1 = layer_dist1[0][:num_layers] - layer_dist2[0][:num_layers]
    diff_lang2 = layer_dist1[1][:num_layers] - layer_dist2[1][:num_layers]
    
    # Lang1 difference
    axes[0, 0].bar(range(num_layers), diff_lang1, color=['#06A77D' if x >= 0 else '#D62828' for x in diff_lang1], alpha=0.7)
    axes[0, 0].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 0].set_xlabel('Слой')
    axes[0, 0].set_ylabel(f'Разница ({name1} - {name2})')
    axes[0, 0].set_title(f'Разница в {lang1_name}-специфичных нейронах по слоям')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Lang2 difference
    axes[0, 1].bar(range(num_layers), diff_lang2, color=['#06A77D' if x >= 0 else '#D62828' for x in diff_lang2], alpha=0.7)
    axes[0, 1].axhline(y=0, color='black', linestyle='-', linewidth=0.8)
    axes[0, 1].set_xlabel('Слой')
    axes[0, 1].set_ylabel(f'Разница ({name1} - {name2})')
    axes[0, 1].set_title(f'Разница в {lang2_name}-специфичных нейронах по слоям')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Cumulative distribution
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist1[0][:num_layers]), 'o-', label=f'{name1} - {lang1_name}', linewidth=2)
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist2[0][:num_layers]), 's-', label=f'{name2} - {lang1_name}', linewidth=2)
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist1[1][:num_layers]), 'o-', label=f'{name1} - {lang2_name}', linewidth=2)
    axes[1, 0].plot(range(num_layers), np.cumsum(layer_dist2[1][:num_layers]), 's-', label=f'{name2} - {lang2_name}', linewidth=2)
    axes[1, 0].set_xlabel('Слой')
    axes[1, 0].set_ylabel('Кумулятивное количество')
    axes[1, 0].set_title('Кумулятивное распределение по слоям')
    axes[1, 0].legend()
    axes[1, 0].grid(alpha=0.3)
    
    # Percentage by layer
    pct_lang1_model1 = 100 * layer_dist1[0][:num_layers] / data1['intermediate_size']
    pct_lang2_model1 = 100 * layer_dist1[1][:num_layers] / data1['intermediate_size']
    pct_lang1_model2 = 100 * layer_dist2[0][:num_layers] / data2['intermediate_size']
    pct_lang2_model2 = 100 * layer_dist2[1][:num_layers] / data2['intermediate_size']
    
    axes[1, 1].plot(range(num_layers), pct_lang1_model1, 'o-', label=f'{name1} - {lang1_name}', linewidth=2, markersize=4)
    axes[1, 1].plot(range(num_layers), pct_lang1_model2, 's-', label=f'{name2} - {lang1_name}', linewidth=2, markersize=4)
    axes[1, 1].plot(range(num_layers), pct_lang2_model1, 'o-', label=f'{name1} - {lang2_name}', linewidth=2, markersize=4)
    axes[1, 1].plot(range(num_layers), pct_lang2_model2, 's-', label=f'{name2} - {lang2_name}', linewidth=2, markersize=4)
    axes[1, 1].set_xlabel('Слой')
    axes[1, 1].set_ylabel('% от intermediate_size')
    axes[1, 1].set_title('Процент языко-специфичных нейронов по слоям')
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{output_prefix}_detailed.png', dpi=150, bbox_inches='tight')
    print(f"Сохранено: {output_prefix}_detailed.png")
    plt.close()


# ================= TOP-LANG1 (rate_diff) =================

def _extract_top_arrays(top_list: List[Dict[str, Any]]) -> Dict[str, np.ndarray]:
    """Преобразовать список dict'ов в numpy-массивы (без лимитов по длине)."""
    if not top_list:
        return {
            "layer": np.array([], dtype=int),
            "neuron": np.array([], dtype=int),
            "lang1_rate": np.array([], dtype=float),
            "lang2_rate": np.array([], dtype=float),
            "rate_diff": np.array([], dtype=float),
        }
    layer = np.array([int(x.get("layer", -1)) for x in top_list], dtype=int)
    neuron = np.array([int(x.get("neuron", -1)) for x in top_list], dtype=int)
    lang1_rate = np.array([float(x.get("lang1_rate", 0.0)) for x in top_list], dtype=float)
    lang2_rate = np.array([float(x.get("lang2_rate", 0.0)) for x in top_list], dtype=float)
    rate_diff = np.array([float(x.get("rate_diff", 0.0)) for x in top_list], dtype=float)
    return {
        "layer": layer,
        "neuron": neuron,
        "lang1_rate": lang1_rate,
        "lang2_rate": lang2_rate,
        "rate_diff": rate_diff,
    }


def _summarize_rate_diff(arr: np.ndarray) -> Dict[str, float]:
    if arr.size == 0:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0, "std": 0.0,
                "p25": 0.0, "p50": 0.0, "p75": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0}
    return {
        "count": int(arr.size),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "p25": float(np.percentile(arr, 25)),
        "p50": float(np.percentile(arr, 50)),
        "p75": float(np.percentile(arr, 75)),
        "p90": float(np.percentile(arr, 90)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def analyze_top_lang1_section(data1: Dict, data2: Dict, name1: str, name2: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
    """Собрать текстовый отчёт по top_lang1_specific_by_rate_diff и подготовить массивы для графиков."""
    report = []
    report.append("\n" + "=" * 80)
    report.append("TOP LANG1-SPECIFIC BY RATE_DIFF (БЕЗ ЛИМИТА)")
    report.append("=" * 80)

    top1 = data1.get("top_lang1_specific_by_rate_diff", [])
    top2 = data2.get("top_lang1_specific_by_rate_diff", [])

    A1 = _extract_top_arrays(top1)
    A2 = _extract_top_arrays(top2)

    s1 = _summarize_rate_diff(A1["rate_diff"])
    s2 = _summarize_rate_diff(A2["rate_diff"])

    report.append(f"\n{name1}: count={s1['count']:,}, min={s1['min']:.6f}, p50={s1['p50']:.6f}, "
                  f"mean={s1['mean']:.6f}, max={s1['max']:.6f}")
    report.append(f"{name2}: count={s2['count']:,}, min={s2['min']:.6f}, p50={s2['p50']:.6f}, "
                  f"mean={s2['mean']:.6f}, max={s2['max']:.6f}")

    # По слоям — топ слоям по количеству попаданий
    if A1["layer"].size > 0:
        unique_l1, counts_l1 = np.unique(A1["layer"], return_counts=True)
        idx1 = np.argsort(counts_l1)[::-1]
        top_layers_1 = list(zip(unique_l1[idx1][:10].tolist(), counts_l1[idx1][:10].tolist()))
        report.append(f"\n{name1}: топ-10 слоёв по числу нейронов в списке:")
        for L, C in top_layers_1:
            report.append(f"  слой {L}: {C}")
    if A2["layer"].size > 0:
        unique_l2, counts_l2 = np.unique(A2["layer"], return_counts=True)
        idx2 = np.argsort(counts_l2)[::-1]
        top_layers_2 = list(zip(unique_l2[idx2][:10].tolist(), counts_l2[idx2][:10].tolist()))
        report.append(f"\n{name2}: топ-10 слоёв по числу нейронов в списке:")
        for L, C in top_layers_2:
            report.append(f"  слой {L}: {C}")

    # Небольшая стабильность: пересечение по (layer, neuron) внутри top-списков обеих моделей
    set_m1 = set(zip(A1["layer"].tolist(), A1["neuron"].tolist()))
    set_m2 = set(zip(A2["layer"].tolist(), A2["neuron"].tolist()))
    inter = set_m1 & set_m2
    union = set_m1 | set_m2
    jaccard = (len(inter) / len(union)) if len(union) else 0.0
    report.append(f"\nПересечение топ-списков (layer,neuron): {len(inter):,} | объединение: {len(union):,} | Jaccard={jaccard:.6f}")

    return "\n".join(report), A1, A2


def _maybe_subsample(x: np.ndarray, y: np.ndarray, max_points: int = 60000) -> Tuple[np.ndarray, np.ndarray]:
    """Ограничить число точек для scatter/hexbin."""
    n = x.size
    if n <= max_points:
        return x, y
    idx = np.linspace(0, n - 1, max_points, dtype=int)
    return x[idx], y[idx]


def create_top_neurons_visualizations(A1: Dict[str, np.ndarray], A2: Dict[str, np.ndarray],
                                      data1: Dict, data2: Dict, name1: str, name2: str,
                                      output_prefix: str):
    """Графики по top_lang1_specific_by_rate_diff (без лимита)."""
    # Если данных нет — просто завершить
    if A1["rate_diff"].size == 0 and A2["rate_diff"].size == 0:
        print("Нет данных для top_lang1_specific_by_rate_diff — пропускаю визуализацию.")
        return

    fig = plt.figure(figsize=(20, 13))
    gs = fig.add_gridspec(3, 3)

    # 1) Гистограммы rate_diff
    ax1 = fig.add_subplot(gs[0, 0])
    bins = 60
    if A1["rate_diff"].size:
        ax1.hist(A1["rate_diff"], bins=bins, alpha=0.55, label=f"{name1}", color='#2E86AB', edgecolor='black')
    if A2["rate_diff"].size:
        ax1.hist(A2["rate_diff"], bins=bins, alpha=0.55, label=f"{name2}", color='#F18F01', edgecolor='black')
    ax1.set_title("Распределение rate_diff (Lang1-specific)")
    ax1.set_xlabel("rate_diff = p_lang1 - p_lang2")
    ax1.set_ylabel("Частота")
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # 2) ECDF
    ax2 = fig.add_subplot(gs[0, 1])
    if A1["rate_diff"].size:
        sorted1 = np.sort(A1["rate_diff"])
        ax2.plot(sorted1, np.linspace(0, 1, sorted1.size), label=name1, linewidth=2)
    if A2["rate_diff"].size:
        sorted2 = np.sort(A2["rate_diff"])
        ax2.plot(sorted2, np.linspace(0, 1, sorted2.size), label=name2, linewidth=2)
    ax2.set_title("ECDF rate_diff")
    ax2.set_xlabel("rate_diff")
    ax2.set_ylabel("F(x)")
    ax2.legend()
    ax2.grid(alpha=0.3)

    # 3) Scatter layer vs rate_diff (оба)
    ax3 = fig.add_subplot(gs[0, 2])
    if A1["layer"].size:
        x1, y1 = _maybe_subsample(A1["layer"], A1["rate_diff"])
        ax3.scatter(x1, y1, s=6, alpha=0.35, label=name1, color='#2E86AB')
    if A2["layer"].size:
        x2, y2 = _maybe_subsample(A2["layer"], A2["rate_diff"])
        ax3.scatter(x2, y2, s=6, alpha=0.35, label=name2, color='#F18F01')
    ax3.set_title("Слой vs rate_diff (scatter)")
    ax3.set_xlabel("Слой")
    ax3.set_ylabel("rate_diff")
    ax3.legend(markerscale=2)
    ax3.grid(alpha=0.3)

    # 4) Hexbin для Model 1
    ax4 = fig.add_subplot(gs[1, 0])
    if A1["layer"].size:
        ax4.hexbin(A1["layer"], A1["rate_diff"], gridsize=50, cmap='viridis')
        ax4.set_title(f"{name1}: Hexbin(layer, rate_diff)")
        ax4.set_xlabel("Слой")
        ax4.set_ylabel("rate_diff")
        cb = plt.colorbar(mappable=ax4.collections[0], ax=ax4)
        cb.set_label("counts")
    else:
        ax4.text(0.5, 0.5, "Нет данных", ha='center', va='center')

    # 5) Hexbin для Model 2
    ax5 = fig.add_subplot(gs[1, 1])
    if A2["layer"].size:
        ax5.hexbin(A2["layer"], A2["rate_diff"], gridsize=50, cmap='viridis')
        ax5.set_title(f"{name2}: Hexbin(layer, rate_diff)")
        ax5.set_xlabel("Слой")
        ax5.set_ylabel("rate_diff")
        cb = plt.colorbar(mappable=ax5.collections[0], ax=ax5)
        cb.set_label("counts")
    else:
        ax5.text(0.5, 0.5, "Нет данных", ha='center', va='center')

    # 6) lang1_rate vs lang2_rate (оба)
    ax6 = fig.add_subplot(gs[1, 2])
    if A1["lang1_rate"].size:
        x1, y1 = _maybe_subsample(A1["lang1_rate"], A1["lang2_rate"])
        ax6.scatter(x1, y1, s=6, alpha=0.25, label=f"{name1}", color='#2E86AB')
    if A2["lang1_rate"].size:
        x2, y2 = _maybe_subsample(A2["lang1_rate"], A2["lang2_rate"])
        ax6.scatter(x2, y2, s=6, alpha=0.25, label=f"{name2}", color='#F18F01')
    lim = ax6.get_xlim()
    ax6.plot([0, max(1e-9, lim[1])], [0, max(1e-9, lim[1])], 'k--', linewidth=1)
    ax6.set_title("lang1_rate vs lang2_rate")
    ax6.set_xlabel("lang1_rate")
    ax6.set_ylabel("lang2_rate")
    ax6.legend()
    ax6.grid(alpha=0.3)

    # 7–8) Бар-чарты по слоям (counts в топ-списке)
    ax7 = fig.add_subplot(gs[2, 0])
    ax8 = fig.add_subplot(gs[2, 1])

    if A1["layer"].size:
        L1, C1 = np.unique(A1["layer"], return_counts=True)
        ord1 = np.argsort(C1)[::-1][:20]
        ax7.bar([str(l) for l in L1[ord1]], C1[ord1], color='#2E86AB', alpha=0.8)
        ax7.set_title(f"{name1}: ТОП-20 слоёв (по количеству)")
        ax7.set_xlabel("Слой")
        ax7.set_ylabel("Число нейронов в списке")
        ax7.tick_params(axis='x', rotation=45)
        ax7.grid(axis='y', alpha=0.3)
    else:
        ax7.text(0.5, 0.5, "Нет данных", ha='center', va='center')

    if A2["layer"].size:
        L2, C2 = np.unique(A2["layer"], return_counts=True)
        ord2 = np.argsort(C2)[::-1][:20]
        ax8.bar([str(l) for l in L2[ord2]], C2[ord2], color='#F18F01', alpha=0.8)
        ax8.set_title(f"{name2}: ТОП-20 слоёв (по количеству)")
        ax8.set_xlabel("Слой")
        ax8.set_ylabel("Число нейронов в списке")
        ax8.tick_params(axis='x', rotation=45)
        ax8.grid(axis='y', alpha=0.3)
    else:
        ax8.text(0.5, 0.5, "Нет данных", ha='center', va='center')

    # 9) Boxplot rate_diff по моделям
    ax9 = fig.add_subplot(gs[2, 2])
    data_box = []
    labels_box = []
    if A1["rate_diff"].size:
        data_box.append(A1["rate_diff"])
        labels_box.append(name1)
    if A2["rate_diff"].size:
        data_box.append(A2["rate_diff"])
        labels_box.append(name2)
    if data_box:
        ax9.boxplot(data_box, labels=labels_box, showfliers=False)
        ax9.set_title("Boxplot rate_diff")
        ax9.set_ylabel("rate_diff")
        ax9.grid(axis='y', alpha=0.3)
    else:
        ax9.text(0.5, 0.5, "Нет данных", ha='center', va='center')

    plt.tight_layout()
    out_path = f"{output_prefix}_top_neurons.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"Сохранено: {out_path}")
    plt.close()


def main():
    print("=" * 80)
    print("СРАВНЕНИЕ ЯЗЫКО-СПЕЦИФИЧНЫХ НЕЙРОНОВ")
    print("=" * 80)
    
    # Load data
    data1 = load_json(JSON1_PATH)
    data2 = load_json(JSON2_PATH)
    
    # Extract language names from JSON files
    lang1_name = data1.get('columns', {}).get('lang1', 'lang1')
    lang2_name = data1.get('columns', {}).get('lang2', 'lang2')
    
    print(f"Языки: Lang1 = {lang1_name}, Lang2 = {lang2_name}")
    
    # Generate reports
    report_parts = []
    
    # Basic statistics
    report_parts.append(compare_basic_stats(data1, data2, NAME1, NAME2, lang1_name, lang2_name))
    
    # Language-specific neurons
    report_parts.append(compare_language_specific_neurons(data1, data2, NAME1, NAME2, lang1_name, lang2_name))
    
    # Layer distribution
    layer_report, layer_dist1, layer_dist2 = analyze_layer_distribution(data1, data2, NAME1, NAME2, lang1_name, lang2_name)
    report_parts.append(layer_report)
    
    # Overlap analysis
    overlap_data, overlap_report = calculate_overlap(data1, data2, NAME1, NAME2, lang1_name, lang2_name)
    report_parts.append(overlap_report)

    # NEW: top_lang1_specific_by_rate_diff — без лимита
    top_report, A1, A2 = analyze_top_lang1_section(data1, data2, NAME1, NAME2)
    report_parts.append(top_report)
    
    # Combine all reports
    full_report = "\n".join(report_parts)
    
    # Print to console
    print(full_report)
    
    # Save to file
    report_file = f"{OUTPUT_PREFIX}_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(full_report)
    print(f"\n{'-'*80}")
    print(f"Отчёт сохранён: {report_file}")
    
    # Generate visualizations (основные)
    print(f"\nГенерация визуализаций...")
    create_visualizations(data1, data2, NAME1, NAME2, 
                          layer_dist1, layer_dist2, overlap_data, OUTPUT_PREFIX, lang1_name, lang2_name)

    # NEW: Отдельное изображение по top_lang1_specific_by_rate_diff
    print(f"Генерация визуализаций для top_lang1_specific_by_rate_diff...")
    create_top_neurons_visualizations(A1, A2, data1, data2, NAME1, NAME2, OUTPUT_PREFIX)
    
    print(f"\n{'='*80}")
    print("ГОТОВО!")
    print(f"{'='*80}")
    print("Файлы созданы:")
    print(f"  - {report_file}")
    print(f"  - {OUTPUT_PREFIX}_comparison.png")
    print(f"  - {OUTPUT_PREFIX}_detailed.png")
    print(f"  - {OUTPUT_PREFIX}_top_neurons.png")


if __name__ == "__main__":
    main()
