# -*- coding: utf-8 -*-
"""从 optimization_log 记录的各轮指标，生成汇报用趋势图与混淆矩阵图"""
import json
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# 中文字体（Windows 常见）
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

OUT_DIR = './results/milestones'
LABELS = ['积极', '中性', '消极']

# 第 0 轮准确率摘自《中文文本情感分析进展报告.pdf》§4.2（验证集约 74%）
# 第 1～5 轮摘自 optimization_log.md 测试集结果
MILESTONES = [
    {
        'round': 0,
        'name': 'v0_进展报告基线',
        'accuracy': 0.74,
        'macro_f1': None,
        'pos_f1': None,
        'neu_f1': None,
        'neg_f1': None,
        'test_set': '验证集（进展报告）',
        'acc_source': '中文文本情感分析进展报告.pdf §4.2',
        'confusion_matrix': None,
    },
    {
        'round': 1,
        'name': 'v1_全量清洗',
        'accuracy': 0.5815,
        'macro_f1': 0.5058,
        'pos_f1': 0.6758,
        'neu_f1': 0.2146,
        'neg_f1': 0.6269,
        'test_set': '原始分布',
        'confusion_matrix': [
            [6577, 2238, 1841],
            [483, 767, 475],
            [1748, 2419, 5447],
        ],
    },
    {
        'round': 2,
        'name': 'v2_均衡补强',
        'accuracy': 0.7944,
        'macro_f1': 0.7948,
        'pos_f1': 0.8243,
        'neu_f1': 0.7650,
        'neg_f1': 0.7951,
        'test_set': '均衡 1:1:1',
        'confusion_matrix': [
            [2440, 368, 121],
            [317, 2363, 249],
            [234, 518, 2177],
        ],
    },
    {
        'round': 3,
        'name': 'v3_清晰中性校准',
        'accuracy': 0.8002,
        'macro_f1': 0.8011,
        'pos_f1': 0.8343,
        'neu_f1': 0.7854,
        'neg_f1': 0.7836,
        'test_set': '均衡 1:1:1',
        'confusion_matrix': [
            [2468, 326, 343],
            [162, 2542, 434],
            [149, 467, 2522],
        ],
    },
    {
        'round': 4,
        'name': 'v4_词典加权',
        'accuracy': 0.8057,
        'macro_f1': 0.8067,
        'pos_f1': 0.8275,
        'neu_f1': 0.7996,
        'neg_f1': 0.7930,
        'test_set': '均衡 1:1:1',
        'confusion_matrix': [
            [2368, 307, 462],
            [119, 2529, 490],
            [99, 352, 2687],
        ],
    },
    {
        'round': 5,
        'name': 'v5_错例定向',
        'accuracy': 0.8586,
        'macro_f1': 0.8596,
        'pos_f1': 0.8836,
        'neu_f1': 0.8348,
        'neg_f1': 0.8605,
        'test_set': '均衡 1:1:1',
        'confusion_matrix': [
            [2734, 348, 58],
            [186, 2895, 60],
            [128, 552, 2461],
        ],
    },
]


def plot_confusion_matrix(cm, title, save_path, normalized=False):
    """绘制单张混淆矩阵热力图"""
    cm = np.array(cm, dtype=float)
    if normalized:
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        display = cm / row_sum
        fmt = '.1%'
    else:
        display = cm
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(display, interpolation='nearest', cmap=plt.cm.Blues)
    ax.set_title(title, fontsize=12)
    plt.colorbar(im, ax=ax, fraction=0.046)
    tick_marks = np.arange(len(LABELS))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(LABELS)
    ax.set_yticklabels(LABELS)
    ax.set_ylabel('真实标签')
    ax.set_xlabel('预测标签')

    thresh = display.max() / 2.0 if display.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = display[i, j]
            if normalized:
                text = f'{val:.1%}'
            else:
                text = f'{int(cm[i, j])}'
            color = 'white' if val > thresh else 'black'
            ax.text(j, i, text, ha='center', va='center', color=color, fontsize=10)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_accuracy_trend(milestones, save_path):
    """准确率与 macro-F1 折线图（第 0 轮为进展报告验证集准确率）"""
    rounds = [m['round'] for m in milestones]
    acc = [m['accuracy'] * 100 for m in milestones]
    f1_rounds = [m['round'] for m in milestones if m.get('macro_f1') is not None]
    f1_vals = [m['macro_f1'] * 100 for m in milestones if m.get('macro_f1') is not None]

    fig, ax1 = plt.subplots(figsize=(10, 5.5))
    ax1.plot(rounds, acc, 'o-', color='#2563eb', linewidth=2.5, markersize=8, label='准确率')
    if f1_rounds:
        ax1.plot(f1_rounds, f1_vals, 's--', color='#dc2626', linewidth=2, markersize=7, label='macro-F1（测试集）')
    ax1.set_xlabel('优化轮次')
    ax1.set_ylabel('百分比 (%)')
    ax1.set_title('各轮优化：准确率与 macro-F1 变化')
    ax1.set_xticks(rounds)
    ax1.set_xticklabels([f"第{r}轮\n{m['name'].split('_')[0]}" for r, m in zip(rounds, milestones)], fontsize=9)
    ax1.set_ylim(45, 92)
    ax1.grid(True, alpha=0.3)

    for m, a in zip(milestones, acc):
        suffix = '（验证集）' if m['round'] == 0 else ''
        ax1.annotate(
            f'{a:.1f}%{suffix}', (m['round'], a),
            textcoords='offset points', xytext=(0, 10), ha='center', fontsize=8,
        )

    ax1.annotate(
        '进展报告基线\n验证集≈74%',
        xy=(0, acc[0]), xytext=(-0.15, 82),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=8, color='#6b7280',
    )
    ax1.annotate(
        '全删表情→测试集下降',
        xy=(1, acc[1]), xytext=(1.3, 52),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='#6b7280',
    )
    ax1.annotate(
        '均衡数据+v5 错例补全',
        xy=(5, acc[5]), xytext=(3.2, 88),
        arrowprops=dict(arrowstyle='->', color='gray'),
        fontsize=9, color='#6b7280',
    )

    ax1.legend(loc='lower right')
    fig.text(0.5, 0.01, '说明：第0轮为进展报告验证集准确率；第1～5轮为各轮测试集准确率', ha='center', fontsize=9, color='#6b7280')
    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_class_f1_trend(milestones, save_path):
    """三类 F1 分组折线（第 1～5 轮，进展报告未记录分项 F1）"""
    items = [m for m in milestones if m.get('pos_f1') is not None]
    rounds = [m['round'] for m in items]
    fig, ax = plt.subplots(figsize=(10, 5.5))
    colors = {'pos_f1': '#22c55e', 'neu_f1': '#eab308', 'neg_f1': '#ef4444'}
    names = {'pos_f1': '积极 F1', 'neu_f1': '中性 F1', 'neg_f1': '消极 F1'}

    for key in ('pos_f1', 'neu_f1', 'neg_f1'):
        vals = [m[key] * 100 for m in items]
        ax.plot(rounds, vals, 'o-', label=names[key], color=colors[key], linewidth=2, markersize=6)

    ax.set_xlabel('优化轮次')
    ax.set_ylabel('F1 (%)')
    ax.set_title('各类别 F1 随优化轮次变化（测试集，第1轮起）')
    ax.set_xticks(rounds)
    ax.set_ylim(0, 95)
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_confusion_grid(milestones, save_path, normalized=True, rounds_filter=None):
    """多轮混淆矩阵拼图（仅均衡测试集轮次）"""
    items = [m for m in milestones if m['confusion_matrix'] is not None]
    if rounds_filter:
        items = [m for m in items if m['round'] in rounds_filter]

    n = len(items)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.2, rows * 3.8))
    if rows == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for idx, m in enumerate(items):
        cm = np.array(m['confusion_matrix'], dtype=float)
        if normalized:
            row_sum = cm.sum(axis=1, keepdims=True)
            row_sum[row_sum == 0] = 1
            display = cm / row_sum
        else:
            display = cm

        ax = axes[idx]
        im = ax.imshow(display, interpolation='nearest', cmap=plt.cm.Blues)
        ax.set_title(f"第{m['round']}轮 {m['name'].split('_', 1)[-1]}\nacc={m['accuracy']:.1%}", fontsize=10)
        ax.set_xticks(range(3))
        ax.set_yticks(range(3))
        ax.set_xticklabels(LABELS, fontsize=8)
        ax.set_yticklabels(LABELS, fontsize=8)
        for i in range(3):
            for j in range(3):
                val = display[i, j]
                txt = f'{val:.0%}' if normalized else f'{int(cm[i, j])}'
                ax.text(j, i, txt, ha='center', va='center', fontsize=8,
                        color='white' if val > display.max() / 2 else 'black')

    for j in range(len(items), len(axes)):
        axes[j].axis('off')

    suffix = '归一化' if normalized else '计数'
    fig.suptitle(f'混淆矩阵对比（{suffix}，行=真实）', fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_error_reduction(milestones, save_path):
    """主要误判类型（非对角线）随轮次变化"""
    balanced = [m for m in milestones if m['test_set'] == '均衡 1:1:1' and m['confusion_matrix']]
    rounds = [m['round'] for m in balanced]
    # 积极→中性+消极、中性→积极+消极、消极→积极+中性
    pos_err = []
    neu_err = []
    neg_err = []
    for m in balanced:
        cm = m['confusion_matrix']
        pos_err.append(cm[0][1] + cm[0][2])
        neu_err.append(cm[1][0] + cm[1][2])
        neg_err.append(cm[2][0] + cm[2][1])

    x = np.arange(len(rounds))
    width = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(x - width, pos_err, width, label='积极类误判数', color='#22c55e')
    ax.bar(x, neu_err, width, label='中性类误判数', color='#eab308')
    ax.bar(x + width, neg_err, width, label='消极类误判数', color='#ef4444')
    ax.set_xlabel('优化轮次（均衡测试集）')
    ax.set_ylabel('误判样本数')
    ax.set_title('各类别误判数量变化（混淆矩阵非对角线之和）')
    ax.set_xticks(x)
    ax.set_xticklabels([f"第{r}轮" for r in rounds])
    ax.legend()
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 保存结构化指标，便于 PPT / 视频引用
    summary = []
    for m in MILESTONES:
        summary.append({k: v for k, v in m.items()})
    with open(os.path.join(OUT_DIR, 'metrics_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    plot_accuracy_trend(MILESTONES, os.path.join(OUT_DIR, 'accuracy_macro_f1_trend.png'))
    plot_class_f1_trend(MILESTONES, os.path.join(OUT_DIR, 'class_f1_trend.png'))
    plot_error_reduction(MILESTONES, os.path.join(OUT_DIR, 'misclassification_by_round.png'))
    plot_confusion_grid(
        MILESTONES,
        os.path.join(OUT_DIR, 'confusion_matrix_grid_balanced_norm.png'),
        normalized=True,
        rounds_filter={2, 3, 4, 5},
    )
    plot_confusion_grid(
        MILESTONES,
        os.path.join(OUT_DIR, 'confusion_matrix_grid_all_norm.png'),
        normalized=True,
        rounds_filter=None,
    )

    for m in MILESTONES:
        if m['confusion_matrix'] is None:
            continue
        # 文件名用英文，避免 Windows 路径编码问题
        tag = f"round{m['round']}_{m['name'].split('_', 1)[0]}"
        title = m['name'].replace('_', ' ')
        base = os.path.join(OUT_DIR, tag)
        plot_confusion_matrix(m['confusion_matrix'], f"{title} 混淆矩阵", f'{base}_cm.png', normalized=False)
        plot_confusion_matrix(
            m['confusion_matrix'], f"{title} 混淆矩阵（按行归一化）",
            f'{base}_cm_norm.png', normalized=True,
        )

    print(f'已生成汇报图表至: {OUT_DIR}')
    for fn in sorted(os.listdir(OUT_DIR)):
        print(f'  - {fn}')


if __name__ == '__main__':
    main()
