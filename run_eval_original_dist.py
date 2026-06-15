# -*- coding: utf-8 -*-
"""v5 模型在原始类别分布测试集上的泛化评估，并与均衡测试集结果对比"""
import json
import os

import numpy as np
import pandas as pd
import pickle as pkl
import torch
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from calibration import load_calibration
from main import (
    Model,
    TextDataset,
    UNK,
    PAD,
    dev_eval,
    label_names,
    num_classes,
    pad_size,
    plot_confusion_matrix,
    save_path,
    vocab_path,
)
from preprocess import clean_weibo_text

RAW_DATA_PATH = './data/weibo_3class.csv'
BALANCED_EVAL_PATH = './results/eval_latest.json'
OUT_JSON = './results/eval_original_dist_v5.json'
OUT_MD = './results/eval_v5_generalization_comparison.md'


def load_original_dataset(path, pad_size, tokenizer, vocab):
    """原始三分类 CSV，推理模式清洗，6:2:2 分层划分（与主流程 random_state 一致）"""
    contents = []
    labels = []
    df = pd.read_csv(path, header=None, names=['label', 'text'])
    df = df.dropna()
    df = df[df['label'].isin([0, 1, 2])]

    for _, row in df.iterrows():
        label = int(row['label'])
        content = clean_weibo_text(str(row['text']).strip(), label=None)
        if len(content) < 2:
            continue
        token = tokenizer(content)
        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
        unk_idx = vocab.get(UNK, 1)
        words_line = [vocab.get(w, unk_idx) for w in token]
        contents.append((words_line, label))
        labels.append(label)

    train, X_t, y_train, y_t = train_test_split(
        contents, labels, test_size=0.4, random_state=42, stratify=labels
    )
    dev, test, _, _ = train_test_split(
        X_t, y_t, test_size=0.5, random_state=42, stratify=y_t
    )

    for name, subset in [('训练集', train), ('验证集', dev), ('测试集', test)]:
        counts = np.bincount([item[1] for item in subset], minlength=num_classes)
        total = len(subset)
        print(
            f"{name}（原始分布）-> 积极:{counts[0]}({counts[0]/total:.1%}) "
            f"中性:{counts[1]}({counts[1]/total:.1%}) "
            f"消极:{counts[2]}({counts[2]/total:.1%}) | 共{total}条"
        )
    return train, dev, test


def pct(v):
    return f'{v * 100:.2f}%'


def f4(v):
    return f'{v:.4f}'


def build_comparison_table(balanced, original):
    """生成 Markdown 对比表"""
    b_report = balanced['classification_report']
    o_report = original['classification_report']
    b_cm = balanced['confusion_matrix']
    o_cm = original['confusion_matrix']

    lines = [
        '# v5 模型：均衡测试集 vs 原始分布测试集',
        '',
        '> 模型与校准参数均为 v5 训练产物，未在原始分布上重新训练或重调校准。',
        '> 原始分布数据：`weibo_3class.csv`，推理模式清洗，`random_state=42` 分层划分测试集。',
        '',
        '## 1. 测试集构成',
        '',
        '| 项目 | 均衡测试集（v5 常规评估） | 原始分布测试集（泛化评估） |',
        '|------|--------------------------|---------------------------|',
        f"| 样本总数 | {int(b_report['积极']['support'] + b_report['中性']['support'] + b_report['消极']['support'])} | "
        f"{int(o_report['积极']['support'] + o_report['中性']['support'] + o_report['消极']['support'])} |",
        f"| 积极占比 | 33.3%（1:1:1） | {o_report['积极']['support'] / o_report['macro avg']['support']:.1%} |",
        f"| 中性占比 | 33.3%（1:1:1） | {o_report['中性']['support'] / o_report['macro avg']['support']:.1%} |",
        f"| 消极占比 | 33.3%（1:1:1） | {o_report['消极']['support'] / o_report['macro avg']['support']:.1%} |",
        '',
        '## 2. 整体指标对比',
        '',
        '| 指标 | 均衡测试集 | 原始分布测试集 | 差值（原始−均衡） |',
        '|------|------------|----------------|-------------------|',
    ]

    metrics = [
        ('准确率 (Accuracy)', 'accuracy', 'accuracy', True),
        ('macro-F1', 'macro_f1', 'macro_f1', False),
        ('加权 F1 (weighted)', None, None, False),
    ]
    for name, b_key, o_key, is_acc in metrics[:2]:
        bv = balanced[b_key]
        ov = original[o_key]
        diff = ov - bv
        if is_acc:
            lines.append(f'| {name} | {pct(bv)} | {pct(ov)} | {diff * 100:+.2f} pp |')
        else:
            lines.append(f'| {name} | {f4(bv)} | {f4(ov)} | {diff:+.4f} |')

    bw = b_report['weighted avg']['f1-score']
    ow = o_report['weighted avg']['f1-score']
    lines.append(f'| 加权 F1 (weighted) | {f4(bw)} | {f4(ow)} | {ow - bw:+.4f} |')

    lines.extend([
        '',
        '## 3. 各类别指标对比',
        '',
        '| 类别 | 指标 | 均衡测试集 | 原始分布测试集 | 差值 |',
        '|------|------|------------|----------------|------|',
    ])
    for cls in label_names:
        for metric, key in [('精确率', 'precision'), ('召回率', 'recall'), ('F1', 'f1-score')]:
            bv = b_report[cls][key]
            ov = o_report[cls][key]
            diff = ov - bv
            if key == 'f1-score':
                lines.append(f'| {cls} | {metric} | {f4(bv)} | {f4(ov)} | {diff:+.4f} |')
            else:
                lines.append(f'| {cls} | {metric} | {pct(bv)} | {pct(ov)} | {diff * 100:+.2f} pp |')

    lines.extend([
        '',
        '## 4. 混淆矩阵（行=真实，列=预测）',
        '',
        '### 均衡测试集',
        '',
        '```',
        '         积极   中性   消极',
    ])
    for i, name in enumerate(label_names):
        lines.append(f'  {name}  {b_cm[i]}')
    lines.extend([
        '```',
        '',
        '### 原始分布测试集',
        '',
        '```',
        '         积极   中性   消极',
    ])
    for i, name in enumerate(label_names):
        lines.append(f'  {name}  {o_cm[i]}')
    lines.extend([
        '```',
        '',
        '## 5. 简要解读',
        '',
    ])

    neu_recall_drop = o_report['中性']['recall'] - b_report['中性']['recall']
    acc_diff = original['accuracy'] - balanced['accuracy']
    if neu_recall_drop < -0.05:
        lines.append(
            f'- **中性召回**在原始分布上下降 **{abs(neu_recall_drop)*100:.1f} pp**（样本占比仅约 '
            f'{o_report["中性"]["support"] / o_report["macro avg"]["support"]:.1%}），是泛化主要风险点。'
        )
    else:
        lines.append('- 中性召回在原始分布上保持相对稳定。')
    if acc_diff > 0:
        lines.append(f'- 整体准确率在原始分布上 **更高**（{acc_diff*100:+.2f} pp），因多数类（积极/消极）占比较大。')
    elif acc_diff < 0:
        lines.append(f'- 整体准确率在原始分布上 **更低**（{acc_diff*100:+.2f} pp）。')
    else:
        lines.append('- 整体准确率两种测试集接近。')
    lines.append(
        '- **macro-F1** 对三类一视同仁，比 accuracy 更能反映原始分布下的真实泛化能力。'
    )

    return '\n'.join(lines) + '\n'


def main():
    os.makedirs('results', exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab = pkl.load(open(vocab_path, 'rb'))
    tokenizer = lambda x: [y for y in x]

    print('=== 加载 v5 模型与校准参数 ===')
    train_data, dev_data, test_data = load_original_dataset(
        RAW_DATA_PATH, pad_size, tokenizer, vocab
    )
    inv_vocab = {v: k for k, v in vocab.items()}
    cal_params = load_calibration()

    train_labels = [item[1] for item in train_data]
    raw_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    weight_values = np.sqrt(raw_weights) / np.sqrt(raw_weights).mean()
    weight_values[0] *= 1.10
    weight_values = weight_values / weight_values.mean()
    class_weights = torch.tensor(weight_values, dtype=torch.float32).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    model = Model(vocab).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))

    test_loader = DataLoader(TextDataset(test_data), 128)
    print('\n=== 原始分布测试集评估（v5 模型 + v5 校准）===')
    test_acc, test_loss, test_f1 = dev_eval(
        model, test_loader, loss_function,
        Result_test=True,
        cal=cal_params,
        report_path=OUT_JSON,
        inv_vocab=inv_vocab,
    )
    print(f'测试集 loss:{test_loss:.3f} | acc:{test_acc:.2%} | macro-F1:{test_f1:.4f}')

    with open(BALANCED_EVAL_PATH, 'r', encoding='utf-8') as f:
        balanced = json.load(f)
    with open(OUT_JSON, 'r', encoding='utf-8') as f:
        original = json.load(f)

    original['eval_type'] = 'original_distribution'
    original['data_source'] = RAW_DATA_PATH
    original['note'] = 'v5 权重与 calibration.pkl，未在原始分布上重训或重校准'
    with open(OUT_JSON, 'w', encoding='utf-8') as f:
        json.dump(original, f, ensure_ascii=False, indent=2)

    md = build_comparison_table(balanced, original)
    with open(OUT_MD, 'w', encoding='utf-8') as f:
        f.write(md)

    # 保存混淆矩阵图：原始分布 + 恢复均衡测试集
    os.makedirs('./results/milestones', exist_ok=True)
    plot_confusion_matrix(
        np.array(original['confusion_matrix']),
        save_path='./results/milestones/confusion_matrix_original_dist_v5.png',
    )
    plot_confusion_matrix(np.array(balanced['confusion_matrix']))

    print(f'\n对比表已写入: {OUT_MD}')
    print(f'原始分布评估 JSON: {OUT_JSON}')


if __name__ == '__main__':
    main()
