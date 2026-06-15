# -*- coding: utf-8 -*-
"""双模型集成：训练第二模型 → logits 平均 → 校准 → 与 v5 基线对比"""
import json
import os
import shutil

import numpy as np
import torch
from sklearn import metrics
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

import main as m
from main import (
    Model,
    TextDataset,
    apply_calibration,
    build_sampler,
    calibrate_inference,
    collect_logits_and_hits,
    dev_eval,
    get_data,
    init_network,
    num_classes,
    plot_confusion_matrix,
    save_calibration,
    train,
    label_names,
)

BASELINE_EVAL = './results/eval_latest.json'
MODEL_A_PATH = './saved_dict/lstm.ckpt'
MODEL_B_PATH = './saved_dict/lstm_ensemble_b.ckpt'
ENSEMBLE_META = './saved_dict/ensemble_config.json'
RESULT_ENSEMBLE = './results/eval_ensemble_trial.json'


def load_baseline():
    """读取 v5 单模型基线指标"""
    with open(BASELINE_EVAL, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return float(data['accuracy']), float(data['macro_f1'])


def set_seed(seed):
    """固定随机种子，便于第二模型与主模型产生差异"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_averaged_logits(model_a, model_b, data):
    """两模型 logits 逐样本平均"""
    model_a.eval()
    model_b.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for texts, labels in data:
            la = model_a(texts).cpu().numpy()
            lb = model_b(texts).cpu().numpy()
            logits_list.append((la + lb) / 2.0)
            labels_list.append(labels.cpu().numpy())
    return np.concatenate(logits_list), np.concatenate(labels_list)


def ensemble_eval(model_a, model_b, data, loss_function, cal, inv_vocab, report_path=None):
    """集成模型在指定集合上评估（含校准）"""
    model_a.eval()
    model_b.eval()
    loss_total = 0.0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data:
            logits = (model_a(texts) + model_b(texts)) / 2.0
            loss = loss_function(logits, labels)
            loss_total += loss.item()
            logits_np = logits.cpu().numpy()
            pos_hits, neg_hits, neu_hits = [], [], []
            for row in texts.cpu().numpy():
                chars = []
                for idx in row:
                    if int(idx) in (0, 1):
                        continue
                    c = inv_vocab.get(int(idx), '')
                    if c and c not in (m.PAD, m.UNK):
                        chars.append(c)
                from sentiment_lexicon import count_lexicon_hits
                p, n, u = count_lexicon_hits(''.join(chars))
                pos_hits.append(p)
                neg_hits.append(n)
                neu_hits.append(u)
            predic = apply_calibration(
                logits_np, cal,
                np.array(pos_hits), np.array(neg_hits), np.array(neu_hits),
            )
            labels_np = labels.data.cpu().numpy()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, labels_np)

    acc = metrics.accuracy_score(labels_all, predict_all)
    macro_f1 = metrics.f1_score(labels_all, predict_all, average='macro', zero_division=0)

    if report_path:
        cm = metrics.confusion_matrix(labels_all, predict_all)
        report_dict = metrics.classification_report(
            labels_all, predict_all, target_names=label_names,
            digits=4, output_dict=True, zero_division=0,
        )
        payload = {
            'accuracy': float(acc),
            'macro_f1': float(macro_f1),
            'calibration': cal,
            'classification_report': report_dict,
            'confusion_matrix': cm.tolist(),
            'ensemble': True,
            'model_a': MODEL_A_PATH,
            'model_b': MODEL_B_PATH,
        }
        with open(report_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        plot_confusion_matrix(cm)

    return acc, loss_total / len(data), macro_f1


def cleanup_trial():
    """集成未提升时清理试验产物，不影响 v5"""
    for path in (MODEL_B_PATH, ENSEMBLE_META, RESULT_ENSEMBLE):
        if os.path.exists(path):
            os.remove(path)
    print("集成路线未带来提升，已清理试验文件，保持 v5 单模型结果不变。")


def train_model_b(vocab, train_data, dev_data, test_data, inv_vocab, device, class_weights):
    """训练第二模型：不同种子与超参，增加与模型 A 的差异"""
    orig_dropout = m.dropout
    orig_lr = m.learning_rate
    orig_save = m.save_path
    try:
        m.dropout = 0.45
        m.learning_rate = 8e-4
        m.save_path = MODEL_B_PATH
        set_seed(2024)

        train_sampler = build_sampler(train_data)
        dataloaders = {
            'train': DataLoader(TextDataset(train_data), m.batch_size, sampler=train_sampler),
            'dev': DataLoader(TextDataset(dev_data), m.batch_size),
            'test': DataLoader(TextDataset(test_data), m.batch_size),
        }
        model_b = Model(vocab).to(device)
        init_network(model_b)
        print(
            f"\n训练集成子模型 B | dropout={m.dropout} | lr={m.learning_rate} | seed=2024"
        )
        train(model_b, dataloaders, class_weights, inv_vocab)
        model_b.load_state_dict(torch.load(MODEL_B_PATH, map_location=device))
        return model_b, dataloaders
    finally:
        m.dropout = orig_dropout
        m.learning_rate = orig_lr
        m.save_path = orig_save


def adopt_ensemble(cal_params, test_acc, test_f1, baseline_acc, baseline_f1):
    """集成有效：写入正式结果与配置"""
    shutil.copy2(RESULT_ENSEMBLE, BASELINE_EVAL)
    save_calibration(cal_params)
    meta = {
        'enabled': True,
        'model_a': MODEL_A_PATH,
        'model_b': MODEL_B_PATH,
        'fusion': 'logits_average',
        'baseline_accuracy': baseline_acc,
        'baseline_macro_f1': baseline_f1,
        'ensemble_accuracy': test_acc,
        'ensemble_macro_f1': test_f1,
    }
    with open(ENSEMBLE_META, 'w', encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"集成方案已采纳，正式结果已写入 {BASELINE_EVAL}")


def main():
    if not os.path.exists(MODEL_A_PATH):
        raise FileNotFoundError(f"未找到 v5 模型: {MODEL_A_PATH}")

    baseline_acc, baseline_f1 = load_baseline()
    print(f"v5 基线 -> 准确率: {baseline_acc:.4%} | macro-F1: {baseline_f1:.4f}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    vocab, train_data, dev_data, test_data = get_data()
    inv_vocab = {v: k for k, v in vocab.items()}

    train_labels = [item[1] for item in train_data]
    raw_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    weight_values = np.sqrt(raw_weights) / np.sqrt(raw_weights).mean()
    weight_values[0] *= 1.10
    weight_values = weight_values / weight_values.mean()
    class_weights = torch.tensor(weight_values, dtype=torch.float32).to(device)
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    model_b, dataloaders = train_model_b(
        vocab, train_data, dev_data, test_data, inv_vocab, device, class_weights
    )

    model_a = Model(vocab).to(device)
    model_a.load_state_dict(torch.load(MODEL_A_PATH, map_location=device))

    dev_logits, dev_labels, dev_pos_h, dev_neg_h, dev_neu_h = collect_logits_and_hits(
        model_a, dataloaders['dev'], inv_vocab
    )
    dev_logits_b, _, _, _, _ = collect_logits_and_hits(model_b, dataloaders['dev'], inv_vocab)
    ens_dev_logits = (dev_logits + dev_logits_b) / 2.0

    cal_params, cal_dev_f1 = calibrate_inference(
        ens_dev_logits, dev_labels, dev_pos_h, dev_neg_h, dev_neu_h
    )
    print(
        f"\n集成推理校准: bias={cal_params['neutral_bias']:.2f} | "
        f"margin={cal_params['emotion_margin']:.2f} | "
        f"neu_min={cal_params['neu_min_prob']:.2f} | "
        f"strong={cal_params['strong_threshold']:.2f} | "
        f"lex=({cal_params['lex_pos_w']:.2f},{cal_params['lex_neg_w']:.2f}) | "
        f"neu_hint={cal_params.get('neu_hint_w', 0):.2f} | "
        f"验证集 macro-F1={cal_dev_f1:.4f}"
    )

    print("\n" + "=" * 50)
    print("集成模型测试集评估")
    print("=" * 50)
    test_acc, test_loss, test_f1 = ensemble_eval(
        model_a, model_b, dataloaders['test'], loss_function,
        cal_params, inv_vocab, report_path=RESULT_ENSEMBLE,
    )
    print(f"测试集 loss:{test_loss:.3f} | acc:{test_acc:.2%} | macro-F1:{test_f1:.4f}")
    print(f"对比 v5 基线 acc:{baseline_acc:.2%} | macro-F1:{baseline_f1:.4f}")

    # 准确率必须严格提升才采纳；准确率持平时 macro-F1 需明显提升
    acc_gain = test_acc - baseline_acc
    f1_gain = test_f1 - baseline_f1
    improved = acc_gain > 1e-9 or (abs(acc_gain) <= 1e-9 and f1_gain > 0.001)

    if improved:
        print("\n集成带来提升，采纳双模型方案。")
        adopt_ensemble(cal_params, test_acc, test_f1, baseline_acc, baseline_f1)
        return True, test_acc, test_f1, baseline_acc, baseline_f1, cal_params

    print("\n集成未超过 v5 基线，舍弃该路线。")
    cleanup_trial()
    return False, test_acc, test_f1, baseline_acc, baseline_f1, cal_params


if __name__ == '__main__':
    main()
