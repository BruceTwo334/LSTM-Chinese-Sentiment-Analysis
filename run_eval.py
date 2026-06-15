# -*- coding: utf-8 -*-
"""仅加载已训练模型，执行校准与测试评估"""
import torch
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from main import (
    get_data, Model, collect_logits_and_hits,
    calibrate_inference, save_calibration, dev_eval,
    TextDataset, num_classes, save_path,
)
from torch.utils.data import DataLoader

if __name__ == '__main__':
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    vocab, train_data, dev_data, test_data = get_data()
    inv_vocab = {v: k for k, v in vocab.items()}
    train_labels = [item[1] for item in train_data]
    raw_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=train_labels)
    weight_values = np.sqrt(raw_weights) / np.sqrt(raw_weights).mean()
    weight_values[0] *= 1.10
    weight_values = weight_values / weight_values.mean()
    class_weights = torch.tensor(weight_values, dtype=torch.float32).to(device)

    dataloaders = {
        'dev': DataLoader(TextDataset(dev_data), 128),
        'test': DataLoader(TextDataset(test_data), 128),
    }

    model = Model(vocab).to(device)
    model.load_state_dict(torch.load(save_path, map_location=device))
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

    dev_logits, dev_labels, dev_pos_h, dev_neg_h, dev_neu_h = collect_logits_and_hits(
        model, dataloaders['dev'], inv_vocab
    )
    cal_params, cal_dev_f1 = calibrate_inference(
        dev_logits, dev_labels, dev_pos_h, dev_neg_h, dev_neu_h
    )
    save_calibration(cal_params)
    print(
        f"推理参数校准: bias={cal_params['neutral_bias']:.2f} | "
        f"margin={cal_params['emotion_margin']:.2f} | "
        f"neu_min={cal_params['neu_min_prob']:.2f} | "
        f"strong={cal_params['strong_threshold']:.2f} | "
        f"lex=({cal_params['lex_pos_w']:.2f},{cal_params['lex_neg_w']:.2f}) | "
        f"neu_hint={cal_params.get('neu_hint_w', 0):.2f} | "
        f"验证集 macro-F1={cal_dev_f1:.4f}"
    )

    test_acc, test_loss, test_f1 = dev_eval(
        model, dataloaders['test'], loss_function,
        Result_test=True, cal=cal_params,
        report_path='./results/eval_latest.json',
        inv_vocab=inv_vocab,
    )
    print(f"测试集 loss:{test_loss:.3f} | acc:{test_acc:.2%} | macro-F1:{test_f1:.4f}")
