# -*- coding: utf-8 -*-
"""加载已训练模型，校准中性类偏置并评估（无需重新训练）"""
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import DataLoader

from main import (
    Model, TextDataset, get_data, build_sampler,
    collect_logits, calibrate_inference, dev_eval,
    compute_class_weight, label_names,
)

device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
vocab, train_data, dev_data, test_data = get_data()

train_labels = [item[1] for item in train_data]
raw_weights = compute_class_weight('balanced', classes=np.arange(3), y=train_labels)
weight_values = np.sqrt(raw_weights) / np.sqrt(raw_weights).mean()
class_weights = torch.tensor(weight_values, dtype=torch.float32).to(device)

train_sampler = build_sampler(train_data)
dataloaders = {
    'train': DataLoader(TextDataset(train_data), 128, sampler=train_sampler),
    'dev': DataLoader(TextDataset(dev_data), 128),
    'test': DataLoader(TextDataset(test_data), 128),
}

model = Model().to(device)
model.load_state_dict(torch.load('./saved_dict/lstm.ckpt', map_location=device))
loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)

dev_logits, dev_labels = collect_logits(model, dataloaders['dev'])
neutral_bias, emotion_margin, neu_min_prob, cal_dev_f1 = calibrate_inference(
    dev_logits, dev_labels
)
cal_params = {
    'neutral_bias': neutral_bias,
    'emotion_margin': emotion_margin,
    'neu_min_prob': neu_min_prob,
}
pkl.dump(cal_params, open('./saved_dict/calibration.pkl', 'wb'))
print(
    f"推理参数校准: bias={neutral_bias:.2f} | margin={emotion_margin:.2f} "
    f"| neu_min={neu_min_prob:.2f} | 验证集 macro-F1={cal_dev_f1:.4f}"
)

test_acc, test_loss, test_f1 = dev_eval(
    model, dataloaders['test'], loss_function,
    Result_test=True,
    neutral_bias=neutral_bias,
    emotion_margin=emotion_margin,
    neu_min_prob=neu_min_prob,
)
print(f"测试集 acc:{test_acc:.2%} | macro-F1:{test_f1:.4f}")
