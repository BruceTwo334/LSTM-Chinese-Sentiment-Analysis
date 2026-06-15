# -*- coding: utf-8 -*-
import numpy as np
import pickle as pkl
from tqdm import tqdm
from datetime import timedelta
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torch.nn as nn
import time
import torch
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from preprocess import build_balanced_dataset_v5, clean_weibo_text
from calibration import (
    predict_from_logits, calibrate_inference, save_calibration,
)
from sentiment_lexicon import count_lexicon_hits, char_keyword_mask

os.makedirs('saved_dict', exist_ok=True)
os.makedirs('results', exist_ok=True)

# ===================== 训练配置 =====================
data_path = './data/weibo_3class_balanced_v5.csv'  # v5：错例定向 + pad50 + 词典 v2
raw_data_path = './data/weibo_3class.csv'
vocab_path = './data/vocab.pkl'
save_path = './saved_dict/lstm.ckpt'
embed = 200
dropout = 0.5          # Dropout 正则
num_classes = 3
num_epochs = 20        # 配合早停使用
batch_size = 128
pad_size = 50          # v5：加长序列，减少长微博截断
learning_rate = 1e-3
weight_decay = 1e-4    # L2 正则化
hidden_size = 128
num_layers = 2
early_stop_patience = 4  # 早停耐心值（按 macro-F1）
UNK, PAD = 'UNK', 'PAD'
label_names = ['积极', '中性', '消极']

# ===================== 公开预训练词向量 =====================
embedding_Tencent = np.load('./data/embedding_Tencent.npz')
print("文件包含的 keys:", list(embedding_Tencent.keys()))
if 'embeddings' in embedding_Tencent:
    embedding_pretrained = torch.tensor(embedding_Tencent['embeddings'].astype('float32'))
elif 'arr_0' in embedding_Tencent:
    embedding_pretrained = torch.tensor(embedding_Tencent['arr_0'].astype('float32'))
else:
    first_key = list(embedding_Tencent.keys())[0]
    embedding_pretrained = torch.tensor(embedding_Tencent[first_key].astype('float32'))


def ensure_balanced_dataset():
    """若 v5 均衡数据集不存在，则从原始数据构建"""
    stats_file = './results/preprocess_v5_stats.json'
    if not os.path.exists(data_path) or not os.path.exists(stats_file):
        print("正在构建 v5 均衡数据集（错例定向 + 边界样本）...")
        build_balanced_dataset_v5(raw_data_path, data_path)
    else:
        print(f"使用均衡数据集: {data_path}")


def get_data():
    ensure_balanced_dataset()
    tokenizer = lambda x: [y for y in x]
    vocab = pkl.load(open(vocab_path, 'rb'))
    print(f"Vocab size: {len(vocab)}")
    train, dev, test = load_dataset(data_path, pad_size, tokenizer, vocab)
    return vocab, train, dev, test


def load_dataset(path, pad_size, tokenizer, vocab):
    contents = []
    labels = []
    df = pd.read_csv(path, header=None, names=['label', 'text'])
    df = df.dropna()
    df = df[df['label'].isin([0, 1, 2])]

    for idx, row in df.iterrows():
        label = int(row['label'])
        # 均衡 CSV 已预处理，直接使用
        content = str(row['text']).strip()
        if len(content) < 2:
            continue
        token = tokenizer(content)
        seq_len = len(token)

        if pad_size:
            if len(token) < pad_size:
                token.extend([PAD] * (pad_size - len(token)))
            else:
                token = token[:pad_size]
                seq_len = pad_size

        unk_idx = vocab.get(UNK, 1)
        words_line = []
        for word in token:
            words_line.append(vocab.get(word, unk_idx))
        contents.append((words_line, label))
        labels.append(label)

    # 分层划分 6:2:2，避免类别扎堆
    train, X_t, y_train, y_t = train_test_split(
        contents, labels, test_size=0.4, random_state=42, stratify=labels
    )
    dev, test, _, _ = train_test_split(
        X_t, y_t, test_size=0.5, random_state=42, stratify=y_t
    )

    # 打印各集合类别分布
    for name, subset in [('训练集', train), ('验证集', dev), ('测试集', test)]:
        counts = np.bincount([item[1] for item in subset], minlength=num_classes)
        print(f"{name}分布 -> 积极:{counts[0]} 中性:{counts[1]} 消极:{counts[2]}")
    return train, dev, test


def build_sampler(train_data):
    """均衡数据集下使用标准加权采样（三类已接近 1:1:1）"""
    labels = [item[1] for item in train_data]
    class_weights = compute_class_weight('balanced', classes=np.arange(num_classes), y=labels)
    sample_weights = [class_weights[l] for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def collect_logits_and_hits(model, data, inv_vocab):
    """收集 logits、标签与词典命中数（含中性提示）"""
    model.eval()
    logits_list, labels_list = [], []
    pos_hits, neg_hits, neu_hits = [], [], []
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            logits_list.append(outputs.cpu().numpy())
            labels_list.append(labels.cpu().numpy())
            for row in texts.cpu().numpy():
                chars = []
                for idx in row:
                    if int(idx) in (0, 1):
                        continue
                    c = inv_vocab.get(int(idx), '')
                    if c and c not in (PAD, UNK):
                        chars.append(c)
                p, n, u = count_lexicon_hits(''.join(chars))
                pos_hits.append(p)
                neg_hits.append(n)
                neu_hits.append(u)
    return (
        np.concatenate(logits_list),
        np.concatenate(labels_list),
        np.array(pos_hits),
        np.array(neg_hits),
        np.array(neu_hits),
    )


def apply_calibration(logits, cal, pos_hits=None, neg_hits=None, neu_hits=None):
    """批量预测（评估用）"""
    return predict_from_logits(
        logits,
        neutral_bias=cal['neutral_bias'],
        emotion_margin=cal['emotion_margin'],
        neu_min_prob=cal['neu_min_prob'],
        strong_threshold=cal['strong_threshold'],
        pos_hits=pos_hits,
        neg_hits=neg_hits,
        lex_pos_w=cal.get('lex_pos_w', 0.0),
        lex_neg_w=cal.get('lex_neg_w', 0.0),
        lex_hit_threshold=cal.get('lex_hit_threshold', 2),
        neu_hits=neu_hits,
        neu_hint_w=cal.get('neu_hint_w', 0.0),
    )


class TextDataset(Dataset):
    def __init__(self, data):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.x_list = []
        self.y_list = []
        for item in data:
            if item is not None:
                self.x_list.append(item[0])
                self.y_list.append(item[1])
        self.x = torch.LongTensor(self.x_list).to(self.device)
        self.y = torch.LongTensor(self.y_list).to(self.device)

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


def get_time_dif(start_time):
    end_time = time.time()
    time_dif = end_time - start_time
    return timedelta(seconds=int(round(time_dif)))


# ===================== 模型：BiLSTM + 注意力 + 情感词典加权 =====================
class Model(nn.Module):
    def __init__(self, vocab=None):
        super(Model, self).__init__()
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(
            embed, hidden_size, num_layers,
            bidirectional=True, batch_first=True, dropout=dropout
        )
        self.attention = nn.Linear(hidden_size * 2, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # 情感词典：注意力加权强度与 logit 偏置强度（可学习）
        self.kw_attn_scale = nn.Parameter(torch.tensor(2.0))
        self.lex_pos_scale = nn.Parameter(torch.tensor(0.8))
        self.lex_neg_scale = nn.Parameter(torch.tensor(0.8))

    def _decode_row(self, row):
        chars = []
        for idx in row:
            if int(idx) in (0, 1):
                continue
            c = self.inv_vocab.get(int(idx), '')
            if c and c not in (PAD, UNK):
                chars.append(c)
        return ''.join(chars)

    def _keyword_mask_batch(self, x):
        """关键词位置掩码，用于增强注意力"""
        masks = []
        for row in x.cpu().numpy():
            text = self._decode_row(row)
            masks.append(char_keyword_mask(text, len(row)))
        return torch.tensor(masks, dtype=torch.float32, device=x.device).unsqueeze(-1)

    def _lexicon_hits_batch(self, x):
        pos_h, neg_h = [], []
        for row in x.cpu().numpy():
            p, n, _ = count_lexicon_hits(self._decode_row(row))
            pos_h.append(p)
            neg_h.append(n)
        return (
            torch.tensor(pos_h, dtype=torch.float32, device=x.device),
            torch.tensor(neg_h, dtype=torch.float32, device=x.device),
        )

    def forward(self, x):
        out = self.embedding(x)
        out, _ = self.lstm(out)
        attn_scores = self.attention(out)
        # 词典关键词位置加权注意力
        kw_mask = self._keyword_mask_batch(x)
        attn_scores = attn_scores + self.kw_attn_scale * kw_mask
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        context = self.dropout_layer(context)
        logits = self.fc(context)
        # 词典命中数加权到输出 logits
        pos_h, neg_h = self._lexicon_hits_batch(x)
        logits[:, 0] = logits[:, 0] + self.lex_pos_scale * pos_h
        logits[:, 2] = logits[:, 2] + self.lex_neg_scale * neg_h
        return logits


def init_network(model, method='xavier', exclude='embedding'):
    for name, w in model.named_parameters():
        if exclude not in name:
            if 'weight' in name:
                nn.init.xavier_normal_(w)
            elif 'bias' in name:
                nn.init.constant_(w, 0)


def plot_acc(train_acc):
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(train_acc)), train_acc)
    plt.xlabel('Epoch')
    plt.ylabel('Acc')
    plt.savefig('results/acc.png')
    plt.close()


def plot_loss(train_loss):
    plt.figure(figsize=(10, 7))
    plt.plot(range(len(train_loss)), train_loss, color='red')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.savefig('results/loss.png')
    plt.close()


def plot_confusion_matrix(cm, save_path='results/confusion_matrix.png'):
    """绘制混淆矩阵热力图"""
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('混淆矩阵')
    plt.colorbar()
    tick_marks = np.arange(len(label_names))
    plt.xticks(tick_marks, label_names)
    plt.yticks(tick_marks, label_names)
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     ha="center", va="center",
                     color="white" if cm[i, j] > thresh else "black")
    plt.ylabel('真实标签')
    plt.xlabel('预测标签')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def dev_eval(model, data, loss_function, Result_test=False, cal=None, report_path=None, inv_vocab=None):
    if cal is None:
        cal = {'neutral_bias': 0.0, 'emotion_margin': 1.0, 'neu_min_prob': 0.0,
               'strong_threshold': 1.0, 'lex_pos_w': 0.0, 'lex_neg_w': 0.0,
               'lex_hit_threshold': 2, 'neu_hint_w': 0.0}
    model.eval()
    loss_total = 0
    predict_all = np.array([], dtype=int)
    labels_all = np.array([], dtype=int)
    with torch.no_grad():
        for texts, labels in data:
            outputs = model(texts)
            loss = loss_function(outputs, labels)
            loss_total += loss.item()
            logits = outputs.cpu().numpy()
            pos_hits, neg_hits, neu_hits = [], [], []
            if inv_vocab:
                for row in texts.cpu().numpy():
                    chars = []
                    for idx in row:
                        if int(idx) in (0, 1):
                            continue
                        c = inv_vocab.get(int(idx), '')
                        if c and c not in (PAD, UNK):
                            chars.append(c)
                    p, n, u = count_lexicon_hits(''.join(chars))
                    pos_hits.append(p)
                    neg_hits.append(n)
                    neu_hits.append(u)
                pos_hits = np.array(pos_hits)
                neg_hits = np.array(neg_hits)
                neu_hits = np.array(neu_hits)
            predic = apply_calibration(logits, cal, pos_hits, neg_hits, neu_hits)
            labels = labels.data.cpu().numpy()
            predict_all = np.append(predict_all, predic)
            labels_all = np.append(labels_all, labels)
    acc = metrics.accuracy_score(labels_all, predict_all)
    macro_f1 = metrics.f1_score(labels_all, predict_all, average='macro', zero_division=0)

    if Result_test:
        # 多维度评估：精确率、召回率、F1、混淆矩阵
        print("\n" + "=" * 50)
        print("测试集详细评估报告")
        print("=" * 50)
        print(f"准确率: {acc:.2%}")
        print("\n各类别指标:")
        report = metrics.classification_report(
            labels_all, predict_all, target_names=label_names, digits=4
        )
        print(report)
        cm = metrics.confusion_matrix(labels_all, predict_all)
        print("混淆矩阵 (行=真实, 列=预测):")
        print(f"       积极  中性  消极")
        for i, name in enumerate(label_names):
            print(f"  {name}  {cm[i]}")
        plot_confusion_matrix(cm)
        if report_path:
            report_dict = metrics.classification_report(
                labels_all, predict_all, target_names=label_names,
                digits=4, output_dict=True, zero_division=0
            )
            payload = {
                'accuracy': float(acc),
                'macro_f1': float(macro_f1),
                'calibration': cal,
                'classification_report': report_dict,
                'confusion_matrix': cm.tolist(),
            }
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)

    return acc, loss_total / len(data), macro_f1


def train(model, dataloaders, class_weights, inv_vocab):
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    # 类别加权损失，重点提升中性类识别
    loss_function = torch.nn.CrossEntropyLoss(weight=class_weights)
    dev_best_f1 = 0.0
    no_improve_epochs = 0
    plot_train_acc = []
    plot_train_loss = []

    for i in range(num_epochs):
        model.train()
        step = 0
        train_lossi = 0
        train_acci = 0
        for inputs, labels in tqdm(dataloaders['train'], desc=f"Epoch {i+1}/{num_epochs}"):
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = loss_function(outputs, labels)
            loss.backward()
            optimizer.step()
            step += 1
            train_lossi += loss.item()
            predic = torch.max(outputs.data, 1)[1].cpu()
            train_acci += metrics.accuracy_score(labels.data.cpu(), predic)

        dev_acc, dev_loss, dev_f1 = dev_eval(
            model, dataloaders['dev'], loss_function, inv_vocab=inv_vocab
        )
        if dev_f1 > dev_best_f1:
            dev_best_f1 = dev_f1
            torch.save(model.state_dict(), save_path)
            no_improve_epochs = 0
            print(f"  -> 验证集 macro-F1 提升至 {dev_f1:.4f}，已保存最优模型")
        else:
            no_improve_epochs += 1
            if no_improve_epochs >= early_stop_patience:
                print(f"早停触发：验证集 macro-F1 连续 {early_stop_patience} 轮未改善")
                break

        train_acc = train_acci / step
        train_loss = train_lossi / step
        plot_train_acc.append(train_acc)
        plot_train_loss.append(train_loss)
        print(
            f"epoch {i+1} | train_loss:{train_loss:.3f} | train_acc:{train_acc:.2%} "
            f"| dev_loss:{dev_loss:.3f} | dev_acc:{dev_acc:.2%} | dev_f1:{dev_f1:.4f}"
        )

    plot_acc(plot_train_acc)
    plot_loss(plot_train_loss)


if __name__ == '__main__':
    start_time = time.time()
    vocab, train_data, dev_data, test_data = get_data()

    # 计算类别权重用于损失函数
    train_labels = [item[1] for item in train_data]
    # 使用平方根缓和权重，避免中性类权重过大导致全盘预测中性
    raw_weights = compute_class_weight(
        'balanced', classes=np.arange(num_classes), y=train_labels
    )
    weight_values = np.sqrt(raw_weights)
    weight_values = weight_values / weight_values.mean()
    # 略提高积极类权重，缓解积极召回偏低
    weight_values[0] *= 1.10
    weight_values = weight_values / weight_values.mean()
    device = torch.device('cuda') if torch.cuda.is_available() else 'cpu'
    class_weights = torch.tensor(weight_values, dtype=torch.float32).to(device)
    print(
        f"类别损失权重 -> 积极:{weight_values[0]:.3f} "
        f"中性:{weight_values[1]:.3f} 消极:{weight_values[2]:.3f}"
    )

    train_sampler = build_sampler(train_data)
    dataloaders = {
        'train': DataLoader(TextDataset(train_data), batch_size, sampler=train_sampler),
        'dev': DataLoader(TextDataset(dev_data), batch_size),
        'test': DataLoader(TextDataset(test_data), batch_size)
    }

    model = Model(vocab).to(device)
    init_network(model)
    inv_vocab = {v: k for k, v in vocab.items()}
    train(model, dataloaders, class_weights, inv_vocab)

    # 加载最优模型，校准中性类偏置后在测试集评估
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
        f"\n推理参数校准: bias={cal_params['neutral_bias']:.2f} | "
        f"margin={cal_params['emotion_margin']:.2f} | "
        f"neu_min={cal_params['neu_min_prob']:.2f} | "
        f"strong={cal_params['strong_threshold']:.2f} | "
        f"lex=({cal_params['lex_pos_w']:.2f},{cal_params['lex_neg_w']:.2f}) | "
        f"neu_hint={cal_params.get('neu_hint_w', 0):.2f} | "
        f"验证集 macro-F1={cal_dev_f1:.4f}"
    )

    test_acc, test_loss, test_f1 = dev_eval(
        model, dataloaders['test'], loss_function,
        Result_test=True,
        cal=cal_params,
        report_path='./results/eval_latest.json',
        inv_vocab=inv_vocab,
    )
    print(f"\n测试集 loss:{test_loss:.3f} | acc:{test_acc:.2%} | macro-F1:{test_f1:.4f}")
    print(f"训练总耗时: {get_time_dif(start_time)}")
