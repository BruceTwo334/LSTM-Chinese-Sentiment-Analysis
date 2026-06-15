# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
import pickle as pkl
from preprocess import clean_weibo_text
from calibration import load_calibration, predict_from_logits, softmax_logits
from sentiment_lexicon import count_lexicon_hits, char_keyword_mask

embed = 200
dropout = 0.5
num_classes = 3
pad_size = 50
UNK, PAD = 'UNK', 'PAD'
label_names = ['积极', '中性', '消极']

embedding_Tencent = np.load('./data/embedding_Tencent.npz')
if 'embeddings' in embedding_Tencent:
    embedding_pretrained = torch.tensor(embedding_Tencent['embeddings'].astype('float32'))
elif 'arr_0' in embedding_Tencent:
    embedding_pretrained = torch.tensor(embedding_Tencent['arr_0'].astype('float32'))
else:
    first_key = list(embedding_Tencent.keys())[0]
    embedding_pretrained = torch.tensor(embedding_Tencent[first_key].astype('float32'))


class Model(nn.Module):
    def __init__(self, vocab=None):
        super().__init__()
        self.vocab = vocab or {}
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
        self.embedding = nn.Embedding.from_pretrained(
            embedding_pretrained, freeze=False, padding_idx=0
        )
        self.lstm = nn.LSTM(embed, 128, 2, bidirectional=True, batch_first=True, dropout=dropout)
        self.attention = nn.Linear(128 * 2, 1)
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(128 * 2, num_classes)
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
        masks = []
        for row in x.cpu().numpy():
            masks.append(char_keyword_mask(self._decode_row(row), len(row)))
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
        kw_mask = self._keyword_mask_batch(x)
        attn_scores = attn_scores + self.kw_attn_scale * kw_mask
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(attn_weights * out, dim=1)
        context = self.dropout_layer(context)
        logits = self.fc(context)
        pos_h, neg_h = self._lexicon_hits_batch(x)
        logits[:, 0] = logits[:, 0] + self.lex_pos_scale * pos_h
        logits[:, 2] = logits[:, 2] + self.lex_neg_scale * neg_h
        return logits


vocab = pkl.load(open('./data/vocab.pkl', 'rb'))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Model(vocab).to(device)
model.load_state_dict(torch.load('./saved_dict/lstm.ckpt', map_location=device))
model.eval()

cal = load_calibration()
print(
    f"已加载推理校准: bias={cal['neutral_bias']:.2f} | margin={cal['emotion_margin']:.2f} | "
    f"neu_min={cal['neu_min_prob']:.2f} | strong={cal['strong_threshold']:.2f} | "
    f"lex=({cal.get('lex_pos_w', 0):.2f},{cal.get('lex_neg_w', 0):.2f}) | "
    f"neu_hint={cal.get('neu_hint_w', 0):.2f}"
)


def predict(sentence):
    """预测单句情感（v4：词典 + 校准）"""
    sentence = clean_weibo_text(sentence, label=None)
    words = list(sentence)
    if len(words) < pad_size:
        words += [PAD] * (pad_size - len(words))
    else:
        words = words[:pad_size]

    unk_idx = vocab.get(UNK, 1)
    ids = [vocab.get(w, unk_idx) for w in words]

    with torch.no_grad():
        inputs = torch.LongTensor([ids]).to(device)
        output = model(inputs)
        logits = output.cpu().numpy()
        p, n, u = count_lexicon_hits(sentence)
        pred = int(predict_from_logits(
            logits,
            pos_hits=np.array([p]),
            neg_hits=np.array([n]),
            neu_hits=np.array([u]),
            **cal,
        )[0])
        adjusted = logits.copy()
        adjusted[0, 1] += cal['neutral_bias']
        from calibration import apply_lexicon_logit_boost
        adjusted = apply_lexicon_logit_boost(
            adjusted, np.array([p]), np.array([n]),
            cal.get('lex_pos_w', 0), cal.get('lex_neg_w', 0),
            np.array([u]), cal.get('neu_hint_w', 0),
        )
        probs = softmax_logits(adjusted)[0]
    return label_names[pred], probs


if __name__ == '__main__':
    print("=" * 50)
    print("        微博情感分析模型测试结果")
    print("=" * 50)

    test_sentences = [
        "今天心情特别好，太开心了！",
        "服务态度极差，再也不会买了",
        "今天天气一般，正常上班",
        "这家店味道不错，推荐！",
        "真的气死我了，什么垃圾东西",
        "今天吃了饭，看了电影",
        "太喜欢这个产品了，超赞！",
        "糟糕透顶，完全不值得",
        "就这样吧，没什么感觉",
        "爱了爱了，无限回购！",
        "还行吧，普普通通",
        "一般般，没有惊喜也没有失望",
    ]

    for sent in test_sentences:
        label, probs = predict(sent)
        prob_str = " | ".join([f"{label_names[i]}:{probs[i]:.2%}" for i in range(num_classes)])
        print(f"输入：{sent}")
        print(f"预测：【{label}】")
        print(f"概率：{prob_str}")
        print("-" * 40)
