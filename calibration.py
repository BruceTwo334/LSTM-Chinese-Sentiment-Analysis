# -*- coding: utf-8 -*-
"""推理阶段概率校准：强情感保护 + 中性判定 + 词典辅助 v5"""
import pickle as pkl
import numpy as np
from sklearn import metrics
from sentiment_lexicon import count_lexicon_hits

DEFAULT_CAL = {
    'neutral_bias': 0.0,
    'emotion_margin': 0.30,
    'neu_min_prob': 0.40,
    'strong_threshold': 0.45,
    'lex_pos_w': 0.0,
    'lex_neg_w': 0.0,
    'lex_hit_threshold': 2,
    'neu_hint_w': 0.0,
}


def softmax_logits(logits):
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


def apply_lexicon_logit_boost(logits, pos_hits, neg_hits, lex_pos_w, lex_neg_w, neu_hits=None, neu_hint_w=0.0):
    """词典命中加权到 logits"""
    if pos_hits is None or neg_hits is None:
        return logits
    boosted = logits.copy()
    boosted[:, 0] += pos_hits * lex_pos_w
    boosted[:, 2] += neg_hits * lex_neg_w
    if neu_hits is not None and neu_hint_w > 0:
        boosted[:, 1] += neu_hits * neu_hint_w
    return boosted


def predict_from_logits(
    logits,
    neutral_bias=0.0,
    emotion_margin=0.30,
    neu_min_prob=0.40,
    strong_threshold=0.45,
    pos_hits=None,
    neg_hits=None,
    lex_pos_w=0.0,
    lex_neg_w=0.0,
    lex_hit_threshold=2,
    neu_hits=None,
    neu_hint_w=0.0,
):
    """v5：词典加权 + 中性提示 + 强情感保护"""
    logits = apply_lexicon_logit_boost(
        logits, pos_hits, neg_hits, lex_pos_w, lex_neg_w, neu_hits, neu_hint_w
    )
    adjusted = logits.copy()
    adjusted[:, 1] += neutral_bias
    probs = softmax_logits(adjusted)
    preds = []
    for i, prob in enumerate(probs):
        pos_p, neu_p, neg_p = float(prob[0]), float(prob[1]), float(prob[2])
        ph = int(pos_hits[i]) if pos_hits is not None else 0
        nh = int(neg_hits[i]) if neg_hits is not None else 0
        uh = int(neu_hits[i]) if neu_hits is not None else 0

        if ph >= lex_hit_threshold and nh == 0 and pos_p >= neg_p:
            preds.append(0)
            continue
        if nh >= lex_hit_threshold and ph == 0 and neg_p >= pos_p:
            preds.append(2)
            continue
        if uh >= 1 and ph == 0 and nh == 0 and neu_p >= pos_p and neu_p >= neg_p:
            preds.append(1)
            continue

        if pos_p >= strong_threshold and pos_p > neg_p and pos_p > neu_p:
            preds.append(0)
        elif neg_p >= strong_threshold and neg_p > pos_p and neg_p > neu_p:
            preds.append(2)
        elif (
            neu_p >= neu_min_prob
            and abs(pos_p - neg_p) < emotion_margin
            and neu_p >= pos_p
            and neu_p >= neg_p
        ):
            preds.append(1)
        else:
            preds.append(int(np.argmax(prob)))
    return np.array(preds)


def calibrate_inference(logits, labels, pos_hits=None, neg_hits=None, neu_hits=None):
    """在验证集上搜索校准参数，最大化 macro-F1"""
    best = dict(DEFAULT_CAL)
    best_f1 = 0.0
    for bias in np.arange(-0.2, 0.5, 0.1):
        for margin in np.arange(0.06, 0.28, 0.04):
            for neu_min in np.arange(0.18, 0.42, 0.04):
                for strong in np.arange(0.28, 0.48, 0.04):
                    for lpw in (0.0, 0.3, 0.6):
                        for lnw in (0.0, 0.3, 0.6):
                            for nhw in (0.0, 0.3, 0.6):
                                for lht in (1, 2):
                                    preds = predict_from_logits(
                                        logits, bias, margin, neu_min, strong,
                                        pos_hits, neg_hits, lpw, lnw, lht,
                                        neu_hits, nhw,
                                    )
                                    f1 = metrics.f1_score(
                                        labels, preds, average='macro', zero_division=0
                                    )
                                    if f1 > best_f1:
                                        best_f1 = f1
                                        best = {
                                            'neutral_bias': float(bias),
                                            'emotion_margin': float(margin),
                                            'neu_min_prob': float(neu_min),
                                            'strong_threshold': float(strong),
                                            'lex_pos_w': float(lpw),
                                            'lex_neg_w': float(lnw),
                                            'lex_hit_threshold': int(lht),
                                            'neu_hint_w': float(nhw),
                                        }
    return best, best_f1


def load_calibration(path='./saved_dict/calibration.pkl'):
    try:
        cal = pkl.load(open(path, 'rb'))
        merged = DEFAULT_CAL.copy()
        merged.update(cal)
        return merged
    except FileNotFoundError:
        return DEFAULT_CAL.copy()


def save_calibration(cal, path='./saved_dict/calibration.pkl'):
    pkl.dump(cal, open(path, 'wb'))
