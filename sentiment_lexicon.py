# -*- coding: utf-8 -*-
"""中文情感词典：用于关键词命中统计与注意力加权"""

# 积极情感词（多字词优先匹配）
POSITIVE_WORDS = [
    '开心', '高兴', '喜欢', '满意', '幸福', '感动', '惊喜', '完美', '优秀', '超赞',
    '太棒', '很好', '不错', '推荐', '回购', '爱了', '超喜欢', '心情好', '特别好',
    '十分满意', '强烈推荐', '无限回购', '太开心', '真棒', '给力', '赞', '美好',
    '挺满意', '还可以', '还行', '愿意再', '偏正面', '值得买', '挺开心', '较满意',
    '体验不错', '味道不错', '服务不错', '总体满意', '超出预期', '会回购',
]

# 消极情感词
NEGATIVE_WORDS = [
    '失望', '生气', '愤怒', '糟糕', '差劲', '垃圾', '难受', '崩溃', '后悔', '讨厌',
    '极差', '太差', '不值得', '气死', '不会再', '再也不', '糟糕透顶', '差极了',
    '让人无法接受', '让人失望', '服务态度差', '质量差', '非常差', '很烂', '恶心',
    '不会再买', '不会再光顾', '略差', '不太好', '不太满意', '略有不满', '偏下',
    '希望改进', '使用感不佳', '不太符合', '略失望', '慢了点', '态度不好',
]

# 中性标志词（无明显情感倾向）
NEUTRAL_HINTS = [
    '一般', '普通', '平常', '正常', '没什么感觉', '不好不坏', '普普通通', '平平常常',
    '就这样', '没有惊喜', '没有失望', '照常', '按部就班', '不过如此', '平淡',
    '一般般', '还行吧', '马马虎虎', '能用就行', '也能接受', '不算好也不算差',
    '无喜无悲', '情绪稳定', '和平时一样', '没什么特别的',
]

# 否定结构：命中后削弱对应极性（简单规则）
NEGATION_PREFIXES = ['不', '没', '无', '别', '未', '不太', '不是很', '并不']


def _count_word_hits(text, words):
    return sum(1 for w in words if w in text)


def count_lexicon_hits(text):
    """统计文本中积极/消极/中性提示词命中数（含简单否定削弱）"""
    if not text:
        return 0, 0, 0
    pos = _count_word_hits(text, POSITIVE_WORDS)
    neg = _count_word_hits(text, NEGATIVE_WORDS)
    neu = _count_word_hits(text, NEUTRAL_HINTS)
    # 简单否定：「不太满意」等已在 NEGATIVE_WORDS；对「不错+不」类做削弱
    for neg_p in NEGATION_PREFIXES:
        for w in POSITIVE_WORDS:
            if f'{neg_p}{w}' in text:
                pos = max(0, pos - 1)
                neg += 1
    return pos, neg, neu


def char_keyword_mask(text, seq_len):
    """生成字符级关键词位置掩码，用于注意力加权"""
    mask = [0.0] * seq_len
    keywords = POSITIVE_WORDS + NEGATIVE_WORDS + NEUTRAL_HINTS
    for kw in keywords:
        start = 0
        while True:
            idx = text.find(kw, start)
            if idx == -1:
                break
            for j in range(idx, min(idx + len(kw), seq_len)):
                mask[j] = 1.0
            start = idx + 1
    return mask
