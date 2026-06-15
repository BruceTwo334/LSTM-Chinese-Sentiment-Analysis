# -*- coding: utf-8 -*-
"""
微博情感文本预处理 v2：
1. 折中清洗：去除 @/转发/链接，保留与标签情感一致的表情，去除不一致表情
2. 补充中性样本 + 下采样积极/消极，使三类数量基本一致
"""
import re
import json
import os
import random
import pandas as pd

RE_AT_USER = re.compile(r'@[\u4e00-\u9fa5\w\-]+')
RE_RETWEET = re.compile(r'//@.+?[:：]')
RE_REPLY = re.compile(r'回复@[^:：]+[:：]')
RE_URL = re.compile(r'https?://[^\s]+|t\.cn/[^\s]+')
RE_EMOJI_TAG = re.compile(r'\[([^\]]+)\]')
RE_TOPIC = re.compile(r'#([^#]+)#')
RE_SPACE = re.compile(r'\s+')

# 与标签一致时保留的微博表情（常见子集）
POSITIVE_EMOJI = {
    '哈哈', '嘻嘻', '呵呵', '偷笑', '太开心', '开心', '高兴', '笑哈哈', '爱你', '爱',
    '鼓掌', '赞', 'good', '威武', '棒', '耶', '给力', '花心', '亲亲', '亲', '可爱',
    '萌', '挤眼', '礼物', '蛋糕', '干杯', '喜', '恭喜', '坏笑', '做鬼脸', '馋嘴',
    '害羞', '得意', '顶', '支持', '握手', '抱抱', '心动', '太阳', '心', '红心',
    '好棒', '太棒', '好喜欢', '爱你哦', '爱你么么哒', '笑而不语', '笑cry',
}
NEGATIVE_EMOJI = {
    '怒', '愤怒', '生气', '怒骂', '哼', '鄙视', '抓狂', '崩溃', '泪', '流泪', '大哭',
    '伤心', '衰', '倒霉', '烦', '累', '困', '晕', '汗', '黑线', '无奈', '委屈', '可怜',
    '吐', '震惊', '吃惊', '害怕', '闭嘴', '弱', '左哼哼', '右哼哼', '打哈欠', '心碎',
    '失望', '泪奔', '悲伤', '生病', '感冒', '不要', 'NO', '怒骂', '崩溃', '抓狂',
}
NEUTRAL_EMOJI = {
    '思考', '疑问', '围观', '钟', '时间', '手机', '电脑', '书', '咖啡', '茶', '月亮',
    '星星', '云', '下雨', '晴天', '睡觉', '困', '汗', '擦汗', 'ok', 'OK', '握手',
}
# 推理时允许保留的全体情感表情（无标签时使用）
ALL_KNOWN_EMOJI = POSITIVE_EMOJI | NEGATIVE_EMOJI | NEUTRAL_EMOJI

LABEL_EMOJI_MAP = {
    0: POSITIVE_EMOJI,
    1: NEUTRAL_EMOJI,
    2: NEGATIVE_EMOJI,
}


def _filter_emoji_tags(text, allowed_set):
    """仅保留 allowed_set 中的表情标签，其余删除"""
    def repl(m):
        name = m.group(1)
        if name in allowed_set:
            return m.group(0)
        return ''
    return RE_EMOJI_TAG.sub(repl, text)


def clean_weibo_text(text, label=None):
    """
    清洗微博文本。
    label=None：推理模式，保留已知情感表情，删除未知方括号标签。
    label=0/1/2：训练模式，仅保留与该标签一致的表情。
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.strip()
    text = RE_URL.sub('', text)
    text = RE_REPLY.sub('', text)
    text = RE_RETWEET.sub('', text)
    text = RE_AT_USER.sub('', text)
    text = RE_TOPIC.sub(r'\1', text)
    if label is None:
        text = _filter_emoji_tags(text, ALL_KNOWN_EMOJI)
    else:
        allowed = LABEL_EMOJI_MAP.get(int(label), set())
        text = _filter_emoji_tags(text, allowed)
    text = RE_SPACE.sub('', text)
    return text.strip()


def analyze_text_features(series):
    s = series.astype(str)
    return {
        'count': len(s),
        'has_at': float(s.str.contains('@', regex=False).mean()),
        'has_url': float(s.str.contains(r'https?://|t\.cn/', regex=True).mean()),
        'has_emoji_tag': float(s.str.contains(r'\[[^\]]+\]', regex=True).mean()),
        'has_retweet': float(s.str.contains(r'//@', regex=True).mean()),
        'avg_len': float(s.str.len().mean()),
        'short_ratio_lt5': float((s.str.len() < 5).mean()),
    }


# 中性生成时禁止出现的情感词，避免与积极/消极混淆
NEUTRAL_FORBIDDEN = {
    '开心', '高兴', '喜欢', '爱', '棒', '赞', '优秀', '满意', '感动', '幸福',
    '差', '烂', '垃圾', '气', '怒', '讨厌', '糟糕', '失望', '恨', '烦死',
    '超赞', '回购', '推荐', '再也不', '垃圾东西',
}


def _has_forbidden(text):
    return any(w in text for w in NEUTRAL_FORBIDDEN)


def generate_neutral_samples(n, seed=42, clear_style=False):
    """
    生成中性陈述句。
    clear_style=True（v3）：更偏客观平淡，禁用情感词，强化「无情感」表述。
    """
    rng = random.Random(seed)
    time_words = ['今天', '明天', '上午', '下午', '晚上', '周末', '这周', '最近']
    if clear_style:
        actions = [
            '正常上班', '按时下班', '去了公司', '在家休息', '吃了午饭',
            '喝了杯水', '出门买了菜', '整理了房间', '接了个电话',
            '回复了消息', '提交了报告', '参加了会议', '等公交车',
            '坐地铁回家', '看了天气预报', '看了电影', '吃了饭',
        ]
        states = [
            '没什么感觉', '不好不坏', '一般般', '普普通通', '平平常常',
            '没什么特别的', '和平时一样', '还算正常', '就这样吧',
            '没有惊喜也没有失望', '正常进行', '按部就班', '不过如此',
            '天气一般', '心情平淡', '无喜无悲', '情绪稳定',
        ]
        extras = ['', '，继续日常事务。', '，没什么可多说的。', '，照常而已。']
        neutral_tags = ['[思考]', '[围观]', '[ok]', '[钟]']
        fixed = [
            '今天天气一般，正常上班',
            '今天吃了饭，看了电影',
            '就这样吧，没什么感觉',
            '还行吧，普普通通',
            '一般般，没有惊喜也没有失望',
            '下午三点有个会，按时参加',
            '收到通知，明天照常处理',
            '这件事已经了解，暂无后续',
            '数据已记录，等待下一步安排',
            '流程走完了，结果待确认',
        ]
    else:
        actions = [
            '正常上班', '按时下班', '去了公司', '在家休息', '看了会书', '吃了午饭',
            '喝了杯水', '出门买了菜', '整理了房间', '洗了个澡', '接了个电话',
            '回复了消息', '完成了作业', '提交了报告', '参加了会议', '路过一家店',
            '等公交车', '坐地铁回家', '睡了个午觉', '看了天气预报',
        ]
        states = [
            '没什么特别的', '和平时一样', '普普通通', '还算正常', '就这样',
            '没什么感觉', '不好不坏', '一般般', '平平常常', '马马虎虎',
            '没什么变化', '按部就班', '照常进行', '不过如此', '仅此而已',
        ]
        extras = [
            '。', '，已经习惯了。', '，继续忙手头的事。', '，然后继续工作。',
            '，没什么可多说的。', '，日子还是那样过。',
        ]
        neutral_tags = ['', '', '', '[思考]', '[围观]', '[ok]']
        fixed = []

    samples = list(fixed) if clear_style else []
    combos = []
    for t in time_words:
        for a in actions:
            for s in states:
                combos.append(f'{t}{a}，{s}')
                combos.append(f'{t}，{s}，{a}')
    rng.shuffle(combos)
    for base in combos:
        if len(samples) >= n:
            break
        tag = rng.choice(neutral_tags) if clear_style else rng.choice(['', '[思考]', '[围观]'])
        sent = f'{base}{tag}' if tag else base
        if clear_style and _has_forbidden(sent):
            continue
        sent = clean_weibo_text(sent, label=1)
        if len(sent) >= 4 and not _has_forbidden(sent):
            samples.append(sent)
    # 若组合不够，再用随机模板补齐
    attempts = 0
    while len(samples) < n and attempts < n * 20:
        attempts += 1
        t, a, s = rng.choice(time_words), rng.choice(actions), rng.choice(states)
        sent = f'{t}{a}，{s}{rng.choice(extras)}'
        if clear_style and _has_forbidden(sent):
            continue
        sent = clean_weibo_text(sent, label=1)
        if len(sent) >= 4 and not _has_forbidden(sent):
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def generate_strong_positive_samples(n, seed=42):
    """生成带一致表情的强积极样本"""
    rng = random.Random(seed)
    adj = ['开心', '满意', '喜欢', '棒', '赞', '幸福', '感动', '惊喜', '完美', '优秀']
    obj = ['这个产品', '这次服务', '这家店', '这次体验', '今天的心情', '这份礼物']
    emojis = ['[哈哈]', '[爱你]', '[赞]', '[鼓掌]', '[太开心]', '[给力]', '[耶]']
    templates = [
        '太{adj}了，{obj}真的超棒{emo}',
        '真的{adj}，{obj}让我很满意{emo}',
        '爱了爱了，{obj}无限回购{emo}',
        '{obj}{adj}极了，强烈推荐{emo}',
        '心情特别好，{obj}太赞了{emo}',
    ]
    samples = []
    for _ in range(n * 3):
        if len(samples) >= n:
            break
        sent = rng.choice(templates).format(
            adj=rng.choice(adj), obj=rng.choice(obj), emo=rng.choice(emojis)
        )
        sent = clean_weibo_text(sent, label=0)
        if len(sent) >= 4 and RE_EMOJI_TAG.search(sent):
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def generate_strong_negative_samples(n, seed=42):
    """生成带一致表情的强消极样本"""
    rng = random.Random(seed)
    adj = ['失望', '生气', '糟糕', '差劲', '烂', '难受', '崩溃', '后悔']
    obj = ['这次服务', '这家店', '这个东西', '这次体验', '产品质量', '物流速度']
    emojis = ['[怒]', '[抓狂]', '[泪]', '[鄙视]', '[衰]', '[失望]', '[黑线]']
    templates = [
        '太{adj}了，{obj}让人无法接受{emo}',
        '真的气死我了，{obj}太差了{emo}',
        '{obj}糟糕透顶，完全不值得{emo}',
        '服务态度极差，{obj}太差了{emo}',
        '再也不会买了，{obj}太让人失望{emo}',
    ]
    samples = []
    for _ in range(n * 3):
        if len(samples) >= n:
            break
        sent = rng.choice(templates).format(
            adj=rng.choice(adj), obj=rng.choice(obj), emo=rng.choice(emojis)
        )
        sent = clean_weibo_text(sent, label=2)
        if len(sent) >= 4 and RE_EMOJI_TAG.search(sent):
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def generate_strong_positive_plain(n, seed=42):
    """无表情的强积极口语样本"""
    rng = random.Random(seed)
    templates = [
        '今天心情特别好，太开心了',
        '太喜欢这个产品了，超赞',
        '爱了爱了，无限回购',
        '这家店味道不错，强烈推荐',
        '服务质量非常好，十分满意',
        '这次体验完美，真的开心',
        '产品质量优秀，非常满意',
        '心情超级好，特别高兴',
        '真的太好了，强烈推荐给大家',
        '太棒了，这次购物很满意',
    ]
    adj_obj = [
        ('开心', '今天'), ('满意', '这次服务'), ('喜欢', '这个东西'),
        ('高兴', '结果'), ('惊喜', '体验'), ('幸福', '生活'),
    ]
    samples = list(templates)
    for _ in range(n * 2):
        if len(samples) >= n:
            break
        adj, obj = rng.choice(adj_obj)
        tpl = rng.choice([
            f'{obj}让人{adj}，真的太好了',
            f'太{adj}了，{obj}超出预期',
            f'非常{adj}，{obj}值得回购',
        ])
        sent = clean_weibo_text(tpl, label=0)
        if len(sent) >= 4 and not RE_EMOJI_TAG.search(sent):
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def generate_strong_negative_plain(n, seed=42):
    """无表情的强消极口语样本"""
    rng = random.Random(seed)
    templates = [
        '服务态度极差，再也不会买了',
        '真的气死我了，什么垃圾东西',
        '糟糕透顶，完全不值得',
        '质量太差了，非常失望',
        '这次体验极差，让人生气',
        '太让人失望了，不会再光顾',
        '东西太烂，完全不值这个价',
        '服务恶劣，十分后悔购买',
        '差极了，浪费时间和金钱',
        '非常糟糕，强烈不推荐',
    ]
    samples = list(templates)
    for _ in range(n * 2):
        if len(samples) >= n:
            break
        obj = rng.choice(['这次服务', '这家店', '这个东西', '产品质量', '物流'])
        tpl = rng.choice([
            f'{obj}太差了，让人非常生气',
            f'太糟糕了，{obj}让人失望',
            f'真的气人，{obj}简直垃圾',
            f'{obj}糟糕透顶，再也不买',
        ])
        sent = clean_weibo_text(tpl, label=2)
        if len(sent) >= 4 and not RE_EMOJI_TAG.search(sent):
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def generate_weak_positive_samples(n, seed=42):
    """弱积极/口语积极：针对积极被误判为中性或消极"""
    rng = random.Random(seed)
    templates = [
        '还不错，挺满意的',
        '这次体验还可以，愿意再试',
        '味道还行，总体偏正面',
        '服务不错，会考虑回购',
        '质量较满意，比预期好',
        '整体体验不错，没有失望',
        '价格合适，用着挺开心',
        '包装一般但东西不错',
        '虽然等了一会，但结果满意',
        '比想象中好，挺高兴的',
    ]
    samples = list(templates)
    objs = ['这次服务', '这家店', '产品', '快递', '体验', '味道']
    for _ in range(n * 2):
        if len(samples) >= n:
            break
        obj = rng.choice(objs)
        tpl = rng.choice([
            f'{obj}还可以，总体满意',
            f'{obj}不错，愿意再买',
            f'{obj}挺满意，没有踩雷',
            f'虽然{obj}普通，但结果还行',
        ])
        sent = clean_weibo_text(tpl, label=0)
        if len(sent) >= 4:
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def generate_borderline_neutral_samples(n, seed=42):
    """边界中性：含「一般/普通/轻微抱怨」但情感仍中性"""
    rng = random.Random(seed)
    templates = [
        '价格一般，能用就行',
        '服务普通，但也能接受',
        '味道马马虎虎，不算好也不算差',
        '快递慢了点，不过东西收到了',
        '包装一般般，没有惊喜也没有失望',
        '体验平平常常，和平时一样',
        '质量普通，不好不坏',
        '态度一般，但事情办完了',
        '环境还行吧，没什么感觉',
        '等的时间长一点，结果正常',
    ]
    samples = list(templates)
    for _ in range(n * 2):
        if len(samples) >= n:
            break
        tpl = rng.choice([
            '这次体验一般般，情绪稳定',
            '东西普普通通，就这样吧',
            '服务正常，没什么特别的',
            '价格平常，按部就班买了',
            '结果一般，无喜无悲',
        ])
        sent = clean_weibo_text(tpl, label=1)
        if len(sent) >= 4 and not _has_forbidden(sent):
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def generate_weak_negative_samples(n, seed=42):
    """弱消极：无强词，针对消极被误判为中性"""
    rng = random.Random(seed)
    templates = [
        '有点失望，不太符合预期',
        '体验一般偏下，略有不满',
        '服务态度不太好，希望改进',
        '质量略差，使用感不佳',
        '价格偏高，感觉不太值',
        '等待时间太长，略让人烦',
        '包装简陋，印象不太好',
        '回复慢，体验不太好',
        '和描述有差距，略失望',
        '不如预期，不太满意',
    ]
    samples = list(templates)
    objs = ['这次服务', '产品', '物流', '体验', '味道']
    for _ in range(n * 2):
        if len(samples) >= n:
            break
        obj = rng.choice(objs)
        tpl = rng.choice([
            f'{obj}不太好，略有不满',
            f'{obj}略差，希望改进',
            f'{obj}不太符合预期',
            f'{obj}一般偏下，有点失望',
        ])
        sent = clean_weibo_text(tpl, label=2)
        if len(sent) >= 4:
            samples.append(sent)
    return list(dict.fromkeys(samples))[:n]


def _clean_labeled_dataframe(df, min_len=2):
    """按标签清洗原始 DataFrame，返回清洗后 DataFrame 及统计"""
    cleaned_rows = []
    dropped_short = 0
    emoji_kept = 0
    emoji_removed = 0
    for label in (0, 1, 2):
        for text in df.loc[df['label'] == label, 'text'].astype(str):
            raw_tags = RE_EMOJI_TAG.findall(text)
            cleaned = clean_weibo_text(text, label=label)
            kept_tags = RE_EMOJI_TAG.findall(cleaned)
            emoji_kept += len(kept_tags)
            emoji_removed += max(0, len(raw_tags) - len(kept_tags))
            if len(cleaned) < min_len:
                dropped_short += 1
                continue
            cleaned_rows.append({'label': label, 'text': cleaned})
    return pd.DataFrame(cleaned_rows), dropped_short, emoji_kept, emoji_removed


LABELED_CLEAN_CACHE = './data/labeled_clean_cache.pkl'


def _get_labeled_clean_df(raw_path, min_len=2, use_cache=True):
    """标签感知清洗，结果缓存避免重复耗时"""
    if use_cache and os.path.exists(LABELED_CLEAN_CACHE):
        print(f'加载清洗缓存: {LABELED_CLEAN_CACHE}')
        return pd.read_pickle(LABELED_CLEAN_CACHE)
    print('正在清洗原始文本（首次运行会较慢，结果将缓存）...')
    df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
    df = df.dropna(subset=['label', 'text'])
    df = df[df['label'].isin([0, 1, 2])].copy()
    df['label'] = df['label'].astype(int)
    df_clean, _, _, _ = _clean_labeled_dataframe(df, min_len)
    df_clean.to_pickle(LABELED_CLEAN_CACHE)
    print(f'清洗缓存已保存: {len(df_clean)} 条')
    return df_clean


def build_clean_dataset(
    raw_path='./data/weibo_3class.csv',
    clean_path='./data/weibo_3class_clean.csv',
    stats_path='./results/preprocess_stats.json',
    min_len=2,
):
    """v1 兼容：全删表情（保留旧接口）"""
    os.makedirs(os.path.dirname(clean_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)
    df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
    df = df.dropna(subset=['label', 'text'])
    df = df[df['label'].isin([0, 1, 2])].copy()
    df['label'] = df['label'].astype(int)
    df['text'] = df['text'].apply(lambda t: clean_weibo_text(str(t), label=None))
    df['text'] = df['text'].apply(lambda t: RE_EMOJI_TAG.sub('', t))
    df = df[df['text'].str.len() >= min_len]
    df[['label', 'text']].to_csv(clean_path, header=False, index=False)
    return {'after': {'total': len(df)}}


def build_balanced_dataset(
    raw_path='./data/weibo_3class.csv',
    output_path='./data/weibo_3class_balanced.csv',
    stats_path='./results/preprocess_v2_stats.json',
    neutral_supplement=6000,
    min_len=2,
    seed=42,
):
    """
    v2 完整流程：标签感知清洗 + 中性补充 + 三类均衡下采样
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)
    rng = random.Random(seed)

    df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
    df = df.dropna(subset=['label', 'text'])
    df = df[df['label'].isin([0, 1, 2])].copy()
    df['label'] = df['label'].astype(int)

    before_stats = analyze_text_features(df['text'])
    before_dist = df['label'].value_counts().sort_index().to_dict()

    # 标签感知清洗
    records = []
    dropped_short = 0
    emoji_kept = 0
    emoji_removed = 0
    for _, row in df.iterrows():
        raw = str(row['text'])
        label = int(row['label'])
        raw_tags = RE_EMOJI_TAG.findall(raw)
        cleaned = clean_weibo_text(raw, label=label)
        kept_tags = RE_EMOJI_TAG.findall(cleaned)
        emoji_kept += len(kept_tags)
        emoji_removed += max(0, len(raw_tags) - len(kept_tags))
        if len(cleaned) < min_len:
            dropped_short += 1
            continue
        records.append({'label': label, 'text': cleaned})

    df_clean = pd.DataFrame(records)
    after_clean_dist = df_clean['label'].value_counts().sort_index().to_dict()

    # 补充中性样本
    neu_sup = generate_neutral_samples(neutral_supplement, seed=seed)
    df_sup = pd.DataFrame({'label': 1, 'text': neu_sup})
    df_neu = pd.concat([
        df_clean[df_clean['label'] == 1],
        df_sup,
    ], ignore_index=True)
    target_n = len(df_neu)

    # 下采样积极/消极至与中性基本一致
    df_pos = df_clean[df_clean['label'] == 0]
    df_neg = df_clean[df_clean['label'] == 2]
    df_pos_sampled = df_pos.sample(n=min(target_n, len(df_pos)), random_state=seed)
    df_neg_sampled = df_neg.sample(n=min(target_n, len(df_neg)), random_state=seed)

    df_final = pd.concat([df_pos_sampled, df_neu, df_neg_sampled], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)

    after_stats = analyze_text_features(df_final['text'])
    final_dist = df_final['label'].value_counts().sort_index().to_dict()

    df_final[['label', 'text']].to_csv(output_path, header=False, index=False)

    stats = {
        'version': 'v2_balanced',
        'raw_path': raw_path,
        'output_path': output_path,
        'neutral_supplement': neutral_supplement,
        'before': {
            'total': int(len(df)),
            'label_dist': {str(k): v for k, v in before_dist.items()},
            'features': before_stats,
        },
        'after_clean': {
            'total': int(len(df_clean)),
            'label_dist': {str(k): v for k, v in after_clean_dist.items()},
            'dropped_too_short': dropped_short,
            'emoji_kept': emoji_kept,
            'emoji_removed': emoji_removed,
        },
        'after_balance': {
            'total': int(len(df_final)),
            'label_dist': {str(k): v for k, v in final_dist.items()},
            'features': after_stats,
            'pos_downsampled_from': int(len(df_pos)),
            'neg_downsampled_from': int(len(df_neg)),
            'neutral_original': int(len(df_clean[df_clean['label'] == 1])),
            'neutral_supplemented': len(neu_sup),
        },
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def build_balanced_dataset_v3(
    raw_path='./data/weibo_3class.csv',
    output_path='./data/weibo_3class_balanced_v3.csv',
    stats_path='./results/preprocess_v3_stats.json',
    neutral_supplement=8000,
    strong_pos_supplement=2500,
    strong_neg_supplement=2500,
    min_len=2,
    seed=42,
):
    """
    v3：v2 基础上
    - 更清晰的中性补充（禁用情感词）
    - 强积极/强消极补充（含一致表情）
    - 三类均衡下采样
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)

    df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
    df = df.dropna(subset=['label', 'text'])
    df = df[df['label'].isin([0, 1, 2])].copy()
    df['label'] = df['label'].astype(int)
    before_stats = analyze_text_features(df['text'])
    before_dist = df['label'].value_counts().sort_index().to_dict()

    df_clean = _get_labeled_clean_df(raw_path, min_len)
    dropped_short = int(len(df) - len(df_clean))
    emoji_kept = emoji_removed = 0

    print('正在生成补充样本...')
    neu_sup = generate_neutral_samples(neutral_supplement, seed=seed, clear_style=True)
    pos_sup = generate_strong_positive_samples(strong_pos_supplement, seed=seed + 1)
    neg_sup = generate_strong_negative_samples(strong_neg_supplement, seed=seed + 2)

    df_neu = pd.concat([
        df_clean[df_clean['label'] == 1],
        pd.DataFrame({'label': 1, 'text': neu_sup}),
    ], ignore_index=True)
    target_n = len(df_neu)

    df_pos_pool = pd.concat([
        df_clean[df_clean['label'] == 0],
        pd.DataFrame({'label': 0, 'text': pos_sup}),
    ], ignore_index=True)
    df_neg_pool = pd.concat([
        df_clean[df_clean['label'] == 2],
        pd.DataFrame({'label': 2, 'text': neg_sup}),
    ], ignore_index=True)

    df_pos_sampled = df_pos_pool.sample(n=min(target_n, len(df_pos_pool)), random_state=seed)
    df_neg_sampled = df_neg_pool.sample(n=min(target_n, len(df_neg_pool)), random_state=seed)

    df_final = pd.concat([df_pos_sampled, df_neu, df_neg_sampled], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_final[['label', 'text']].to_csv(output_path, header=False, index=False)

    after_stats = analyze_text_features(df_final['text'])
    final_dist = df_final['label'].value_counts().sort_index().to_dict()

    stats = {
        'version': 'v3_balanced_strong',
        'output_path': output_path,
        'neutral_supplement': neutral_supplement,
        'strong_pos_supplement': strong_pos_supplement,
        'strong_neg_supplement': strong_neg_supplement,
        'before': {'total': int(len(df)), 'label_dist': {str(k): v for k, v in before_dist.items()}, 'features': before_stats},
        'after_balance': {
            'total': int(len(df_final)),
            'label_dist': {str(k): v for k, v in final_dist.items()},
            'features': after_stats,
            'neutral_original': int(len(df_clean[df_clean['label'] == 1])),
            'neutral_supplemented': len(neu_sup),
            'strong_pos_added': len(pos_sup),
            'strong_neg_added': len(neg_sup),
            'dropped_too_short': dropped_short,
            'emoji_kept': emoji_kept,
            'emoji_removed': emoji_removed,
        },
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def build_balanced_dataset_v4(
    raw_path='./data/weibo_3class.csv',
    output_path='./data/weibo_3class_balanced_v4.csv',
    stats_path='./results/preprocess_v4_stats.json',
    neutral_supplement=8000,
    strong_pos_supplement=2500,
    strong_neg_supplement=2500,
    plain_pos_supplement=1000,
    plain_neg_supplement=1000,
    min_len=2,
    seed=42,
):
    """
    v4：v3 + 无表情强情感样本各 1000 条（A）
    配合模型内情感词典加权（B，见 sentiment_lexicon.py）
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)

    df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
    df = df.dropna(subset=['label', 'text'])
    df = df[df['label'].isin([0, 1, 2])].copy()
    df['label'] = df['label'].astype(int)
    before_stats = analyze_text_features(df['text'])
    before_dist = df['label'].value_counts().sort_index().to_dict()

    df_clean = _get_labeled_clean_df(raw_path, min_len)
    dropped_short = int(len(df) - len(df_clean))

    print('正在生成 v4 补充样本...')
    neu_sup = generate_neutral_samples(neutral_supplement, seed=seed, clear_style=True)
    pos_emo = generate_strong_positive_samples(strong_pos_supplement, seed=seed + 1)
    neg_emo = generate_strong_negative_samples(strong_neg_supplement, seed=seed + 2)
    pos_plain = generate_strong_positive_plain(plain_pos_supplement, seed=seed + 3)
    neg_plain = generate_strong_negative_plain(plain_neg_supplement, seed=seed + 4)

    df_neu = pd.concat([
        df_clean[df_clean['label'] == 1],
        pd.DataFrame({'label': 1, 'text': neu_sup}),
    ], ignore_index=True)
    target_n = len(df_neu)

    df_pos_pool = pd.concat([
        df_clean[df_clean['label'] == 0],
        pd.DataFrame({'label': 0, 'text': pos_emo + pos_plain}),
    ], ignore_index=True)
    df_neg_pool = pd.concat([
        df_clean[df_clean['label'] == 2],
        pd.DataFrame({'label': 2, 'text': neg_emo + neg_plain}),
    ], ignore_index=True)

    df_pos_sampled = df_pos_pool.sample(n=min(target_n, len(df_pos_pool)), random_state=seed)
    df_neg_sampled = df_neg_pool.sample(n=min(target_n, len(df_neg_pool)), random_state=seed)

    df_final = pd.concat([df_pos_sampled, df_neu, df_neg_sampled], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_final[['label', 'text']].to_csv(output_path, header=False, index=False)

    after_stats = analyze_text_features(df_final['text'])
    final_dist = df_final['label'].value_counts().sort_index().to_dict()

    stats = {
        'version': 'v4_plain_strong_lexicon',
        'output_path': output_path,
        'neutral_supplement': neutral_supplement,
        'strong_pos_supplement': strong_pos_supplement,
        'strong_neg_supplement': strong_neg_supplement,
        'plain_pos_supplement': plain_pos_supplement,
        'plain_neg_supplement': plain_neg_supplement,
        'before': {'total': int(len(df)), 'label_dist': {str(k): v for k, v in before_dist.items()}, 'features': before_stats},
        'after_balance': {
            'total': int(len(df_final)),
            'label_dist': {str(k): v for k, v in final_dist.items()},
            'features': after_stats,
            'neutral_original': int(len(df_clean[df_clean['label'] == 1])),
            'neutral_supplemented': len(neu_sup),
            'strong_pos_emoji': len(pos_emo),
            'strong_neg_emoji': len(neg_emo),
            'strong_pos_plain': len(pos_plain),
            'strong_neg_plain': len(neg_plain),
            'dropped_too_short': dropped_short,
        },
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


def build_balanced_dataset_v5(
    raw_path='./data/weibo_3class.csv',
    output_path='./data/weibo_3class_balanced_v5.csv',
    stats_path='./results/preprocess_v5_stats.json',
    neutral_supplement=8000,
    strong_pos_supplement=2500,
    strong_neg_supplement=2500,
    plain_pos_supplement=1000,
    plain_neg_supplement=1000,
    weak_pos_supplement=800,
    borderline_neu_supplement=800,
    weak_neg_supplement=800,
    min_len=2,
    seed=42,
):
    """
    v5：v4 + 错例定向样本（弱积极/边界中性/弱消极各 800）
    配合 pad_size 加长与词典 v2
    """
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    os.makedirs(os.path.dirname(stats_path) or '.', exist_ok=True)

    df = pd.read_csv(raw_path, header=None, names=['label', 'text'])
    df = df.dropna(subset=['label', 'text'])
    df = df[df['label'].isin([0, 1, 2])].copy()
    df['label'] = df['label'].astype(int)
    before_stats = analyze_text_features(df['text'])
    before_dist = df['label'].value_counts().sort_index().to_dict()

    df_clean = _get_labeled_clean_df(raw_path, min_len)
    dropped_short = int(len(df) - len(df_clean))

    print('正在生成 v5 补充样本（含错例定向）...')
    neu_sup = generate_neutral_samples(neutral_supplement, seed=seed, clear_style=True)
    border_neu = generate_borderline_neutral_samples(borderline_neu_supplement, seed=seed + 5)
    pos_emo = generate_strong_positive_samples(strong_pos_supplement, seed=seed + 1)
    neg_emo = generate_strong_negative_samples(strong_neg_supplement, seed=seed + 2)
    pos_plain = generate_strong_positive_plain(plain_pos_supplement, seed=seed + 3)
    neg_plain = generate_strong_negative_plain(plain_neg_supplement, seed=seed + 4)
    pos_weak = generate_weak_positive_samples(weak_pos_supplement, seed=seed + 6)
    neg_weak = generate_weak_negative_samples(weak_neg_supplement, seed=seed + 7)

    df_neu = pd.concat([
        df_clean[df_clean['label'] == 1],
        pd.DataFrame({'label': 1, 'text': neu_sup + border_neu}),
    ], ignore_index=True)
    target_n = len(df_neu)

    df_pos_pool = pd.concat([
        df_clean[df_clean['label'] == 0],
        pd.DataFrame({'label': 0, 'text': pos_emo + pos_plain + pos_weak}),
    ], ignore_index=True)
    df_neg_pool = pd.concat([
        df_clean[df_clean['label'] == 2],
        pd.DataFrame({'label': 2, 'text': neg_emo + neg_plain + neg_weak}),
    ], ignore_index=True)

    df_pos_sampled = df_pos_pool.sample(n=min(target_n, len(df_pos_pool)), random_state=seed)
    df_neg_sampled = df_neg_pool.sample(n=min(target_n, len(df_neg_pool)), random_state=seed)

    df_final = pd.concat([df_pos_sampled, df_neu, df_neg_sampled], ignore_index=True)
    df_final = df_final.sample(frac=1, random_state=seed).reset_index(drop=True)
    df_final[['label', 'text']].to_csv(output_path, header=False, index=False)

    after_stats = analyze_text_features(df_final['text'])
    final_dist = df_final['label'].value_counts().sort_index().to_dict()

    stats = {
        'version': 'v5_error_targeted',
        'output_path': output_path,
        'weak_pos_supplement': weak_pos_supplement,
        'borderline_neu_supplement': borderline_neu_supplement,
        'weak_neg_supplement': weak_neg_supplement,
        'before': {'total': int(len(df)), 'label_dist': {str(k): v for k, v in before_dist.items()}, 'features': before_stats},
        'after_balance': {
            'total': int(len(df_final)),
            'label_dist': {str(k): v for k, v in final_dist.items()},
            'features': after_stats,
            'neutral_supplemented': len(neu_sup) + len(border_neu),
            'positive_supplemented': len(pos_emo) + len(pos_plain) + len(pos_weak),
            'negative_supplemented': len(neg_emo) + len(neg_plain) + len(neg_weak),
            'dropped_too_short': dropped_short,
        },
    }
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    return stats


if __name__ == '__main__':
    stats = build_balanced_dataset_v5()
    print('v5 均衡数据集已生成:', stats['after_balance']['total'])
    print('类别分布:', stats['after_balance']['label_dist'])
    print('统计已写入 results/preprocess_v5_stats.json')
