# LSTM-Chinese-Sentiment-Analysis

《人工智能导论》课程项目：**基于 BiLSTM 的微博三分类情感分析系统**（积极 / 中性 / 消极）

## 项目简介

输入一条中文微博或评论，系统自动判断情感倾向，并输出三类概率。项目采用字符级 BiLSTM + 注意力机制 + 腾讯预训练词向量，结合情感词典加权与推理校准，经过五轮迭代优化。

**v5 均衡测试集指标**：准确率 **85.86%**，macro-F1 **0.8596**

## 小组成员

| 成员 | 分工 |
|------|------|
| 韩康宁 | 组长，进度统筹、测试与材料协调 |
| 梁瑞政 | 核心开发、模型训练、演示界面、汇报主讲 |
| 詹智旋 | 数据搜集整理、测试用例与 PPT 协助 |

## 项目结构

```
├── main.py                 # 模型训练
├── preprocess.py           # 数据清洗与样本构建
├── test.py                 # 单句推理
├── calibration.py          # 推理校准
├── sentiment_lexicon.py    # 情感词典
├── demo_app.py             # Gradio 可视化演示
├── run_eval.py             # 测试集评估
├── generate_milestone_figures.py  # 生成汇报图表
├── data/
│   ├── weibo_3class.csv              # 原始数据（约 11 万条）
│   ├── weibo_3class_balanced_v5.csv  # v5 均衡训练集
│   ├── vocab.pkl                     # 词表
│   ├── embedding_Tencent.npz         # 腾讯词向量
│   └── class.txt                     # 类别标签
├── saved_dict/
│   ├── lstm.ckpt           # 训练好的模型权重
│   └── calibration.pkl     # 校准参数
└── results/
    ├── optimization_log.md # 五轮优化完整记录
    ├── eval_latest.json    # 最新评估指标
    └── milestones/         # 趋势图、混淆矩阵等
```

## 环境配置

```bash
pip install -r requirements.txt
```

依赖：Python 3.9+、PyTorch、scikit-learn、pandas、gradio 等。

## 快速开始

### 1. 演示界面（推荐）

```bash
python demo_app.py
```

浏览器打开 `http://127.0.0.1:7860`，输入文本或选择示例即可查看预测结果。

### 2. 单句推理

```bash
python test.py
```

### 3. 重新训练

若 `data/weibo_3class_balanced_v5.csv` 不存在，训练时会自动从原始数据构建。

```bash
python main.py
```

### 4. 测试集评估

```bash
python run_eval.py
```

## 五轮优化概要

| 轮次 | 主要改动 | 准确率 |
|------|----------|--------|
| v1 | 全量噪声清洗（含删表情） | 58.15% |
| v2 | 中性补强 + 标签感知表情 + 三类均衡 | 79.44% |
| v3 | 清晰中性样本 + 强情感保护校准 | 80.02% |
| v4 | 情感词典 + 无表情强情感样本 | 80.57% |
| v5 | 错例定向补数据 + pad50 + 词典 v2 | **85.86%** |

详细实验记录见 [`results/optimization_log.md`](results/optimization_log.md)。

## 数据说明

- 原始语料：`data/weibo_3class.csv`（公开微博三分类数据集）
- 类别分布极不平衡：中性约 **7.8%**，是本项目主要优化难点
- 中间版本 CSV 可通过 `preprocess.py` 本地重建，未全部上传以减小仓库体积

## License

MIT License
