# 各轮优化中间产物（汇报用图）

## 历史图片是否保留？

**没有。** 训练脚本 `main.py` 每次只写入固定路径：

- `results/acc.png` / `results/loss.png`（训练过程曲线）
- `results/confusion_matrix.png`（最后一轮测试混淆矩阵）

每轮训练会**覆盖**上一版，且当前仓库里**已不存在任何 PNG**。  
`train_v5.log` 也只有一次失败报错，不含完整 epoch 曲线。

本目录图表由 `generate_milestone_figures.py` 根据 `optimization_log.md` 中**已记录的测试集指标与混淆矩阵**重新绘制。

## 文件说明

| 文件 | 用途 |
|------|------|
| `accuracy_macro_f1_trend.png` | 第 0～5 轮准确率 + macro-F1 折线（汇报主图） |
| `class_f1_trend.png` | 三类 F1 分轮变化，突出中性 F1 从 0.21→0.83 |
| `misclassification_by_round.png` | 均衡测试集上各类误判数柱状对比 |
| `confusion_matrix_grid_balanced_norm.png` | 第 2～5 轮归一化混淆矩阵拼图 |
| `confusion_matrix_grid_all_norm.png` | 含第 1 轮（注意测试集为原始分布） |
| `round1_v1_cm.png` … `round5_v5_cm.png` | 各轮单张混淆矩阵（计数） |
| `round*_cm_norm.png` | 各轮混淆矩阵（按真实类行归一化，便于跨轮对比） |
| `metrics_summary.json` | 各轮数值，供 PPT 制表或视频字幕 |

## 重新生成

```bash
python generate_milestone_figures.py
```

## 第 0 轮数据来源

摘自 `中文文本情感分析进展报告.pdf` §4.2：

- **验证集准确率约 74%**（10 轮训练、基础 BiLSTM + 预训练词向量）
- 报告**未记录**测试集准确率、macro-F1、分项 F1、混淆矩阵

第 1～5 轮为 `optimization_log.md` 中的**测试集**指标。  
趋势图底部有脚注：第 0 轮为验证集，其余为测试集（口径不同，对比时需在 PPT 中口头说明）。

## 缺失项

- **第 0 轮混淆矩阵 / F1**：报告与日志均未记录，图表中不展示
- **各轮训练 epoch 曲线**：未按版本存档；用各轮**训练完成后的测试准确率**体现工作量即可

## 待解决问题对照

见同目录 `pending_issues.md`（对照进展报告第五节与 v5 现状）。
