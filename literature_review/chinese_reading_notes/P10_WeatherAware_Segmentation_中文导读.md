# P10 Weather-aware Segmentation 中文导读

## 基本信息

- 英文标题：Exploring Weather-aware Aggregation and Adaptation for Semantic Segmentation under Adverse Conditions
- 中文标题：面向恶劣天气语义分割的天气感知聚合与适应
- 会议年份：ICCV 2025
- 本地 PDF：`literature_review/papers/2025_ICCV_WeatherAware_Segmentation.pdf`

## 一句话总结

这篇论文研究恶劣天气下的语义分割，对停车位区域、车道区域或地面标线分割有参考价值。

## 这篇论文解决什么问题

雨、雾、雪、夜间等天气条件会明显影响语义分割。停车场项目如果需要自动识别停车线、车位区域、道路区域或可行驶区域，那么恶劣天气下的分割鲁棒性就很重要。

## 方法思路

论文强调 weather-aware，也就是模型需要意识到不同天气条件下图像特征的变化，并进行适应或聚合。它不是简单把所有天气混在一起训练，而是考虑天气因素对特征分布的影响。

## 阅读重点

- 看它如何定义 adverse weather segmentation。
- 看 weather-aware aggregation/adaptation 的具体含义。
- 看不同天气条件下的分割性能。
- 关注是否能迁移到停车位线或地面区域分割。

## 对停车场项目的启发

如果停车位不是人工标注多边形，而是希望系统自动识别车位线，那么这篇很有参考价值。雨天、夜间和反光地面会让车位线变得不清晰，weather-aware segmentation 可以作为未来扩展方向。

## 局限性

- 任务是语义分割，不是目标检测或跟踪。
- 不直接判断车位是否被占用。
- 对项目主线来说是辅助模块。

## 推荐精读程度

可选阅读。若后续做自动车位线识别，建议精读。

