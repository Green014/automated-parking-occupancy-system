# 候选论文中文阅读指南

本目录为 `literature_review/papers/` 中 16 篇候选论文准备中文导读。导读不是逐字翻译，而是帮助快速理解论文的阅读版材料：每篇包含中文标题、核心问题、方法思路、阅读重点、局限性，以及它和停车场占用识别系统的关系。

## 推荐阅读顺序

### 第一组：先读，最贴近项目主线

1. `P01_YOLO-World_中文导读.md`
2. `P02_Multi-Object_Tracking_in_the_Dark_中文导读.md`
3. `P07_AODRaw_中文导读.md`
4. `P05_TrackTrack_中文导读.md`
5. `P06_MOTIP_中文导读.md`

这一组覆盖停车场系统最重要的能力：车辆检测、夜间跟踪、恶劣天气鲁棒性、在线跟踪和身份关联。

### 第二组：用于增强系统设计

6. `P03_DiffMOT_中文导读.md`
7. `P04_Generalizable_MOT_中文导读.md`
8. `P09_OW-OVD_中文导读.md`

这一组补充非线性运动、跨场景泛化和未知物体检测。

### 第三组：作为扩展方向阅读

9. `P10_WeatherAware_Segmentation_中文导读.md`
10. `P11_DeCLIP_中文导读.md`
11. `P12_SearchDet_中文导读.md`
12. `P13_OmniTrack_中文导读.md`
13. `P14_MITracker_中文导读.md`
14. `P08_MoME_中文导读.md`
15. `P15_EventCamera_Detection_中文导读.md`
16. `P16_Cant_Slow_Me_Down_中文导读.md`

这一组适合在项目需要扩展到语义分割、多摄像头、全景摄像头、特殊传感器或边缘部署时阅读。

## 建议阅读方法

每篇论文可以按下面顺序读：

1. 先读本目录中的中文导读。
2. 打开原始 PDF，只读 Abstract、Introduction 和 Figure 1。
3. 回到中文导读中的“阅读重点”，再读 Method 的相关小节。
4. 最后看 Experiments 中的数据集、评价指标和主要表格。

## 工具建议

- DeepL：适合整段翻译或上传 PDF 做文档翻译。
- Zotero + Translate for Zotero：适合边读 PDF 边划词、划句翻译。
- NotebookLM：适合上传多篇 PDF 后用中文提问，例如“这篇文章的方法和停车场项目有什么关系？”。
- SciSpace：适合做论文问答、公式解释和图表解释。

