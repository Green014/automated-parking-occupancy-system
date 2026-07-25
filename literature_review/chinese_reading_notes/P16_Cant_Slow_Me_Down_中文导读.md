# P16 Can't Slow Me Down 中文导读

## 基本信息

- 英文标题：Can't Slow Me Down: Learning Robust and Hardware-Adaptive Object Detectors against Latency Attacks for Edge Devices
- 中文标题：不能拖慢我：面向边缘设备延迟攻击的鲁棒硬件自适应目标检测器
- 会议年份：CVPR 2025
- 本地 PDF：`literature_review/papers/2025_CVPR_Cant_Slow_Me_Down.pdf`

## 一句话总结

这篇论文关注边缘设备上的检测器延迟和硬件自适应，对停车场系统的实时部署有参考价值。

## 这篇论文解决什么问题

很多检测论文只报告准确率，但真实系统还要考虑运行速度和硬件限制。停车场系统如果部署在边缘设备或本地服务器上，需要持续处理视频流。延迟过高会导致车位状态更新不及时。

这篇论文还关注 latency attacks，即某些输入或条件可能显著拖慢模型推理。

## 方法思路

论文训练更鲁棒、更硬件自适应的目标检测器，使模型在边缘设备上面对延迟干扰时仍能保持较好的推理效率。

## 阅读重点

- 看论文如何定义 latency attack。
- 看 hardware-adaptive detector 的含义。
- 看准确率和延迟之间如何权衡。
- 思考停车场系统应如何做部署测试。

## 对停车场项目的启发

停车场系统最终需要实时运行，因此不能只看 mAP 或准确率。应该增加：

```text
FPS
end-to-end latency
单路/多路摄像头吞吐量
GPU/CPU 占用
夜间和雨天场景下的速度
```

这篇论文提醒我们把部署性能写进系统评价。

## 局限性

- 它不是停车场场景论文。
- 它关注检测器部署安全和延迟，不解决跟踪或占用判断。
- 对 literature review 主线来说是辅助文献。

## 推荐精读程度

可选阅读。适合写 real-time deployment 和 edge robustness。

