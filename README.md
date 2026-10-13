# Flood-Watch 城市洪涝监测与公安应急处置平台

基于多模态智能模型的城市洪涝监测与公安应急处置研究配套软件（《视频分析技术》课程大作业）。

## 开源基线（大作业流程第 1-2 步）

- **论文**: Urban Waterlogging Detection: A Challenging Benchmark and Large-Small Model Co-Adapter (ECCV 2024)
- **代码**: https://github.com/zhang-chenxu/LSM-Adapter （SAM 大模型 + 小模型协同提示适配, UW-Bench 数据集）
- **辅助**: https://github.com/facebookresearch/sam2 、https://github.com/ultralytics/ultralytics

## 创新点（第 6-9 步）

LSM-Adapter 仅做单帧积水分割；本平台在其基础上扩展为 **多模态时序监测 + 公安应急处置闭环**：

1. 视觉大/小模型协同分割（OpenCV 轻量实现，可插拔 SAM/LSM-Adapter 复现论文精度）
2. 人员/车辆目标检测（YOLO 可选，OpenCV 轮廓启发式回退），融合判断"涉水风险"
3. 时序风险评估引擎（面积占比 + 上涨速率 + 涉水目标 → 四级预警）
4. 多模态大模型应急研判（Qwen2.5-VL 等 OpenAI 兼容接口，离线自动回退预案模板）
5. 告警事件管理与处置受理（公安指挥闭环）
6. 真实监控视频接入：大屏上可直接上传监控录像（mp4/avi/mov/mkv 等），
   热切换对应城市分析线程为真实视频源，AI 分析链照常运行，一键恢复模拟画面

## 快速开始

```bash
pip install -r requirements.txt
python run.py                # 内置模拟内涝视频源, 开箱即用
python run.py --source 0     # 摄像头
python run.py --source x.mp4 # 视频文件/UW-Bench 样本
```

浏览器打开 http://127.0.0.1:8000

可选环境变量（在线多模态研判）：`LLM_API_BASE` / `LLM_API_KEY` / `LLM_MODEL`。

## 目录结构

```
backend/            FastAPI 服务与分析流水线
  analyzer/         积水分割 / 目标检测 / 风险评估 / 多模态研判
  video_source.py   模拟/摄像头/文件视频源
  events.py         告警事件持久化
frontend/index.html 监控大屏 (Tailwind, 响应式)
```
