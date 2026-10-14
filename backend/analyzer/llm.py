# -*- coding: utf-8 -*-
"""
多模态大模型应急研判模块
- 在线模式: 配置环境变量 LLM_API_BASE / LLM_API_KEY / LLM_MODEL 后,
  将监控帧(base64) + 结构化风险指标发送给 OpenAI 兼容多模态模型
  (如 Qwen2.5-VL / GPT-4o), 生成事件描述与公安应急处置建议。
- 离线模式: 未配置时回退到内置预案模板, 保证软件开箱即用。
"""
import base64
import json
import os
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request

import cv2
import numpy as np

API_BASE = os.environ.get("LLM_API_BASE", "")   # 例: https://dashscope.aliyuncs.com/compatible-mode/v1
API_KEY = os.environ.get("LLM_API_KEY", "")
MODEL = os.environ.get("LLM_MODEL", "qwen-vl-max")

PROMPT = (
    "你是城市洪涝监测与公安应急处置专家。请分析这张监控画面与结构化指标, "
    "严格只输出一个 JSON 对象(不要其他文字): "
    '{"summary": "一句话事件描述", "advice": "3条公安应急处置建议(封控/疏散/排水/交通管制等, 用分号分隔)"}。'
    "若指标含\u201c省内内涝易发区背景\u201d(该省历史易涝县域/村镇简介), 请结合其灾因特点给出针对性建议。"
    "指标: "
)

TEMPLATES = {
    1: "积水轻微, 维持常规视频巡查, 关注水情变化趋势。",
    2: "路面积水扩大, 建议通知属地交警加强该路段巡查, 提醒车辆减速绕行。",
    3: "积水达到警戒水平, 建议: 1) 交警对该路段实施临时交通管制; 2) 通知排水部门现场强排; 3) 通过情报板发布绕行提示。",
    4: "检测到人员/车辆涉水, 情况紧急, 建议: 1) 立即指令就近巡警与救援力量到场; 2) 封控涉事路段双向交通; 3) 组织涉水人员转移并核查被困车辆; 4) 联动排水、消防、医疗开展处置。",
}


def analyze(frame: np.ndarray, metrics: dict) -> dict:
    """返回 {"summary":..., "advice":..., "source": "llm"|"template"}"""
    if API_BASE and API_KEY:
        try:
            return _call_llm(frame, metrics)
        except Exception as e:  # 在线失败自动回退
            print("[llm] 在线研判失败, 回退模板:", e)
    return {
        "summary": f"{metrics['name']}事件: 积水占比 {metrics['water_ratio']*100:.1f}%, "
                   f"涉水人员 {metrics['persons_in_water']} 人, 涉水车辆 {metrics['cars_in_water']} 辆。",
        "advice": TEMPLATES[metrics["level"]],
        "source": "template",
    }


_CALL_LOCK = threading.Lock()  # 串行化推理: 小模型并发请求易触发异常


def _unload_model():
    """调用 ollama stop 卸载模型: 用于从 Ollama 防重复看门狗的异常状态中自愈。"""
    exe = shutil.which("ollama") or os.path.join(
        os.environ.get("LOCALAPPDATA", ""), "Programs", "Ollama", "ollama.exe")
    if not os.path.exists(exe):
        return
    try:
        subprocess.run([exe, "stop", MODEL], capture_output=True, timeout=30)
    except Exception:
        pass


def _call_llm(frame: np.ndarray, metrics: dict) -> dict:
    ok, buf = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
    if not ok:
        raise RuntimeError("frame encode failed")
    b64 = base64.b64encode(buf.tobytes()).decode()
    body = {
        "model": MODEL,
        "max_tokens": 512,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": PROMPT + json.dumps(metrics, ensure_ascii=False)},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
            ],
        }],
    }
    req = urllib.request.Request(
        API_BASE.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"},
    )
    with _CALL_LOCK:
        for attempt in (1, 2):
            try:
                with urllib.request.urlopen(req, timeout=120) as resp:  # 本地 GPU 排队推理需较长时间
                    data = json.load(resp)
                break
            except urllib.error.HTTPError as e:
                # Ollama 防重复看门狗误杀后模型会持续 500, 自动卸载重载后重试一次
                if attempt == 1 and e.code == 500:
                    print("[llm] 推理服务异常(500), 自动重载模型重试…")
                    _unload_model()
                    time.sleep(2)
                    continue
                raise
    text = data["choices"][0]["message"]["content"]
    try:
        parsed = json.loads(text[text.index("{"): text.rindex("}") + 1])
        return {"summary": parsed.get("summary", text), "advice": parsed.get("advice", ""), "source": "llm"}
    except Exception:
        return {"summary": text[:120], "advice": text, "source": "llm"}
