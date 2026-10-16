# -*- coding: utf-8 -*-
"""
人员/车辆检测模块
优先使用 ultralytics YOLO(可选依赖); 缺失时回退到 OpenCV 轮廓启发式检测,
保证软件在纯 CPU 环境开箱即用。
"""
import cv2
import numpy as np

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except Exception:
    YOLO_AVAILABLE = False


class ObjectDetector:
    def __init__(self, model_name: str = "yolov8n.pt"):
        self.model = None
        if YOLO_AVAILABLE:
            try:
                self.model = YOLO(model_name)
            except Exception:
                self.model = None

    def detect(self, frame: np.ndarray):
        """返回目标列表: [{cls, conf, box(x1,y1,x2,y2)}]"""
        if self.model is not None:
            return self._detect_yolo(frame)
        return self._detect_contour(frame)

    # ---------- YOLO 路径 ----------
    def _detect_yolo(self, frame):
        results = self.model.predict(frame, verbose=False, classes=[0, 2, 3, 5])
        out = []
        for r in results:
            for b in r.boxes:
                cls_id = int(b.cls.item())
                name = {0: "person", 2: "car", 3: "motorcycle", 5: "bus"}.get(cls_id, "object")
                x1, y1, x2, y2 = [int(v) for v in b.xyxy[0].tolist()]
                out.append({"cls": name, "conf": float(b.conf.item()), "box": [x1, y1, x2, y2]})
        return out

    # ---------- OpenCV 轮廓启发式回退(适配模拟视频源) ----------
    def _detect_contour(self, frame):
        h, w = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 前景: 与背景(路面)差异较大的暗/亮目标
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        bg = cv2.medianBlur(blur, 51)
        diff = cv2.absdiff(blur, bg)
        _, th = cv2.threshold(diff, 22, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=2)

        out = []
        cnts, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x, y, bw, bh = cv2.boundingRect(c)
            area = bw * bh
            if area < 300 or y + bh < h * 0.3:
                continue
            # 排除录屏 UI 区域: 左侧 App 边栏与底部字幕/按钮条(非真实场景目标)
            if x + bw < int(w * 0.13) or y > int(h * 0.86):
                continue
            ar = bh / max(bw, 1)
            if 1.3 <= ar <= 3.5 and 300 <= area <= 0.02 * w * h:
                out.append({"cls": "person", "conf": 0.6, "box": [x, y, x + bw, y + bh]})
            elif 1.5 <= bw / max(bh, 1) <= 4 and 0.004 * w * h <= area <= 0.08 * w * h:
                out.append({"cls": "car", "conf": 0.55, "box": [x, y, x + bw, y + bh]})
        return out


def person_in_water(objs, water_mask):
    """判断哪些人员目标位于积水掩码内(脚部区域与掩码交集占比)。"""
    h, w = water_mask.shape[:2]
    hits = []
    for o in objs:
        if o["cls"] != "person":
            continue
        x1, y1, x2, y2 = o["box"]
        # 取目标下部 1/3(脚部)区域
        fy1 = int(y1 + (y2 - y1) * 0.66)
        roi = water_mask[fy1:y2, x1:x2]
        if roi.size == 0:
            continue
        ratio = float((roi > 0).mean())
        if ratio > 0.5:
            hits.append(o)
    return hits
