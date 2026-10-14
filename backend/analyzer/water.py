# -*- coding: utf-8 -*-
"""
积水区域分割模块
基线思路来自 LSM-Adapter (ECCV 2024, 城市内涝检测大-小模型协同范式):
  https://github.com/zhang-chenxu/LSM-Adapter
本模块提供轻量 OpenCV 实现(可离线运行), 并预留 SAM/LSM-Adapter 大模型接口,
GPU 环境下可无缝替换为论文中的分割模型以复现精度。
"""
import cv2
import numpy as np

try:  # 可选: SAM / LSM-Adapter 大模型后端
    from segment_anything import sam_model_registry, SamPredictor  # noqa: F401
    SAM_AVAILABLE = True
except Exception:
    SAM_AVAILABLE = False


class WaterSegmenter:
    """城市道路积水分割器: OpenCV 颜色+形态学方法, 可插拔 SAM 大模型。"""

    def __init__(self, use_large_model: bool = False, checkpoint: str | None = None):
        # use_large_model=True 且环境具备 SAM 权重时走大模型路径(论文基线)
        self.use_large_model = use_large_model and SAM_AVAILABLE
        self.predictor = None
        if self.use_large_model and checkpoint:
            sam = sam_model_registry["vit_h"](checkpoint=checkpoint)
            self.predictor = SamPredictor(sam)

    def segment(self, frame: np.ndarray):
        """返回 (积水掩码, 积水面积占比, 最大连通域外接框或 None)。"""
        if self.use_large_model and self.predictor is not None:
            return self._segment_sam(frame)
        return self._segment_opencv(frame)

    # ---------- OpenCV 轻量实现(默认, 离线可跑) ----------
    # 互补三分支自适应融合(移植自 Prithvi-CAFE, M2FAF 思想的轻量近似):
    #   分支1 颜色: 蓝色调 HSV 主导(判别力强, 作主分支固定先验)
    #   分支2 纹理: 局部梯度低 -> 水面镜面比路面/建筑平滑(颜色分支漏检的灰亮水面)
    #   分支3 亮度: 暗于路面中位数 -> 积水吸光偏暗
    #   纹理/亮度分支合成"镜面互补候选", 其动态权重按候选在颜色区内外的
    #   区分度(均值差)自适应设定: 区分度高才参与融合补召回, 区分度低自动
    #   退化为原颜色基线, 不会引入建筑/阴影误检。
    def _segment_opencv(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        b = frame[:, :, 0].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)
        # 分支1(颜色): 蓝色调主导 (B 明显大于 R), 排除灰色路面/建筑误检
        blue_dom = ((b - r) > 18).astype(np.uint8) * 255
        m1 = cv2.inRange(hsv, (90, 25, 40), (135, 255, 255))    # 蓝色调水面
        m2 = cv2.inRange(hsv, (85, 10, 90), (140, 60, 235))     # 低饱和灰蓝反光水面
        color = cv2.bitwise_or(cv2.bitwise_and(m1, blue_dom), cv2.bitwise_and(m2, blue_dom))

        # 分支2(纹理): 局部梯度平滑度(0~1), 水面镜面梯度远低于路面纹理
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (15, 15), 0)
        smooth = np.clip(1.0 - grad / 45.0, 0.0, 1.0)
        # 分支3(亮度): 相对路面中位数的偏暗程度(0~1)
        med = float(np.median(gray[int(h * 0.35):, :]))
        dark = np.clip((med - gray.astype(np.float32)) / 50.0 + 0.5, 0.0, 1.0)
        # 互补候选: 纹理+亮度双分支同时高置信(颜色分支易漏的灰镜面水面)
        mirror = smooth * dark

        # 自适应加权: 候选在"颜色已检出区"与"路面ROI其余区"的均值差越大,
        # 说明该候选特征与水面越同步, 权重越高; 否则权重趋 0(退回颜色基线)
        c_in = color > 0
        roi = np.zeros((h, w), np.uint8)
        roi[int(h * 0.35):, :] = 1
        outside = (roi > 0) & (~c_in)
        if c_in.any() and outside.any():
            sep = float(mirror[c_in].mean()) - float(mirror[outside].mean())
        else:
            sep = 0.0
        w_mirror = float(np.clip(sep * 2.5, 0.0, 0.9))

        score = color.astype(np.float32) / 255.0 + w_mirror * mirror
        mask = (score >= 0.6).astype(np.uint8) * 255
        # 形态学: 闭运算填补空洞, 开运算去除噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 只保留下半画面(路面区域), 排除天空反光误检
        mask = cv2.bitwise_and(mask, roi * 255)

        ratio = float(mask.sum()) / 255.0 / (h * w)
        box = self._largest_box(mask)
        return mask, ratio, box

    # ---------- SAM 大模型接口(论文基线, 需 GPU/权重) ----------
    def _segment_sam(self, frame: np.ndarray):
        self.predictor.set_image(frame)
        # 以画面中心下半区点提示, 模拟小模型引导大模型的 prompt 生成
        h, w = frame.shape[:2]
        pts = np.array([[w // 2, int(h * 0.8)]])
        masks, _, _ = self.predictor.predict(point_coords=pts, point_labels=np.ones(len(pts), np.int32))
        mask = (masks[0].astype(np.uint8)) * 255
        ratio = float(mask.sum()) / 255.0 / (h * w)
        return mask, ratio, self._largest_box(mask)

    @staticmethod
    def _largest_box(mask: np.ndarray):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        c = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(c) < 500:
            return None
        return cv2.boundingRect(c)
