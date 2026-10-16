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
    # 四分支互补融合(轻量近似 Prithvi-CAFE M2FAF 思想), 针对真实监控视频重调:
    #   分支1 颜色: 蓝色调 HSV 主导 —— 命中模拟帧水面/天空倒影水面(高精度)。
    #   分支2 镜面: 真实内涝水面多为灰绿/灰蓝(反射树冠与阴天天空), 单靠蓝色调
    #     会大面积漏检并误检白色车身。以"绿→蓝色相 + 中低饱和 + 中等亮度 +
    #     局部平滑(镜面) + 非纯色(有波纹亮度起伏)"联合门控, 排除树木(高纹理)、
    #     车身(高亮或饱和)、UI 面板(纯色平坦)。
    #   分支3 暗镜: 暗于路面中位数且平滑的吸光水面。
    #   分支4 浊水: 航拍/泥水的黄褐色调水面, 靠强波纹起伏与平坦路面区分。
    #   最后按连通域面积过滤: 真实积水连片成面, 车/牌/人均为碎块, 直接剔除。
    def _segment_opencv(self, frame: np.ndarray):
        h, w = frame.shape[:2]
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        g8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = g8.astype(np.float32)
        b = frame[:, :, 0].astype(np.int16)
        gch = frame[:, :, 1].astype(np.int16)
        r = frame[:, :, 2].astype(np.int16)
        hue = hsv[:, :, 0].astype(np.int16)
        sat = hsv[:, :, 1].astype(np.int16)
        val = hsv[:, :, 2].astype(np.int16)
        # 路面 ROI: 只保留下半画面, 排除天空/远景建筑
        roi = np.zeros((h, w), bool)
        roi[int(h * 0.35):, :] = True

        # 纹理平滑度: 水面镜面局部梯度远低于树冠/沥青颗粒/车体边缘
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = cv2.GaussianBlur(cv2.magnitude(gx, gy), (15, 15), 0)
        smooth = np.clip(1.0 - grad / 45.0, 0.0, 1.0)
        # 局部亮度起伏: 水面有波纹/反光(specular), 黑边/UI 面板是纯色平坦
        mean = cv2.boxFilter(gray, cv2.CV_32F, (15, 15))
        sq = cv2.boxFilter(gray * gray, cv2.CV_32F, (15, 15))
        lstd = np.sqrt(np.clip(sq - mean * mean, 0, None))

        # 分支1(颜色)双路: 实测模拟帧水面为"高饱和强蓝镜面"(S≈96, b-r≈31,
        # smooth≈0.63, lstd≈8), 真实误检源为"低饱和蓝物体"(幕墙 b-r19.6/S77、
        # 墙裙 S51、指路牌 b-r61 但 smooth0.03)。两路门控互斥:
        #   m1 极高饱和强蓝 + 高镜面平滑 → 模拟水面(S96/smooth0.63);
        #     排除玻璃幕墙(S77/smooth0.55)/指路牌(smooth0.03)/墙体
        #   m2 低饱和灰蓝 + 中等亮度 → 真实灰蓝水面; 排除墙体(V172)/墙裙(S51)/幕墙(S77)
        m1 = (cv2.inRange(hsv, (90, 85, 40), (135, 255, 255)) > 0) & \
             (smooth > 0.58) & ((b - r) > 25) & ((b - r) <= 50)
        m2 = (cv2.inRange(hsv, (85, 10, 85), (140, 45, 150)) > 0) & \
             ((b - r) > 4) & ((b - r) <= 25)
        color = m1 | m2

        # 分支2(真实灰绿/灰蓝镜面水面): 绿-蓝色相 + 中低饱和 + 中等亮度 + 平滑 + 有起伏
        # (lstd>10 排除玻璃幕墙低频反射; g-r<=12 排除深绿玻璃/绿墙: 幕墙 g-r18/墙体15)
        mirror = (hue >= 35) & (hue <= 140) & (sat >= 6) & (sat <= 110) & \
                 (val >= 60) & (val <= 220) & (smooth > 0.5) & (lstd > 10) & \
                 ((gch - r) >= -6) & ((gch - r) <= 12)

        # 分支3(暗吸光水面): 绿蓝色相 + 强波纹起伏(lstd>12 排除玻璃幕墙);
        # g-r<=12 排除暗绿玻璃幕墙(g-r18)
        med = float(np.median(g8[int(h * 0.35):, :]))
        darkm = (gray < med - 10) & (smooth > 0.6) & (lstd > 12) & \
                ((gch - r) >= -2) & ((gch - r) <= 12) & (hue >= 35) & (hue <= 140)

        # 分支4(浑浊黄褐水面): 泥水/航拍褐水 hue 15-55, 靠波纹起伏与平坦路面区分;
        # g-r<=8 排除绿树冠(绿色 g-r 大), sat<=80 排除黄车/蓝牌等高饱和物体
        turbid = (hue >= 15) & (hue < 55) & (sat >= 8) & (sat <= 80) & \
                 (val >= 55) & (val <= 210) & (smooth > 0.25) & (lstd > 8) & \
                 ((gch - r) <= 8) & ((r - gch) >= -2)

        score = color.astype(np.float32) + 0.85 * mirror + 0.7 * darkm + 0.7 * turbid
        mask = ((score >= 0.6) & roi).astype(np.uint8) * 255
        # 形态学: 闭运算连接波纹间隙, 开运算去椒盐
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        # 连通域面积过滤(膨胀合并版): 真实积水常被车辆/路缘切成多个 2~3% 碎块,
        # 先适度膨胀把邻近碎块并成一个连通域再按原掩码面积过滤; 墙面/车身误检
        # 块四周孤立, 膨胀后仍 <3% 被剔除。
        dil = cv2.dilate(mask, kernel, iterations=2)
        n, lab, _, _ = cv2.connectedComponentsWithStats(dil, 8)
        min_area = h * w * 0.030
        keep = np.zeros((h, w), np.uint8)
        for i in range(1, n):
            comp = (lab == i) & (mask > 0)
            if int(comp.sum()) >= min_area:
                keep[comp] = 255
        mask = keep

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
