# -*- coding: utf-8 -*-
"""
单目参照物水深估计模块(移植自方案1: Robust and Fast Sensing of Urban Flood
Depth with Social Media Images Using Pre-Trained Large Models and Simple Edge
Training, Hydrology 2025)。
原论文范式: 大模型做图文相关性过滤 + YOLO-World 检测"标准高度参照物"(车辆/
路灯/花坛) + 轻量 MLP 回归水深 —— 核心思想是"大模型管理解, 微型网络管数值"。
本模块保留其最可复现的内核: 利用已有检测框(车辆/行人)与积水掩码求框内水线
浸没比例, 再乘以参照物标准物理高度得到厘米级水深估计, 无需训练任何网络。
"""
import numpy as np

# 参照物标准物理高度(cm): 取典型车型/行人体格标定值
STD_HEIGHT = {"car": 150, "motorcycle": 160, "bus": 300, "person": 170}


def estimate(objs, water_mask) -> dict:
    """返回 {"max_cm": 最大水深, "mean_cm": 均值水深, "refs": 参与估计的参照物数}。"""
    depths = []
    h, w = water_mask.shape[:2]
    for o in objs:
        std = STD_HEIGHT.get(o["cls"])
        if std is None:
            continue
        x1, y1, x2, y2 = o["box"]
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(w, int(x2)), min(h, int(y2))
        if x2 - x1 < 8 or y2 - y1 < 8:
            continue
        roi = water_mask[y1:y2, x1:x2] > 0
        if roi.mean() < 0.08:      # 目标基本不在积水中, 不参与估计
            continue
        # 水线: 从框底向上扫描, 连续"横向过半被淹"的行即为浸没段
        rows = roi.mean(axis=1)
        sub = 0
        for i in range(roi.shape[0] - 1, -1, -1):
            if rows[i] >= 0.5:
                sub = roi.shape[0] - i
            else:
                break
        frac = sub / roi.shape[0]
        if frac < 0.03:
            continue
        depths.append(round(std * frac))
    if not depths:
        return {"max_cm": 0, "mean_cm": 0, "refs": 0}
    return {"max_cm": max(depths), "mean_cm": round(sum(depths) / len(depths)),
            "refs": len(depths)}
