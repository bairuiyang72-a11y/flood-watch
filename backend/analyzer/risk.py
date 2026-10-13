# -*- coding: utf-8 -*-
"""洪涝风险评估规则引擎: 融合积水面积占比、变化速率、涉水人员等模态。"""

LEVELS = {
    1: ("低风险", "#22c55e", "常规监测"),
    2: ("较低风险", "#eab308", "加强巡查"),
    3: ("中风险", "#f97316", "预置力量"),
    4: ("高风险", "#ef4444", "紧急处置"),
}


class RiskEngine:
    def __init__(self):
        self._history = []  # 最近积水占比, 用于估计上涨速率

    def assess(self, water_ratio: float, persons_in_water: int, cars_in_water: int) -> dict:
        self._history.append(water_ratio)
        if len(self._history) > 30:
            self._history.pop(0)
        # 速率: 当前窗口内占比变化(近似每帧增长)
        growth = 0.0
        if len(self._history) >= 10:
            growth = (self._history[-1] - self._history[-10]) / 9.0

        score = 0
        if water_ratio > 0.03: score += 1
        if water_ratio > 0.10: score += 1
        if water_ratio > 0.22: score += 1
        if growth > 0.002: score += 1
        if persons_in_water > 0: score += 2
        if cars_in_water > 0: score += 1
        level = max(1, min(4, 1 + score // 2)) if score else 1
        if persons_in_water > 0:
            level = max(level, 4)

        name, color, action = LEVELS[level]
        return {
            "level": level, "name": name, "color": color, "action": action,
            "water_ratio": round(water_ratio, 4),
            "growth": round(growth, 5),
            "persons_in_water": persons_in_water,
            "cars_in_water": cars_in_water,
        }
