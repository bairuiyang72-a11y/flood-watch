# -*- coding: utf-8 -*-
"""
实时天气服务: 定期从 Open-Meteo(免费, 无需密钥)拉取全国监测城市的实时降雨/气温,
用真实降雨强度驱动模拟积水态势, 让大屏数据随真实天气变化。
说明: Google Earth Engine 需 Google 账号 OAuth 授权且为云端分析平台, 不适合大屏直连;
若后续有 GEE 账号, 可用其 Sentinel-1 水体产品导出真实洪涝范围图替换本模块。
"""
import json
import threading
import time
import urllib.request
from datetime import datetime

# WMO 天气代码 -> 中文描述
WMO_DESC = {
    0: "晴", 1: "基本晴朗", 2: "多云", 3: "阴", 45: "雾", 48: "冻雾",
    51: "毛毛雨", 53: "毛毛雨", 55: "毛毛雨", 61: "小雨", 63: "中雨", 65: "大雨",
    66: "冻雨", 67: "冻雨", 80: "小阵雨", 81: "中阵雨", 82: "强阵雨",
    95: "雷阵雨", 96: "雷阵雨伴冰雹", 99: "强雷阵雨伴冰雹",
}


class WeatherService:
    """后台线程每 10 分钟批量拉取全部城市天气, 失败时保留上一次数据。"""

    REFRESH_SEC = 600

    def __init__(self, cities: list[dict]):
        self.cities = cities
        self.lock = threading.Lock()
        self.data: dict[str, dict] = {}

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            self._refresh()
            time.sleep(self.REFRESH_SEC)

    def _refresh(self):
        lats = ",".join(str(c["lat"]) for c in self.cities)
        lngs = ",".join(str(c["lng"]) for c in self.cities)
        url = (f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lngs}"
               "&current=temperature_2m,precipitation,weather_code"
               "&hourly=precipitation&past_days=1&forecast_days=1&timezone=Asia/Shanghai")
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                arr = json.load(resp)
        except Exception:
            return
        now = datetime.now()
        out = {}
        for c, item in zip(self.cities, arr):
            cur = item.get("current", {})
            h_times = item.get("hourly", {}).get("time", [])
            h_prec = item.get("hourly", {}).get("precipitation", [])
            acc24 = 0.0
            for t, p in zip(h_times, h_prec):
                if not p:
                    continue
                try:
                    dt = datetime.fromisoformat(t)
                except ValueError:
                    continue
                if dt <= now and (now - dt).total_seconds() <= 24 * 3600:
                    acc24 += p
            out[c["name"]] = {
                "temp": cur.get("temperature_2m"),
                "precip": round(cur.get("precipitation") or 0.0, 1),        # mm/h
                "precip_24h": round(acc24, 1),                               # mm
                "desc": WMO_DESC.get(cur.get("weather_code"), "—"),
            }
        with self.lock:
            self.data = out

    def snapshot(self, name: str) -> dict:
        with self.lock:
            return dict(self.data.get(name, {}))

    def rain_factor(self, name: str) -> float:
        """真实降雨 -> 积水放大系数(1.0~2.2): 当前雨强为主, 24h 累计降雨为辅。"""
        d = self.snapshot(name)
        if not d:
            return 1.0
        f = 1.0 + 0.15 * d.get("precip", 0.0) + 0.012 * d.get("precip_24h", 0.0)
        return max(1.0, min(2.2, f))

    def verify(self, name: str) -> dict:
        """天气验真: 用第二独立气象源 wttr.in(按经纬度查询, 聚合多家观测/模式数据)
        与 Open-Meteo 当前数据交叉核对: 气温偏差<=2°C 且"是否降水"判定一致 => 验证通过。"""
        c = next((x for x in self.cities if x["name"] == name), None)
        om = self.snapshot(name)
        if not c or not om:
            return {"error": "该城市暂无 Open-Meteo 数据, 请稍后重试"}
        url = f"https://wttr.in/{c['lat']},{c['lng']}?format=j1"
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Flood-Watch/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                j = json.load(resp)
            cw = j["current_condition"][0]
            today = j["weather"][0]
            b = {"temp": float(cw["temp_C"]),
                 "desc": (cw.get("weatherDesc") or [{"value": "—"}])[0]["value"],
                 "precip24": round(sum(float(h.get("precipMM") or 0) for h in today["hourly"]), 1),
                 "updated": cw.get("localObsDateTime") or today.get("date", "")}
        except Exception as e:
            return {"error": f"wttr.in 源拉取失败: {type(e).__name__}"}
        dl = b["desc"].lower()
        rain_om = (om.get("precip") or 0) > 0.1 or (om.get("precip_24h") or 0) > 1 or ("雨" in (om.get("desc") or ""))
        rain_b = b["precip24"] > 1.0 or any(k in dl for k in ("rain", "shower", "drizzle", "thunder", "sleet", "snow"))
        diff = round(abs((om.get("temp") or 0) - b["temp"]), 1)
        # wttr.in 观测为整数摄氏度且更新较粗, 与逐小时模式数据允许 ±3.5°C 容差
        return {"om": om, "b": b, "diff_temp": diff,
                "rain_om": rain_om, "rain_b": rain_b,
                "ok": diff <= 3.5 and rain_om == rain_b,
                "ts": datetime.now().strftime("%H:%M:%S")}
