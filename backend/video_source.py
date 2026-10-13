# -*- coding: utf-8 -*-
"""
视频源: 支持三种输入
1. 模拟城市内涝场景(默认, 演示/实验用): 照片级真实背景素材 + 每帧动态叠加
   (积水/波纹/倒影、车辆行人剪影与车灯、雨丝闪电、屏幕闪烁、信号故障) +
   CCTV 画质退化管线(低分辨率回弹/噪点/JPEG 块效应/暗角), 逼近真实监控画质。
2. 本地视频文件 (python run.py --source 某.mp4)
3. 摄像头     (python run.py --source 0)
"""
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np

_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
_ASSETS = {}  # 类间共享缓存: {文件名: 预处理后的 960x540 BGR 图}


def _imread_unicode(path: str):
    """Windows 下 cv2.imread 对含中文路径会失败, 用 fromfile+imdecode 规避。"""
    try:
        data = np.fromfile(path, dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _load_photo_bgs() -> list:
    """加载 assets/bg_*.jpg 照片级背景, 预处理成监控画质: 降饱和、压暗、低分辨率回弹。"""
    if "list" in _ASSETS:
        return _ASSETS["list"]
    out = []
    try:
        names = sorted(n for n in os.listdir(_ASSET_DIR) if n.startswith("bg_") and n.endswith(".jpg"))
    except OSError:
        names = []
    for n in names:
        img = _imread_unicode(os.path.join(_ASSET_DIR, n))
        if img is None or img.ndim != 3:
            continue
        # 跳过"生成中"占位图(固定 1832x1832 近白底), 素材就绪重启后自动生效
        if img.shape[0] == img.shape[1] == 1832 or img.mean() > 170:
            continue
        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_AREA)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv[:, :, 1] = (hsv[:, :, 1].astype(np.float32) * 0.7).astype(np.uint8)   # 降饱和(监控色彩)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        img = cv2.convertScaleAbs(img, alpha=0.86, beta=-8)                        # 压暗
        img = cv2.resize(img, (640, 360), interpolation=cv2.INTER_AREA)            # 低分辨率感
        img = cv2.resize(img, (960, 540), interpolation=cv2.INTER_LINEAR)
        img = cv2.GaussianBlur(img, (3, 3), 0)
        out.append(img)
    _ASSETS["list"] = out
    return out


class FileOrCameraSource:
    W, H = 960, 540  # 统一输出尺寸, 与模拟源/分析绘制坐标一致

    def __init__(self, src):
        try:
            src = int(src)
        except (TypeError, ValueError):
            pass
        self.src = src
        self._tmp = None
        self.cap = cv2.VideoCapture(src)
        # Windows 下部分 OpenCV 构建对含非 ASCII 字符的视频路径打不开,
        # 回退: 复制到纯 ASCII 临时路径再读
        if not self.cap.isOpened() and isinstance(src, str) and not src.isascii():
            import shutil
            import tempfile
            ext = os.path.splitext(src)[1] or ".mp4"
            fd, self._tmp = tempfile.mkstemp(prefix="floodsrc_", suffix=ext)
            os.close(fd)
            shutil.copyfile(src, self._tmp)
            self.cap = cv2.VideoCapture(self._tmp)
        if not self.cap.isOpened():
            raise RuntimeError(f"无法打开视频源: {src}")

    def read(self):
        ok, frame = self.cap.read()
        if not ok:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 循环播放
            ok, frame = self.cap.read()
        if frame is not None and (frame.shape[1], frame.shape[0]) != (self.W, self.H):
            frame = cv2.resize(frame, (self.W, self.H), interpolation=cv2.INTER_AREA)
        return frame

    def close(self):
        try:
            self.cap.release()
        except Exception:
            pass
        if self._tmp:
            try:
                os.remove(self._tmp)
            except OSError:
                pass


class SimulatedSource:
    """合成一段城市下穿通道内涝 CCTV 监控画面: 优先使用照片级背景素材
    (backend/assets/bg_*.jpg, 缺失时回退为程序化绘制场景), 每帧动态渲染
    积水(渐变/波纹/灯光倒影/泡沫)、车辆(剪影+车灯晕+溅水+水面倒影)、
    行人(摆腿/打伞/涟漪)、雨丝闪电、屏幕闪烁与信号故障条带, 最后经
    CCTV 画质退化管线(降分辨率/传感器噪点/JPEG 块效应/暗角)输出。
    水位随 base±amp 周期及真实降雨系数涨落, 参数化后覆盖不同城市态势。"""

    W, H = 960, 540
    VPY = 205  # 地平线 y (仅程序化回退背景使用)
    _VIGNETTE = None  # 暗角系数, 类级共享

    def __init__(self, cam_label: str = "CAM-01 建设路下穿段", base: float = 0.18,
                 amp: float = 0.13, period: float = 25.0, wading: bool = True, seed: int = 0):
        self.t0 = time.time()
        self.cam_label = cam_label
        self.cam_id = cam_label.split()[0]
        self.base, self.amp, self.period, self.wading = base, amp, period, wading
        self.base0 = base          # 原始基线, 供真实降雨系数缩放
        self.rain = 0.0            # 降雨强度(0=无雨), 由外部天气服务设置
        self.rng = np.random.RandomState(seed)
        self.noise = np.random.RandomState(seed + 7)
        self.seed = seed
        # 城市间差异化: 车速/车色/灯位随 seed 变化
        self.car_speed1 = 105 + (seed * 37) % 70
        self.car_speed2 = 80 + (seed * 53) % 60
        self.car_tint = [(36, 40, 120), (30, 60, 90), (60, 40, 40), (28, 28, 34), (50, 66, 110)][seed % 5]
        self.lamp_xs = [180 + (seed * 61) % 120, 460 + (seed * 97) % 120, 720 + (seed * 43) % 120]
        photos = _load_photo_bgs()
        self.photo = bool(photos)
        self.bg = photos[seed % len(photos)] if photos else self._build_background()

    @classmethod
    def _vignette(cls):
        if cls._VIGNETTE is None:
            yy, xx = np.mgrid[0:cls.H, 0:cls.W].astype(np.float32)
            d = np.sqrt(((xx - cls.W / 2) / (cls.W / 2)) ** 2 + ((yy - cls.H / 2) / (cls.H / 2)) ** 2)
            cls._VIGNETTE = np.clip(1.06 - 0.30 * np.clip(d - 0.35, 0, None), 0.62, 1.06)[:, :, None]
        return cls._VIGNETTE

    def set_rain_factor(self, factor: float):
        """按真实降雨放大积水基线(上限 0.42), factor=1 表示无雨。"""
        factor = max(1.0, min(2.2, factor))
        self.base = min(0.42, self.base0 * factor)
        self.rain = factor - 1.0

    # ---------------- 程序化背景(素材缺失时) ----------------
    def _glow(self, img, cx, cy, rad, color, strength=0.55):
        """在指定位置叠加柔和光晕(辉光层高斯模糊后回贴, 留足模糊边距避免方块状裁切)。"""
        W, H = self.W, self.H
        pad = int(rad * 2.4) + 2
        x1, y1 = max(cx - pad, 0), max(cy - pad, 0)
        x2, y2 = min(cx + pad, W), min(cy + pad, H)
        if x2 - x1 < 4 or y2 - y1 < 4:
            return
        roi = img[y1:y2, x1:x2].astype(np.float32)
        g = np.zeros_like(roi)
        cv2.circle(g, (cx - x1, cy - y1), rad, color, -1)
        g = cv2.GaussianBlur(g, (0, 0), rad / 2.0)
        img[y1:y2, x1:x2] = np.clip(roi + g * strength, 0, 255).astype(np.uint8)

    def _build_background(self):
        W, H, VPY = self.W, self.H, self.VPY
        rng = self.rng
        img = np.zeros((H, W, 3), np.uint8)
        # 阴天天空渐变(压暗, 傍晚雨前)
        top, bot = np.array([42, 48, 66], float), np.array([108, 112, 120], float)
        for y in range(VPY):
            img[y, :] = top + (bot - top) * (y / VPY) ** 1.3
        # 远景楼群: 灰蓝剪影 + 大气透视(整带模糊)
        for x in range(-20, W, 58):
            hgt, w = int(rng.randint(50, 130)), int(rng.randint(38, 66))
            cv2.rectangle(img, (x, VPY - hgt), (min(x + w, W), VPY), (86, 92, 104), -1)
        band = img[VPY - 140:VPY]
        img[VPY - 140:VPY] = cv2.GaussianBlur(band, (0, 0), 2.6)
        # 近景楼群: 深色 + 暖色亮窗(带辉光)
        for x in range(-10, W, 92):
            hgt, w = int(rng.randint(60, 128)), int(rng.randint(46, 76))
            x2 = min(x + w, W)
            cv2.rectangle(img, (x, VPY - hgt), (x2, VPY), (34, 38, 48), -1)
            cv2.rectangle(img, (x, VPY - hgt), (x2, VPY - hgt + 3), (26, 29, 37), -1)   # 女儿墙
            for wy in range(VPY - hgt + 8, VPY - 6, 13):
                for wx in range(x + 5, x2 - 5, 11):
                    if rng.rand() < 0.2:
                        c = (126, 168, 208) if rng.rand() < 0.75 else (190, 180, 140)
                        cv2.rectangle(img, (wx, wy), (wx + 4, wy + 6), c, -1)
                        if rng.rand() < 0.3:
                            self._glow(img, wx + 2, wy + 3, 7, (70, 100, 130), 0.35)
        # 下穿场景主体: 横跨高架桥(梁体+阴影渐变) + 桥墩 + 桥下钠灯
        cv2.rectangle(img, (0, 143), (W, 152), (62, 66, 74), -1)
        cv2.rectangle(img, (0, 152), (W, 182), (41, 45, 53), -1)
        cv2.rectangle(img, (0, 182), (W, 196), (52, 56, 64), -1)                        # 桥下受光带
        for x in range(6, W, 96):
            cv2.rectangle(img, (x, 182), (x + 16, VPY), (48, 52, 60), -1)
            cv2.rectangle(img, (x, 182), (x + 3, VPY), (60, 64, 72), -1)                # 墩受光棱
            self._glow(img, x + 40, 190, 16, (60, 140, 205), 0.5)                       # 桥下照明
        # 人行道: 基底 + 斑驳纹理 + 盲道
        img[VPY:, :] = (84, 84, 90)
        for _ in range(2600):
            x, y = int(rng.randint(0, W)), int(rng.randint(VPY, H))
            g = int(rng.randint(70, 105))
            img[y, x] = (g, g, g + 6)
        for x in range(0, W, 26):
            cv2.line(img, (x, VPY + 4), (x + 10, VPY + 4), (74, 78, 96), 2)             # 盲道条带
        # 道路透视梯形 + 沥青噪点
        road = np.array([[W // 2 - 26, VPY], [W // 2 + 26, VPY], [W - 64, H], [64, H]], np.int32)
        cv2.fillPoly(img, [road], (56, 56, 62))
        for _ in range(4200):
            x = int(rng.randint(64, W - 64))
            y = int(rng.randint(VPY, H))
            if cv2.pointPolygonTest(road, (float(x), float(y)), False) >= 0:
                g = int(rng.randint(42, 78))
                img[y, x] = (g, g, g + 6)
        # 路缘石阴影(路面与人行道分界, 消除"贴纸"感)
        for side in (-1, 1):
            p0 = (int(W // 2 + side * 26), VPY)
            p1 = (int(W // 2 + side * (W // 2 - 64)), H)
            cv2.line(img, p0, p1, (66, 66, 72), 4)
            cv2.line(img, (p0[0] + side * 4, p0[1]), (p1[0] + side * 6, p1[1]), (98, 98, 104), 2)
        # 油污斑块与裂缝(先建层再模糊回贴, 更像旧沥青)
        patch = np.zeros_like(img)
        for _ in range(9):
            px, py = int(rng.randint(140, W - 140)), int(rng.randint(VPY + 60, H - 30))
            cv2.ellipse(patch, (px, py), (int(rng.randint(22, 55)), int(rng.randint(8, 18))),
                        0, 0, 360, (28, 28, 32), -1)
        for _ in range(5):
            px, py = int(rng.randint(120, W - 120)), int(rng.randint(VPY + 40, H - 40))
            pts = np.array([[px + k * 9, py + int(rng.randint(-6, 7))] for k in range(6)], np.int32)
            cv2.polylines(patch, [pts], False, (30, 30, 34), 1)
        m = (patch.sum(axis=2) > 60).astype(np.uint8)
        patch = cv2.GaussianBlur(patch, (0, 0), 2.2)
        img = np.where(m[:, :, None] > 0, cv2.addWeighted(img, 0.55, patch, 0.45, 0), img).astype(np.uint8)
        # 透视车道虚线(黄色中线 + 白色车道线, 近大远小)
        s = 0.06
        while s < 1.0:
            y = int(VPY + (H - VPY) * s ** 1.75)
            half = (W // 2 - 64) * s ** 1.75
            dash = max(int(16 * s), 2)
            if int(s * 16) % 2 == 0:
                cv2.line(img, (W // 2, y), (W // 2, y + dash), (55, 55, 195), max(int(3 * s), 1))
                for off in (-0.55, 0.55):
                    lx = int(W // 2 + off * half)
                    cv2.line(img, (lx, y), (lx, y + dash), (200, 200, 205), max(int(3 * s), 1))
            s += 0.04
        # 路缘白线
        for k in range(60):
            s0, s1 = k / 60, min((k + 1) / 60, 1.0)
            for side in (-1, 1):
                p0 = (int(W // 2 + side * (W // 2 - 70) * s0 ** 1.75), int(VPY + (H - VPY) * s0 ** 1.75))
                p1 = (int(W // 2 + side * (W // 2 - 70) * s1 ** 1.75), int(VPY + (H - VPY) * s1 ** 1.75))
                cv2.line(img, p0, p1, (185, 185, 190), 1)
        # 远处车辆灯光(消失点附近模糊亮点)
        for dx, c in ((-9, (70, 70, 220)), (7, (70, 70, 220)), (-3, (60, 120, 210))):
            self._glow(img, W // 2 + dx, VPY + 16 + abs(dx), 6, c, 0.8)
        # 路灯(杆+挑臂+灯头辉光)与行道树
        for x, side in [(118, 1), (W - 118, -1)]:
            cv2.line(img, (x, H - 30), (x, 236), (28, 32, 40), 4)
            cv2.line(img, (x, 236), (x - side * 34, 244), (28, 32, 40), 3)
            cv2.circle(img, (x - side * 34, 248), 4, (195, 210, 255), -1)
            self._glow(img, x - side * 34, 250, 26, (90, 120, 170), 0.5)
            self._glow(img, x - side * 34, 252, 60, (50, 70, 110), 0.28)                # 大范围光雾
        for x in (52, W - 52):
            cv2.rectangle(img, (x - 3, 330), (x + 3, 392), (44, 36, 28), -1)
            cv2.circle(img, (x, 312), 26, (38, 68, 44), -1)
            cv2.circle(img, (x - 14, 326), 18, (32, 58, 38), -1)
            cv2.circle(img, (x + 12, 322), 15, (42, 74, 48), -1)
        # 湿滑路面反光带(钠灯暖色竖向拖影; 避免冷蓝以免被积水分割误检)
        for lx in (76, W - 76, W // 2):
            refl = np.zeros_like(img)
            for k in range(14):
                cv2.line(refl, (lx + int(rng.randint(-4, 5)), VPY + 30 + k * 22),
                         (lx + int(rng.randint(-6, 7)), VPY + 44 + k * 22), (95, 125, 165), 3)
            refl = cv2.GaussianBlur(refl, (0, 0), 3.5)
            img = cv2.addWeighted(img, 1.0, refl, 0.4, 0)
        # 井盖
        cv2.ellipse(img, (int(W * 0.6), int(H * 0.86)), (26, 9), 0, 0, 360, (44, 44, 50), -1)
        cv2.ellipse(img, (int(W * 0.6), int(H * 0.86)), (26, 9), 0, 0, 360, (60, 60, 66), 1)
        # 景深: 地平线附近轻微失焦
        blur_band = cv2.GaussianBlur(img[VPY - 8:VPY + 60], (0, 0), 1.8)
        img[VPY - 8:VPY + 60] = blur_band
        return img

    # ---------------- 每帧动态渲染 ----------------
    def read(self):
        t = time.time() - self.t0
        frame = self.bg.copy()
        W, H = self.W, self.H
        level = self.base + self.amp * math.sin(t / self.period) + 0.02 * math.sin(t / 3.7)
        y_water = int(H * (1 - min(level, 0.52)))

        # 行人先画, 下半身会被后续水层覆盖(没水效果)
        persons = [(46, 0, H - 42, 64), (33, 0, H - 76, 64)]
        if self.wading:
            persons.append((0, int(W * 0.62 + 30 * math.sin(t / 6)), H - 26, 70))
        for i, (speed, x, foot_y, hgt) in enumerate(persons):
            if speed:
                x = int((W + 80) - ((t * speed + i * 400 + self.seed * 130) % (W + 160)))
            self._draw_person(frame, x, min(foot_y, H - 8), hgt, t, i * 1.7)

        # 车辆: 双向行驶(先画, 水线以下部分随后被水覆盖)
        c1x = int((t * self.car_speed1 + self.seed * 210) % (W + 320)) - 160
        c2x = W - int((t * self.car_speed2 + self.seed * 90) % (W + 320)) + 160
        self._draw_car(frame, c1x, int(H * 0.62), 0.62, 1)
        self._draw_car(frame, c2x, int(H * 0.86), 1.0, -1)

        # 积水层(保持蓝色调主导, 供 OpenCV 积水分割识别)
        if y_water < H - 8:
            hgt = H - y_water
            yy = np.linspace(0, 1, hgt)[:, None, None]
            c1c, c2c = np.array([132, 106, 66], float), np.array([92, 64, 36], float)
            band = np.tile((c1c + (c2c - c1c) * yy).astype(np.uint8), (1, W, 1))
            frame[y_water:H] = cv2.addWeighted(frame[y_water:H], 0.40, band, 0.60, 0)
            # 波纹亮线(间距/相位不规则, 避免"斑马纹"塑料感)
            for i in range(hgt // 12):
                if i % 3 == 2:
                    continue
                wy = y_water + 6 + i * 12 + int(3 * math.sin(t * 1.5 + i * 1.3))
                b = 26 - (i % 4) * 7
                pts = [(x, wy + int(2.6 * math.sin(x / 34 + t * 2.2 + i) + 1.4 * math.sin(x / 13 + t * 3.1)))
                       for x in range(0, W + 24, 24)]
                for a, c in zip(pts, pts[1:]):
                    cv2.line(frame, a, c, (100 + b, 86 + b, 60 + b), 1)
            # 灯光水面倒影(位置随机位变化)
            for rx in self.lamp_xs:
                for k in range(0, hgt, 12):
                    jx = rx + int(5 * math.sin(t * 2.2 + k / 8))
                    cv2.line(frame, (jx, y_water + k), (jx, min(y_water + k + 7, H - 1)), (150, 144, 104), 1)
            # 车灯光在水面的竖向拖影
            for cx, cy in ((c1x + 80, int(H * 0.62)), (c2x + 120, int(H * 0.86))):
                if cy > y_water - 6 and 0 < cx < W:
                    for k in range(0, min(46, H - y_water), 4):
                        jx = cx + int(3 * math.sin(t * 3.1 + k / 5))
                        cv2.line(frame, (jx, y_water + k), (jx, y_water + k + 3), (70, 70, 200), 2)
            # 水线泡沫
            pts = np.array([(x, y_water + int(3 * math.sin(x / 20 + t * 3.2))) for x in range(0, W, 12)], np.int32)
            cv2.polylines(frame, [pts], False, (178, 182, 190), 1)
            # 涉水人员脚部涟漪
            if self.wading and H - 26 > y_water:
                px = int(W * 0.62 + 30 * math.sin(t / 6))
                cv2.ellipse(frame, (px, y_water + 6), (20, 6), 0, 0, 360, (185, 185, 192), 1)
            # 车辆驶过积水区溅起水花
            for cx, cy in ((c1x + 80, int(H * 0.62)), (c2x + 120, int(H * 0.86))):
                if cy > y_water and 0 < cx < W:
                    for _ in range(7):
                        dx = int(self.noise.randint(-26, 26))
                        dy = int(self.noise.randint(-16, -2))
                        cv2.circle(frame, (cx + dx, cy + dy), int(self.noise.randint(1, 4)), (195, 195, 202), -1)

        # 雨丝(与真实降雨联动) + 偶尔闪电
        if self.rain > 0.15:
            n = int(60 * min(self.rain, 1.2))
            for i in range(n):
                rx = (i * 97 + int(t * 900)) % W
                ry = (i * 53 + int(t * 1300)) % H
                cv2.line(frame, (rx, ry), (rx - 6, ry + 18), (200, 202, 232), 1)
            if self.rain > 0.55 and (t % 6.3) < 0.14:
                frame = cv2.add(frame, np.full_like(frame, 46))

        # 屏幕亮度微闪(监控摄像机自动增益抖动)
        if self.noise.rand() < 0.12:
            frame = cv2.convertScaleAbs(frame, alpha=1.0, beta=int(self.noise.randint(-7, 5)))

        frame = self._degrade(frame, t)
        return self._osd(frame, t)

    # ---------------- CCTV 画质退化管线 ----------------
    def _degrade(self, frame, t):
        """降分辨率回弹 + 传感器噪点 + 偶发信号故障条带 + JPEG 块效应 + 暗角,
        让合成元素与照片素材统一成监控摄像头的低画质观感。"""
        W, H = self.W, self.H
        frame = cv2.resize(frame, (672, 378), interpolation=cv2.INTER_AREA)
        frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_LINEAR)
        # 传感器噪点(半分辨率生成, 省算力)
        nz = self.noise.randint(-4, 5, (H // 2, W // 2, 3)).astype(np.int16)
        nz = cv2.resize(nz, (W, H), interpolation=cv2.INTER_LINEAR).astype(np.int16)
        frame = np.clip(frame.astype(np.int16) + nz, 0, 255).astype(np.uint8)
        # 偶发信号故障: 横向错位条带
        if self.rng.rand() < 0.05:
            y = int(self.rng.randint(0, H))
            bh = int(self.rng.randint(2, 7))
            dx = int(self.rng.randint(-7, 8))
            frame[y:y + bh] = np.roll(frame[y:y + bh], dx, axis=1)
        # JPEG 压缩块效应
        ok, enc = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 34])
        if ok:
            frame = cv2.imdecode(enc, cv2.IMREAD_COLOR)
        # 暗角
        frame = (frame.astype(np.float32) * self._vignette()).astype(np.uint8)
        return frame

    def _draw_person(self, img, x, foot_y, hgt, t, phase):
        color = (26, 26, 32)
        top = foot_y - hgt
        cv2.circle(img, (x, top), max(hgt // 7, 5), color, -1)                      # 头
        cv2.rectangle(img, (x - hgt // 6, top + 8), (x + hgt // 6, top + hgt // 2), color, -1)  # 躯干
        swing = int(5 * math.sin(t * 6 + phase))
        hip_y = top + hgt // 2
        cv2.line(img, (x - 2, hip_y), (x - 5 + swing, foot_y), color, max(hgt // 14, 3))  # 腿
        cv2.line(img, (x + 2, hip_y), (x + 5 - swing, foot_y), color, max(hgt // 14, 3))
        cv2.line(img, (x - hgt // 6, top + 12), (x - hgt // 5 - swing // 2, hip_y), color, 3)  # 手臂
        if self.rain > 0.15:  # 打伞
            cv2.ellipse(img, (x, top - 4), (24, 12), 0, 180, 360, (48, 48, 150), -1)
            cv2.line(img, (x, top - 4), (x, top + 10), (34, 34, 40), 2)
        self._soften(img, x - 30, top - 20, x + 30, foot_y + 4)

    def _draw_car(self, img, x, y, s, dirn):
        """dirn=1 车头朝右(行驶方向), -1 车头朝左; 深色剪影+车灯+软边, 低画质下接近真实车辆。"""
        w, h = int(160 * s), int(46 * s)
        body = self.car_tint
        roof = (min(body[0] + 16, 255), min(body[1] + 16, 255), min(body[2] + 22, 255))
        cv2.rectangle(img, (x, y - h), (x + w, y - int(h * 0.35)), body, -1)          # 车身下半
        cv2.rectangle(img, (x + int(w * 0.22), y - h - int(22 * s)),
                      (x + int(w * 0.74), y - h + 5), roof, -1)                        # 车顶
        cv2.fillPoly(img, [np.array([[x + int(w * 0.24), y - h + 5], [x + int(w * 0.34), y - h - int(18 * s)],
                                     [x + int(w * 0.62), y - h - int(18 * s)], [x + int(w * 0.70), y - h + 5]], np.int32)],
                     (52, 52, 60))                                                     # 车窗
        for wx in (x + int(w * 0.2), x + int(w * 0.8)):
            cv2.circle(img, (wx, y - int(h * 0.12)), int(10 * s), (16, 16, 20), -1)    # 车轮
        fx = x + w - 8 if dirn > 0 else x + 8                                          # 车头灯
        bx = x + 6 if dirn > 0 else x + w - 6                                          # 尾灯
        cv2.circle(img, (fx, y - int(h * 0.55)), int(5 * s), (150, 215, 255), -1)
        cv2.circle(img, (fx, y - int(h * 0.55)), int(11 * s), (90, 140, 200), -1)      # 前照灯光晕
        cv2.circle(img, (bx, y - int(h * 0.55)), int(4 * s), (40, 40, 225), -1)
        cv2.circle(img, (bx, y - int(h * 0.55)), int(9 * s), (26, 26, 120), -1)        # 尾灯光晕
        self._soften(img, x - 14, y - h - int(30 * s), x + w + 14, y + 6)

    @staticmethod
    def _soften(img, x1, y1, x2, y2):
        """对目标区域做轻微高斯模糊, 消除硬边缘的"贴图感"。"""
        W, H = img.shape[1], img.shape[0]
        x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
        if x2 - x1 < 6 or y2 - y1 < 6:
            return
        roi = img[y1:y2, x1:x2]
        img[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (5, 5), 0)

    def _osd(self, frame, t):
        """监控摄像头风格叠加: 机位号/分辨率 + 闪烁 REC + 毫秒时间码。"""
        W, H = self.W, self.H
        frame[0:36] = cv2.addWeighted(frame[0:36], 0.45, np.zeros((36, W, 3), np.uint8), 0.55, 0)
        cv2.putText(frame, f"{self.cam_id}  UNDERPASS CAM  1080P", (10, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (235, 235, 235), 1)
        if int(t * 2) % 2 == 0:
            cv2.circle(frame, (W - 92, 18), 5, (0, 0, 255), -1)
            cv2.putText(frame, "REC", (W - 80, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        now = datetime.now()
        ts = now.strftime("%Y-%m-%d %H:%M:%S") + f".{now.microsecond // 10000:02d}"
        cv2.putText(frame, ts, (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (240, 240, 240), 2)
        return frame


def make_source(arg: str | None):
    if arg is None or arg == "sim":
        return SimulatedSource()
    return FileOrCameraSource(arg)
