# -*- coding: utf-8 -*-
"""
FastAPI 主服务: 多城市视频分析流水线 + MJPEG 推流 + REST API
城市洪涝监测与公安应急处置平台 (Flood-Watch)
"""
import base64
import os
import threading
import time
import urllib.request

import cv2
from fastapi import FastAPI, File, Form, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

from .analyzer import llm, objects, risk, water
from .cities import CITIES
from .events import EventStore
from .hotspots import HOTSPOTS
from .video_source import SimulatedSource, FileOrCameraSource
from .weather import WeatherService

FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "uploads")
VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".flv", ".wmv", ".webm"}

app = FastAPI(title="Flood-Watch 城市洪涝监测与公安应急处置平台")
segmenter = water.WaterSegmenter()
detector = objects.ObjectDetector()
store = EventStore()
weather = WeatherService(CITIES)


class Analyzer:
    """单城市检测站分析线程: 逐帧分割/检测/评估, 触发告警事件。"""

    def __init__(self, city: dict, source):
        self.city = city["name"]
        self.cam = city["cam"]
        self.source = source
        self.lock = threading.Lock()
        self.annotated = None
        self.mask_png = None
        self.metrics = {"level": 1, "name": "低风险", "color": "#22c55e", "action": "常规监测",
                        "water_ratio": 0.0, "growth": 0.0, "persons_in_water": 0, "cars_in_water": 0}
        self.judgment = {"summary": "", "advice": "", "source": "template"}
        self.risk_engine = risk.RiskEngine()
        self.last_alert_level = 0
        self.last_alert_ts = 0.0
        self.fps = 0.0
        self.running = True
        self.last_weather_ts = 0.0
        self.source_kind = "sim"   # sim=模拟画面 / file=真实视频
        self.source_path = None

    def start(self):
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        prev = time.time()
        while self.running:
            frame = self.source.read()
            if frame is None:
                time.sleep(0.05)
                continue
            # 每 30s 按城市实时降雨调整模拟积水态势
            if isinstance(self.source, SimulatedSource) and time.time() - self.last_weather_ts > 30:
                self.last_weather_ts = time.time()
                self.source.set_rain_factor(weather.rain_factor(self.city))
            mask, ratio, _ = segmenter.segment(frame)
            objs = detector.detect(frame)
            w_hits = objects.person_in_water(objs, mask)
            metrics = self.risk_engine.assess(ratio, len(w_hits), 0)
            annotated = self._draw(frame, mask, objs, w_hits, metrics)
            now = time.time()
            self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - prev, 1e-3))
            prev = now

            # 告警去抖: 风险等级>=3 且(升高或距上次>120s)才生成新事件+AI研判
            if metrics["level"] >= 3 and (
                metrics["level"] > self.last_alert_level or now - self.last_alert_ts > 120
            ):
                self.last_alert_level = metrics["level"]
                self.last_alert_ts = now
                ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 60])
                snap = base64.b64encode(buf.tobytes()).decode() if ok else ""
                judgment = llm.analyze(annotated, metrics)
                store.add(self.city, self.cam, metrics, judgment, snap)
                with self.lock:
                    self.judgment = judgment

            ok, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 72])
            _, png = cv2.imencode(".png", mask)
            with self.lock:
                self.annotated = buf.tobytes()
                self.mask_png = png.tobytes()
                self.metrics = metrics
            time.sleep(0.2)  # 每城市约 5fps, 全国 31 城并行时控制 CPU 占用

    def _draw(self, frame, mask, objs, w_hits, metrics):
        overlay = frame.copy()
        overlay[mask > 0] = (255, 120, 0)  # 积水区橙色高亮
        out = cv2.addWeighted(frame, 0.65, overlay, 0.35, 0)
        hit_ids = {id(o) for o in w_hits}
        for o in objs:
            x1, y1, x2, y2 = o["box"]
            color = (0, 0, 255) if id(o) in hit_ids else (0, 220, 0)
            cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
            label = "WADING!" if id(o) in hit_ids else o["cls"]
            cv2.putText(out, label, (x1, max(y1 - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        bar = (f"{self.cam.split()[0]} | RISK L{metrics['level']}  "
               f"WATER {metrics['water_ratio']*100:.1f}%  PERSON {metrics['persons_in_water']}")
        cv2.rectangle(out, (0, 36), (640, 66), (0, 0, 0), -1)
        cv2.putText(out, bar, (8, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.62,
                    tuple(int(metrics["color"][i:i+2], 16) for i in (1, 3, 5)), 2)
        return out

    def snapshot(self):
        with self.lock:
            return (self.annotated, self.mask_png, dict(self.metrics), dict(self.judgment))


analyzers: dict[str, Analyzer] = {}
city_meta: dict[str, dict] = {c["name"]: c for c in CITIES}
_city_index = {c["name"]: i for i, c in enumerate(CITIES)}


def _make_sim_source(city_name: str) -> SimulatedSource:
    c = city_meta[city_name]
    return SimulatedSource(cam_label=c["cam"], base=c["base"], amp=c["amp"],
                           period=c["period"], wading=c["wading"], seed=_city_index[city_name])


def _close_source_later(src):
    """换源后延后关闭旧 VideoCapture: 分析线程可能正处在一次 read() 中, 立即释放有竞态。"""
    if not isinstance(src, FileOrCameraSource):
        return

    def _w():
        time.sleep(0.8)
        src.close()

    threading.Thread(target=_w, daemon=True).start()


@app.on_event("startup")
def _startup():
    weather.start()  # 后台线程拉取全国实时天气
    real_src = os.environ.get("FLOOD_SOURCE", "sim")
    for i, c in enumerate(CITIES):
        if real_src != "sim" and i == 0:
            source = FileOrCameraSource(real_src)  # 首个城市接真实视频源
        else:
            source = _make_sim_source(c["name"])
        a = Analyzer(c, source)
        if real_src != "sim" and i == 0:
            a.source_kind = "file"
        a.start()
        analyzers[c["name"]] = a


def _pick(city: str | None) -> Analyzer:
    return analyzers.get(city or "") or next(iter(analyzers.values()))


@app.get("/api/cities")
def api_cities():
    out = []
    for c in CITIES:
        a = analyzers[c["name"]]
        _, _, m, _ = a.snapshot()
        out.append({"name": c["name"], "province": c.get("province", ""),
                    "lng": c["lng"], "lat": c["lat"],
                    "cam": c["cam"], "cams_online": c["cams_online"],
                    "level": m["level"], "level_name": m["name"], "color": m["color"],
                    "water_ratio": m["water_ratio"], "persons_in_water": m["persons_in_water"],
                    "weather": weather.snapshot(c["name"]),
                    "fps": round(a.fps, 1)})
    return out


@app.get("/api/hotspots")
def api_hotspots():
    """全国内涝易发区地图标注(地级市/县域/村镇), 仅作态势标注不参与视频分析。"""
    return HOTSPOTS


@app.get("/api/bootstrap")
def api_bootstrap(city: str = Query("武汉")):
    """页面首屏数据: 城市列表 + 易发区标注 + 默认城市状态/事件一次性返回。
    把首轮 4 波共 6 个请求合并为 1 个, 页面加载中途刷新时在途请求窗口从
    ~1.2s 压缩到几十毫秒, 从根本上消除刷新时批量 net::ERR_ABORTED 报错。"""
    return {"cities": api_cities(), "hotspots": HOTSPOTS,
            "status": api_status(city), "events": store.list(city)}


@app.get("/api/status")
def api_status(city: str | None = Query(None)):
    a = _pick(city)
    _, _, metrics, judgment = a.snapshot()
    return {
        "city": a.city, "cam": a.cam, "fps": round(a.fps, 1),
        "cams_online": city_meta[a.city]["cams_online"],
        "metrics": metrics, "judgment": judgment,
        "weather": weather.snapshot(a.city),
        "events_today": store.today_count(a.city),
        "detector": "ultralytics-yolo" if detector.model else "opencv-heuristic",
        "segmenter": "sam-lsm-adapter" if segmenter.use_large_model else "opencv",
        "source_kind": a.source_kind,
        "source_file": os.path.basename(a.source_path or ""),
    }


@app.get("/api/events")
def api_events(city: str | None = Query(None)):
    return store.list(city)


@app.get("/api/weather/verify")
def api_weather_verify(city: str = Query(...)):
    """双独立气象源交叉验证(Open-Meteo vs 挪威气象研究所)。"""
    return weather.verify(city)


# ---------------- 真实监控视频接入 ----------------
@app.post("/api/video/upload")
async def api_video_upload(city: str = Form(...), file: UploadFile = File(...)):
    """上传监控录像文件并热切换该城市分析线程的视频源为真实视频。"""
    a = analyzers.get(city)
    if a is None:
        return JSONResponse({"ok": False, "error": f"未知城市: {city}"}, status_code=404)
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in VIDEO_EXTS:
        return JSONResponse({"ok": False,
                             "error": f"不支持的格式 {ext or '(无扩展名)'}, 仅支持 "
                                      + "/".join(sorted(VIDEO_EXTS))}, status_code=400)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    safe_name = os.path.basename(file.filename or "video.mp4").replace("/", "_").replace("\\", "_")
    save_path = os.path.join(UPLOAD_DIR, f"{city}_{safe_name}")
    data = await file.read()
    if len(data) == 0:
        return JSONResponse({"ok": False, "error": "空文件"}, status_code=400)
    with open(save_path, "wb") as f:
        f.write(data)
    # 先验证可解码, 失败则回滚删除
    try:
        new_src = FileOrCameraSource(save_path)
    except Exception as e:
        try:
            os.remove(save_path)
        except OSError:
            pass
        return JSONResponse({"ok": False,
                             "error": f"视频无法解码({len(data)}字节): {e}"}, status_code=400)
    old = a.source
    a.source = new_src
    a.source_kind = "file"
    a.source_path = save_path
    _close_source_later(old)
    return {"ok": True, "city": city, "file": safe_name,
            "size": len(data), "source_kind": "file"}


@app.post("/api/video/sim")
def api_video_sim(city: str = Form(...)):
    """恢复某城市为模拟画面源。"""
    a = analyzers.get(city)
    if a is None:
        return JSONResponse({"ok": False, "error": f"未知城市: {city}"}, status_code=404)
    old = a.source
    a.source = _make_sim_source(city)
    a.source_kind = "sim"
    a.source_path = None
    _close_source_later(old)
    return {"ok": True, "city": city, "source_kind": "sim"}


@app.get("/api/videos")
def api_videos():
    """列出已上传的监控视频文件。"""
    if not os.path.isdir(UPLOAD_DIR):
        return []
    out = []
    for n in sorted(os.listdir(UPLOAD_DIR)):
        p = os.path.join(UPLOAD_DIR, n)
        if os.path.isfile(p) and os.path.splitext(n)[1].lower() in VIDEO_EXTS:
            st = os.stat(p)
            city = n.split("_", 1)[0] if "_" in n else ""
            out.append({"name": n, "city": city, "size": st.st_size,
                        "mtime": int(st.st_mtime)})
    return out


@app.post("/api/events/{ev_id}/ack")
def api_ack(ev_id: str):
    ok = store.ack(ev_id)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


@app.get("/video_feed")
def video_feed(city: str | None = Query(None)):
    def gen():
        while True:
            a = _pick(city)
            data, _, _, _ = a.snapshot()
            if data:
                yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + data + b"\r\n")
            time.sleep(1 / 15)
    return StreamingResponse(gen(), media_type="multipart/x-mixed-replace; boundary=frame")


@app.get("/mask.png")
def mask_png(city: str | None = Query(None)):
    _, png, _, _ = _pick(city).snapshot()
    return Response(png or b"", media_type="image/png")


@app.get("/snapshot.jpg")
def snapshot_jpg(city: str | None = Query(None)):
    """单帧快照: 前端以"上一帧加载完再请求下一帧"方式轮询, 替代 MJPEG 长连接,
    页面刷新时几乎没有在途请求, 从根源消除 net::ERR_ABORTED 控制台报错。"""
    data, _, _, _ = _pick(city).snapshot()
    return Response(data or b"", media_type="image/jpeg", headers={"Cache-Control": "no-store"})


# ---------------- 地图瓦片同源代理 + 磁盘缓存 ----------------
# 前端不再直连高德域名: 瓦片经本服务缓存后同源返回, 加载快且避免浏览器
# 在页面重载时取消外部在途请求而产生的 net::ERR_ABORTED 控制台报错。
TILE_LAYERS = {
    "road": "https://webrd0{sub}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}",
    "img": "https://webst0{sub}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}",
    "label": "https://webst0{sub}.is.autonavi.com/appmaptile?style=8&lang=zh_cn&x={x}&y={y}&z={z}",
}
TILE_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".tilecache")


@app.get("/tiles/{kind}/{z}/{x}/{y}")
def tiles(kind: str, z: int, x: int, y: int):
    tpl = TILE_LAYERS.get(kind)
    if not tpl or not (0 <= z <= 18):
        return Response(status_code=404)
    os.makedirs(TILE_CACHE_DIR, exist_ok=True)
    fp = os.path.join(TILE_CACHE_DIR, f"{kind}_{z}_{x}_{y}.png")
    if os.path.exists(fp):
        return FileResponse(fp, media_type="image/png", headers={"Cache-Control": "max-age=86400"})
    url = tpl.format(sub=(x + y) % 4 + 1, x=x, y=y, z=z)
    try:
        with urllib.request.urlopen(url, timeout=6) as r:
            data = r.read()
        with open(fp, "wb") as f:
            f.write(data)
        return Response(data, media_type="image/png", headers={"Cache-Control": "max-age=86400"})
    except Exception:
        return Response(status_code=502)


app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
