# -*- coding: utf-8 -*-
"""
民警终端服务 (独立端口, 默认 8001):
- 接收指挥端派发的告警事件, 一线民警在此接受派发(处置中)并办结(已处置)。
- 与指挥端同进程运行, 共享同一 EventStore, 状态实时互通。
"""
import os

from fastapi import FastAPI, Query
from fastapi.responses import FileResponse, JSONResponse

from .app import store
from .paths import FRONTEND_DIR

officer_app = FastAPI(title="Flood-Watch 民警终端")


@officer_app.get("/")
def officer_index():
    return FileResponse(os.path.join(FRONTEND_DIR, "officer.html"))


@officer_app.get("/api/events")
def officer_events(city: str | None = Query(None), status: str | None = Query(None)):
    evs = store.list(city)
    if status:
        evs = [e for e in evs if e.get("status") == status]
    return evs


@officer_app.post("/api/events/{ev_id}/ack")
def officer_ack(ev_id: str):
    ok = store.ack(ev_id)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)


@officer_app.post("/api/events/{ev_id}/resolve")
def officer_resolve(ev_id: str):
    ok = store.resolve(ev_id)
    return JSONResponse({"ok": ok}, status_code=200 if ok else 404)
