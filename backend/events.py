# -*- coding: utf-8 -*-
"""告警事件管理: 内存列表 + JSON 文件持久化。"""
import json
import os
import threading
import time
import uuid

DATA_FILE = os.path.join(os.path.dirname(__file__), "events.json")


class EventStore:
    def __init__(self):
        self._lock = threading.Lock()
        self._events = []
        if os.path.exists(DATA_FILE):
            try:
                with open(DATA_FILE, encoding="utf-8") as f:
                    self._events = json.load(f)
            except Exception:
                self._events = []

    def add(self, city: str, cam: str, metrics: dict, judgment: dict, snapshot_b64: str = ""):
        ev = {
            "id": uuid.uuid4().hex[:8],
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "city": city,
            "cam": cam,
            "level": metrics["level"],
            "level_name": metrics["name"],
            "type": "人员涉水" if metrics["persons_in_water"] else "路面积水",
            "summary": judgment["summary"],
            "advice": judgment["advice"],
            "ai_source": judgment["source"],
            "status": "待处置",
            "snapshot": snapshot_b64,
        }
        with self._lock:
            self._events.insert(0, ev)
            self._events = self._events[:200]
            self._save()
        return ev

    def ack(self, ev_id: str) -> bool:
        """接受派发: 待处置/处置中 -> 处置中(已处置的事件不可回退)。"""
        with self._lock:
            for e in self._events:
                if e["id"] == ev_id:
                    if e["status"] == "已处置":
                        return False
                    e["status"] = "处置中"
                    self._save()
                    return True
        return False

    def resolve(self, ev_id: str) -> bool:
        """民警办结: 处置中 -> 已处置。"""
        with self._lock:
            for e in self._events:
                if e["id"] == ev_id:
                    if e["status"] != "处置中":
                        return False
                    e["status"] = "已处置"
                    e["resolved_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
                    self._save()
                    return True
        return False

    def list(self, city: str | None = None):
        with self._lock:
            evs = [dict(e, snapshot="") for e in self._events if not city or e.get("city") == city]
            return evs[:50]

    def today_count(self, city: str | None = None):
        today = time.strftime("%Y-%m-%d")
        return sum(1 for e in self._events
                   if e["time"].startswith(today) and (not city or e.get("city") == city))

    def _save(self):
        tmp = DATA_FILE + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self._events, f, ensure_ascii=False, indent=1)
        os.replace(tmp, DATA_FILE)
