# -*- coding: utf-8 -*-
"""
运行路径统一解析(源码运行 / PyInstaller exe 冻结运行 双模式):
- 冻结模式: 随程序打包的只读资源(frontend/)位于 _MEIPASS 临时解压目录;
  运行期可写数据(uploads 视频、地图瓦片缓存、events.json)一律放在 exe 同级目录,
  保证程序退出/升级后数据不丢失。
- 源码模式: 两者均为项目根目录, 与原有行为完全一致。
"""
import os
import sys

FROZEN = getattr(sys, "frozen", False)

if FROZEN:
    RES_DIR = sys._MEIPASS                                  # 打包进 exe 的资源根
    DATA_DIR = os.path.dirname(os.path.abspath(sys.executable))  # exe 同级目录
else:
    RES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATA_DIR = RES_DIR

FRONTEND_DIR = os.path.join(RES_DIR, "frontend")
UPLOAD_DIR = os.path.join(DATA_DIR, "uploads")
TILE_CACHE_DIR = os.path.join(DATA_DIR, ".tilecache")
