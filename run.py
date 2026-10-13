# -*- coding: utf-8 -*-
"""
一键启动:
  python run.py                 # 模拟视频源演示
  python run.py --source 0      # 摄像头
  python run.py --source a.mp4  # 本地视频文件
然后浏览器打开 http://127.0.0.1:8000
"""
import argparse
import os

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sim", help="sim | 摄像头索引 | 视频文件路径")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    os.environ["FLOOD_SOURCE"] = args.source
    import uvicorn
    uvicorn.run("backend.app:app", host=args.host, port=args.port)
