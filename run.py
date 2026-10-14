# -*- coding: utf-8 -*-
"""
一键启动 (双端口: 指挥端 + 民警终端):
  python run.py                 # 模拟视频源演示
  python run.py --source 0      # 摄像头
  python run.py --source a.mp4  # 本地视频文件
指挥端: http://127.0.0.1:8000   民警终端: http://127.0.0.1:8001
"""
import argparse
import os
import threading

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sim", help="sim | 摄像头索引 | 视频文件路径")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--officer-port", type=int, default=0, help="民警终端端口(默认=指挥端+1)")
    args = parser.parse_args()
    officer_port = args.officer_port or args.port + 1

    os.environ["FLOOD_SOURCE"] = args.source
    import uvicorn
    from backend.officer import officer_app

    def serve_officer():
        uvicorn.run(officer_app, host=args.host, port=officer_port, log_level="warning")

    threading.Thread(target=serve_officer, daemon=True).start()
    print(f"[run] 指挥端 http://{args.host}:{args.port} | 民警终端 http://{args.host}:{officer_port}")
    uvicorn.run("backend.app:app", host=args.host, port=args.port)
