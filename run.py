# -*- coding: utf-8 -*-
"""
一键启动 (双端口: 指挥端 + 民警终端):
  python run.py                 # 模拟视频源演示
  python run.py --source 0      # 摄像头
  python run.py --source a.mp4  # 本地视频文件
指挥端: http://127.0.0.1:8000   民警终端: http://127.0.0.1:8001
exe 启动后自动打开浏览器; 若检测到端口已被占用(已有实例在运行),
则直接打开正在运行实例的网页并退出, 避免 Errno 10048 崩溃.
"""
import argparse
import os
import socket
import sys
import threading
import time
import webbrowser

# Windows 控制台默认 GBK, 与 Python 的 UTF-8 输出不匹配会导致中文乱码: 切到 UTF-8 代码页
if os.name == "nt":
    try:
        import ctypes
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        ctypes.windll.kernel32.SetConsoleCP(65001)
    except Exception:
        pass
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass


def port_in_use(host: str, port: int) -> bool:
    """探测端口是否已有进程在监听."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.6)
        return s.connect_ex((host, port)) == 0


def wait_and_open(url: str, host: str, port: float, timeout: float = 60.0):
    """轮询等待服务进入监听状态后自动打开浏览器."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if port_in_use(host, port):
            webbrowser.open(url)
            return
        time.sleep(0.4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="sim", help="sim | 摄像头索引 | 视频文件路径")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--officer-port", type=int, default=0, help="民警终端端口(默认=指挥端+1)")
    parser.add_argument("--no-browser", action="store_true", help="启动后不自动打开浏览器")
    args = parser.parse_args()
    officer_port = args.officer_port or args.port + 1
    cmd_url = f"http://{args.host}:{args.port}"

    # 端口已被占用 => 已有实例在运行: 直接打开其网页并退出, 不再重复启动
    if port_in_use(args.host, args.port):
        print(f"[run] 检测到指挥端已在 {cmd_url} 运行, 已为你打开网页; 无需重复启动实例.")
        print("[run] 关闭此窗口即可, 正在运行的系统不受影响.")
        if not args.no_browser:
            webbrowser.open(cmd_url)
        time.sleep(1)
        raise SystemExit(0)

    os.environ["FLOOD_SOURCE"] = args.source
    import uvicorn
    from backend.app import app
    from backend.officer import officer_app

    def serve_officer():
        uvicorn.run(officer_app, host=args.host, port=officer_port, log_level="warning")

    if not args.no_browser:
        threading.Thread(target=wait_and_open, args=(cmd_url, args.host, args.port), daemon=True).start()
    threading.Thread(target=serve_officer, daemon=True).start()
    print(f"[run] 指挥端 {cmd_url} | 民警终端 http://{args.host}:{officer_port}")
    print("[run] 启动完成后浏览器将自动打开; 关闭此窗口即停止服务.")
    # 直接传 app 对象(而非 "backend.app:app" 导入字符串): exe 冻结环境下模块字符串导入不可靠
    uvicorn.run(app, host=args.host, port=args.port)
