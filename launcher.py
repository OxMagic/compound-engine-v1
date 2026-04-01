"""
复利引擎 - 启动器
launcher.py

双击 exe 后自动：
  1. 在后台启动 Flask 后端（端口 7788）
  2. 等后端就绪后用系统默认浏览器打开 http://localhost:7788
  3. 主窗口显示状态提示（sys.frozen 环境下用 tkinter 小窗口）
"""

import sys
import os
import threading
import time
import webbrowser

# ──────────────────────────────────────────────
# PyInstaller 打包后，数据文件在 sys._MEIPASS 下
# ──────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS          # 解压临时目录
    WORK_DIR = os.path.dirname(sys.executable)  # exe 所在目录（存 models/logs）
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WORK_DIR = BASE_DIR

# 切到工作目录（让 models/log 写在 exe 旁边，不是临时目录）
os.chdir(WORK_DIR)

# 把 BASE_DIR 加入 sys.path，让 import compound_engine_server / auto_trade 找得到
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PORT = 7788
URL  = f"http://localhost:{PORT}"


# ──────────────────────────────────────────────
# 后端线程
# ──────────────────────────────────────────────
def run_server():
    """在子线程里启动 Flask，不阻塞主线程"""
    import compound_engine_server as srv
    # 关掉 werkzeug 的彩色 banner，避免控制台乱码
    import logging
    log = logging.getLogger('werkzeug')
    log.setLevel(logging.ERROR)
    srv.app.run(host='127.0.0.1', port=PORT, debug=False, use_reloader=False)


def wait_and_open():
    """等后端 ready 再开浏览器"""
    import urllib.request
    for _ in range(30):          # 最多等 15 秒
        try:
            urllib.request.urlopen(URL, timeout=1)
            break
        except Exception:
            time.sleep(0.5)
    webbrowser.open(URL)


# ──────────────────────────────────────────────
# 托盘/状态小窗（仅在打包环境启用，避免控制台闪退）
# ──────────────────────────────────────────────
def keep_alive():
    """用 Windows API 静默等待，不弹出任何窗口"""
    try:
        import ctypes
        # 用 kernel32 的 Sleep 循环保持进程存活
        # 每60秒检查一次，进程会因为 daemon 线程自动退出
        kernel32 = ctypes.windll.kernel32
        while True:
            kernel32.Sleep(60000)
    except Exception:
        # 兜底方案
        threading.Event().wait()


# ──────────────────────────────────────────────
# 入口
# ──────────────────────────────────────────────
if __name__ == '__main__':
    # 1. 后台跑服务
    t_server = threading.Thread(target=run_server, daemon=True)
    t_server.start()

    # 2. 等服务 ready → 开浏览器
    t_browser = threading.Thread(target=wait_and_open, daemon=True)
    t_browser.start()

    # 3. 静默保持进程存活（不弹任何窗口）
    if getattr(sys, 'frozen', False):
        keep_alive()
    else:
        # 开发模式：直接等
        t_server.join()
