"""
复利引擎 - 启动器 v11
launcher.py

极简策略（与 C# v11 对称）：
  1. 双击 exe → 检查端口 → 已有服务在跑 → 开浏览器然后退出
  2. 没有服务 → 获取 Mutex → 失败则等2秒轮询端口 → 开浏览器或退出
  3. Mutex 成功 → 清理残留 → 启动服务 → 等就绪 → 开浏览器
  4. 浏览器带 10 秒去重保护（通过 .browser_opened 标志文件，与 C# 共用）

窗口闪动修复：FreeConsole() + stdout/stderr 重定向
"""

import sys
import os
import threading
import time

# ──────────────────────────────────────────────
# 第一步：立即隐藏控制台窗口
# ──────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        user32 = ctypes.windll.user32
        hwnd = kernel32.GetConsoleWindow()
        if hwnd:
            user32.ShowWindow(hwnd, 0)
        kernel32.FreeConsole()
    except Exception:
        pass

    _log_dir = os.path.dirname(sys.executable)
    _stdout_path = os.path.join(_log_dir, 'console_output.log')
    try:
        _log_file = open(_stdout_path, 'a', encoding='utf-8', buffering=1)
        sys.stdout = _log_file
        sys.stderr = _log_file
    except Exception:
        pass

# ──────────────────────────────────────────────
# 路径设置
# ──────────────────────────────────────────────
if getattr(sys, 'frozen', False):
    BASE_DIR = sys._MEIPASS
    WORK_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    WORK_DIR = BASE_DIR

os.chdir(WORK_DIR)

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

PORT = 7788
URL = f"http://localhost:{PORT}"
BROWSER_FLAG = '.browser_opened'
BROWSER_COOLDOWN = 10  # 10秒冷却期

# ──────────────────────────────────────────────
# Windows Mutex
# ──────────────────────────────────────────────
_mutex_handle = None

def acquire_mutex():
    global _mutex_handle
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        _mutex_handle = kernel32.CreateMutexW(None, False, "Global\\CompoundEngineMutex")
        if _mutex_handle:
            last_error = kernel32.GetLastError()
            return last_error != 183
        return False
    except Exception:
        return False

def release_mutex():
    global _mutex_handle
    try:
        if _mutex_handle:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.ReleaseMutex(_mutex_handle)
            kernel32.CloseHandle(_mutex_handle)
            _mutex_handle = None
    except Exception:
        pass


def is_server_running():
    """检查 7788 端口"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(0.3)
        result = s.connect_ex(('127.0.0.1', PORT))
        s.close()
        return result == 0
    except Exception:
        return False


def open_browser():
    """打开浏览器，带去重保护 + 训练中不打开"""
    flag_path = os.path.join(WORK_DIR, BROWSER_FLAG)
    try:
        # 训练中 → 不开浏览器
        if os.path.exists(os.path.join(WORK_DIR, '.training')):
            log_path = os.path.join(WORK_DIR, 'launcher_debug.log')
            try:
                with open(log_path, 'a', encoding='utf-8') as f:
                    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] open_browser() skipped (training in progress), PID={os.getpid()}\n")
            except Exception:
                pass
            return

        # 检查标志文件：10秒内已开过就跳过
        if os.path.exists(flag_path):
            try:
                with open(flag_path, 'r') as f:
                    content = f.read().strip()
                timestamp = int(content)
                elapsed = (time.time() * 1000) - timestamp  # ms
                if elapsed < BROWSER_COOLDOWN * 1000:
                    # 冷却期内，不开浏览器
                    log_path = os.path.join(WORK_DIR, 'launcher_debug.log')
                    try:
                        with open(log_path, 'a', encoding='utf-8') as f:
                            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] open_browser() skipped (cooldown {elapsed:.0f}ms < {BROWSER_COOLDOWN*1000}ms), PID={os.getpid()}\n")
                    except Exception:
                        pass
                    return
            except (ValueError, OSError):
                pass

        # 更新标志文件（Unix 毫秒时间戳，与 C# 启动器格式一致）
        with open(flag_path, 'w') as f:
            f.write(str(int(time.time() * 1000)))
    except Exception:
        pass

    try:
        import ctypes
        log_path = os.path.join(WORK_DIR, 'launcher_debug.log')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] open_browser(), PID={os.getpid()}\n")
        except Exception:
            pass
        ctypes.windll.shell32.ShellExecuteW(None, "open", URL, None, None, 1)
    except Exception:
        pass


def kill_port_users():
    """杀掉占用 7788 端口的残留进程"""
    try:
        import subprocess
        result = subprocess.run(
            ['netstat', '-ano', '-p', 'TCP'],
            capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.splitlines():
            if f':{PORT}' in line and 'LISTENING' in line:
                parts = line.split()
                if len(parts) >= 5:
                    pid = parts[-1]
                    try:
                        subprocess.run(['taskkill', '/F', '/PID', pid],
                                       capture_output=True, timeout=5)
                    except Exception:
                        pass
        # 清理浏览器标志和训练标志（杀残留意味着要重新启动）
        for flag in (BROWSER_FLAG, '.training'):
            try:
                os.remove(os.path.join(WORK_DIR, flag))
            except Exception:
                pass
    except Exception:
        pass


def run_server():
    """启动 Flask 后端"""
    try:
        import compound_engine_server as srv
        import logging
        log = logging.getLogger('werkzeug')
        log.setLevel(logging.ERROR)
        srv.main()
    except Exception as e:
        log_path = os.path.join(WORK_DIR, 'launcher_debug.log')
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] server error: {e}\n")
        except Exception:
            pass


def wait_and_open():
    """等 Flask 就绪 → 开浏览器"""
    import urllib.request
    for _ in range(30):  # 最多 3 秒
        try:
            urllib.request.urlopen(URL, timeout=0.1)
            break
        except Exception:
            time.sleep(0.1)
    open_browser()


def keep_alive():
    """静默保持进程存活"""
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        while True:
            kernel32.Sleep(60000)
    except Exception:
        threading.Event().wait()


def cleanup():
    """退出时清理"""
    release_mutex()
    for f in ['.compound_engine.pid', '.compound_engine.lock']:
        try:
            p = os.path.join(WORK_DIR, f)
            if os.path.exists(p):
                os.remove(p)
        except Exception:
            pass


# ──────────────────────────────────────────────
# 主入口
# ──────────────────────────────────────────────
if __name__ == '__main__':
    is_frozen = getattr(sys, 'frozen', False)

    log_path = os.path.join(WORK_DIR, 'launcher_debug.log')
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] === launcher v11 started, PID={os.getpid()}, frozen={is_frozen} ===\n")
    except Exception:
        pass

    if not is_frozen:
        # 非 frozen：开发模式，直接启动
        try:
            import compound_engine_server as srv
            srv.main()
        except Exception:
            pass
        sys.exit(0)

    # ── frozen 模式 ──

    # 步骤1：检查服务是否已经在跑
    if is_server_running():
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] server already running, opening browser\n")
        except Exception:
            pass
        open_browser()
        sys.exit(0)

    # 步骤2：没在跑 → 获取 Mutex
    if not acquire_mutex():
        # Mutex 被占 → 别人在启动中 → 等2秒轮询
        try:
            with open(log_path, 'a', encoding='utf-8') as f:
                f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] mutex busy, waiting\n")
        except Exception:
            pass
        for _ in range(20):
            time.sleep(0.1)
            if is_server_running():
                open_browser()
                sys.exit(0)
        sys.exit(0)

    # 步骤3：我是第一个 → 清理残留 → 启动服务
    try:
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] first instance, starting server\n")
    except Exception:
        pass

    kill_port_users()
    time.sleep(0.2)

    import atexit
    atexit.register(cleanup)

    try:
        with open(os.path.join(WORK_DIR, '.compound_engine.pid'), 'w') as f:
            f.write(str(os.getpid()))
    except Exception:
        pass

    t_server = threading.Thread(target=run_server, daemon=True)
    t_server.start()

    t_browser = threading.Thread(target=wait_and_open, daemon=True)
    t_browser.start()

    keep_alive()
