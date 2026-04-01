"""
复利引擎 - 自动下单模块
auto_trade.py

功能：
  - 坐标标定（点击界面上的4个位置）
  - 普通模式：固定金额下单
  - 复利模式：余额×赔率滚仓，赢N次停，输了重置
  - 支持币安（≥5u整数）和山寨所（≥3u三位小数）
  - 注册码验证（GitHub JSON在线验证）
"""

import os
import json
import time
import math
import threading
import requests

try:
    import pyautogui
    pyautogui.FAILSAFE = True   # 鼠标移到左上角可紧急停止
    pyautogui.PAUSE = 0.3
    PYAUTOGUI_OK = True
except ImportError:
    PYAUTOGUI_OK = False

# ========== 配置 ==========
COORDS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'auto_trade_coords.json')

# 注册码验证：优先读本地 license_repo/licenses.json，未配置 GitHub 时离线验证
# 如已上传 GitHub，把下面 LICENSE_URL 改成 raw 地址即可切换为在线验证
LICENSE_URL = "https://raw.githubusercontent.com/OxMagic/compound-engine-license/refs/heads/main/licenses.json"  # GitHub 在线验证
LICENSE_LOCAL = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'license_repo', 'licenses.json')
LICENSE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), '.license_cache')

# ========== 平台规则 ==========
PLATFORM_RULES = {
    "binance": {
        "name": "币安",
        "min_amount": 5.0,
        "decimals": 0,      # 取整，不能有小数
        "default_payout": 0.8,
    },
    "altcoin": {
        "name": "山寨所",
        "min_amount": 3.0,
        "decimals": 3,      # 最多3位小数
        "default_payout": 0.85,
    }
}


# ========== 金额格式化 ==========
def format_amount(amount: float, platform: str) -> float:
    """根据平台规则格式化下注金额"""
    rule = PLATFORM_RULES.get(platform, PLATFORM_RULES["binance"])
    decimals = rule["decimals"]
    min_amount = rule["min_amount"]

    if decimals == 0:
        # 币安：向下取整
        formatted = math.floor(amount)
    else:
        # 山寨所：保留N位小数，向下截断
        factor = 10 ** decimals
        formatted = math.floor(amount * factor) / factor

    # 不能低于最低金额
    return max(formatted, min_amount)


# ========== 复利计划计算 ==========
def calculate_compound_plan(initial: float, payout_rate: float, rounds: int, platform: str) -> list:
    """
    预先计算复利计划表（每轮应下注的金额）

    返回：[
        {"round": 1, "bet": 10.0, "expected_balance": 18.0},
        {"round": 2, "bet": 14.0, "expected_balance": 29.2},
        ...
    ]
    """
    plan = []
    balance = initial

    for i in range(1, rounds + 1):
        if i == 1:
            bet = format_amount(initial, platform)
        else:
            # 下注 = 当前余额 × 赔率
            bet = format_amount(balance * payout_rate, platform)

        win_profit = bet * payout_rate
        expected_balance = balance + win_profit

        plan.append({
            "round": i,
            "bet": bet,
            "balance_before": round(balance, 3),
            "win_profit": round(win_profit, 3),
            "expected_balance": round(expected_balance, 3),
        })

        balance = expected_balance

    return plan


# ========== 坐标管理 ==========
def load_coords() -> dict:
    """读取保存的坐标"""
    if os.path.exists(COORDS_FILE):
        with open(COORDS_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {
        "coords": {
            "amount": [0, 0],
            "buy_up": [0, 0],
            "buy_down": [0, 0],
            "confirm": [0, 0],
        },
        "trade_amount": 10,
        "saved_at": None
    }


def save_coords(coords: dict, trade_amount: float):
    """保存坐标到文件"""
    data = {
        "coords": coords,
        "trade_amount": trade_amount,
        "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    with open(COORDS_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def calibrate_coord(label: str) -> list:
    """
    交互式标定单个坐标
    提示用户把鼠标移到目标位置，按Enter确认
    """
    print(f"\n请把鼠标移到【{label}】的位置，然后按 Enter 确认...")
    input()
    x, y = pyautogui.position()
    print(f"  已记录：({x}, {y})")
    return [x, y]


def calibrate_all() -> dict:
    """标定全部4个坐标"""
    if not PYAUTOGUI_OK:
        return {"error": "pyautogui未安装"}

    print("\n=== 坐标标定模式 ===")
    print("请切换到交易所界面，然后按照提示依次标定4个坐标。")
    print("提示：鼠标移到左上角可紧急停止程序。\n")

    coords = {}
    coord_labels = {
        "amount":    "金额输入框",
        "buy_up":    "买涨按钮（UP/看涨）",
        "buy_down":  "买跌按钮（DOWN/看跌）",
        "confirm":   "确认下单按钮",
    }

    for key, label in coord_labels.items():
        coords[key] = calibrate_coord(label)
        time.sleep(0.5)

    print("\n✅ 标定完成！坐标已保存。")
    return coords


# ========== 下单执行 ==========
def click_amount_and_type(coords: dict, amount: float):
    """点击金额框并输入金额"""
    x, y = coords["amount"]
    pyautogui.click(x, y)
    time.sleep(0.3)
    pyautogui.hotkey('ctrl', 'a')
    time.sleep(0.1)
    pyautogui.press('delete')
    time.sleep(0.1)
    pyautogui.typewrite(str(amount), interval=0.05)
    time.sleep(0.2)


def click_direction(coords: dict, direction: str):
    """点击买涨或买跌"""
    if direction.lower() in ('up', 'long', '涨', 'buy_up'):
        x, y = coords["buy_up"]
    else:
        x, y = coords["buy_down"]
    pyautogui.click(x, y)
    time.sleep(0.3)


def click_confirm(coords: dict):
    """点击确认按钮"""
    x, y = coords["confirm"]
    pyautogui.click(x, y)
    time.sleep(0.5)


def execute_single_trade(coords: dict, amount: float, direction: str) -> bool:
    """
    执行单次下单
    返回 True=操作完成（不代表赢了）
    """
    if not PYAUTOGUI_OK:
        return False

    try:
        click_amount_and_type(coords, amount)
        click_direction(coords, direction)
        click_confirm(coords)
        return True
    except Exception as e:
        print(f"下单失败：{e}")
        return False


# ========== 自动交易核心类 ==========
class AutoTrader:
    """
    自动下单管理器
    支持普通模式和复利模式
    """

    def __init__(self):
        self.running = False
        self.thread = None
        self.status = "idle"       # idle / running / stopped / completed / error
        self.log = []
        self.current_round = 0
        self.current_balance = 0.0
        self._stop_event = threading.Event()

    def _append_log(self, msg: str):
        ts = time.strftime("%H:%M:%S")
        entry = f"[{ts}] {msg}"
        self.log.append(entry)
        print(entry)
        # 只保留最近100条
        if len(self.log) > 100:
            self.log = self.log[-100:]

    def stop(self):
        """外部停止信号"""
        self._stop_event.set()
        self.status = "stopped"
        self._append_log("⏹ 用户手动停止")

    def start_normal(self, coords: dict, amount: float, direction: str):
        """
        普通模式：固定金额，单次下单
        direction: 'up' 或 'down'，由预测信号给
        """
        if self.running:
            return {"error": "已有下单任务在运行"}

        def _run():
            self.running = True
            self.status = "running"
            self._append_log(f"▶ 普通模式下单：{amount}u → {direction}")
            ok = execute_single_trade(coords, amount, direction)
            if ok:
                self._append_log(f"✅ 下单完成：{amount}u {direction}")
                self.status = "completed"
            else:
                self._append_log("❌ 下单失败")
                self.status = "error"
            self.running = False

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return {"ok": True}

    def start_compound(self, coords: dict, initial: float, payout_rate: float,
                       rounds: int, platform: str, direction_func):
        """
        复利模式
        direction_func: callable，每轮调用获取本轮方向（'up'/'down'），
                        若返回None则等待信号
        """
        if self.running:
            return {"error": "已有下单任务在运行"}

        plan = calculate_compound_plan(initial, payout_rate, rounds, platform)
        self.current_balance = initial
        self._stop_event.clear()

        def _run():
            self.running = True
            self.status = "running"
            won_count = 0
            balance = initial

            self._append_log(f"▶ 复利模式启动：初始{initial}u，赔率{payout_rate}，目标{rounds}连赢")
            self._append_log(f"   平台：{PLATFORM_RULES[platform]['name']}")

            # 打印计划表
            for p in plan:
                self._append_log(
                    f"   第{p['round']}注预计：下注{p['bet']}u，"
                    f"余额{p['balance_before']}u → 赢后{p['expected_balance']}u"
                )

            while won_count < rounds and not self._stop_event.is_set():
                self.current_round = won_count + 1

                # 计算本轮下注金额
                if won_count == 0:
                    bet = format_amount(initial, platform)
                else:
                    bet = format_amount(balance * payout_rate, platform)

                self._append_log(f"\n⚡ 第{self.current_round}注 | 余额:{round(balance,3)}u | 下注:{bet}u")

                # 获取方向
                direction = None
                wait_count = 0
                while direction is None and not self._stop_event.is_set():
                    direction = direction_func()
                    if direction is None:
                        if wait_count % 10 == 0:
                            self._append_log("   ⏳ 等待预测信号...")
                        wait_count += 1
                        time.sleep(3)

                if self._stop_event.is_set():
                    break

                self._append_log(f"   信号：{'📈 买涨' if direction == 'up' else '📉 买跌'}")

                # 执行下单
                ok = execute_single_trade(coords, bet, direction)
                if not ok:
                    self._append_log("❌ 下单操作失败，停止复利")
                    self.status = "error"
                    break

                # 等待用户手动告知结果（或由外部回调设置）
                # 这里用一个简单的等待：等待外部调用 report_result()
                self._append_log("   ⏳ 等待本轮结果（请在结果出来后点击'赢'或'输'）...")
                result = self._wait_for_result()

                if result == 'win':
                    profit = bet * payout_rate
                    balance += profit
                    won_count += 1
                    self.current_balance = balance
                    self._append_log(
                        f"   🎉 赢！盈利:{round(profit,3)}u | "
                        f"余额:{round(balance,3)}u | 已连赢:{won_count}/{rounds}"
                    )

                elif result == 'loss':
                    balance_lost = balance - bet
                    self._append_log(
                        f"   💸 输！损失:{bet}u | "
                        f"余额重置为初始:{initial}u | 计数归零"
                    )
                    balance = initial
                    won_count = 0
                    self.current_balance = balance
                    # 短暂暂停再重新开始
                    self._append_log("   ⏸ 3秒后重新开始...")
                    time.sleep(3)

                else:
                    # 停止
                    break

            if won_count >= rounds:
                self._append_log(
                    f"\n🏆 复利完成！连赢{rounds}次，"
                    f"最终余额：{round(balance,3)}u"
                )
                self.status = "completed"
            elif self._stop_event.is_set():
                self.status = "stopped"
            
            self.running = False

        self._result_event = threading.Event()
        self._result_value = None

        self.thread = threading.Thread(target=_run, daemon=True)
        self.thread.start()
        return {"ok": True, "plan": plan}

    def _wait_for_result(self, timeout: int = 300) -> str:
        """等待外部设置本轮结果，最多等5分钟"""
        self._result_event = threading.Event()
        self._result_value = None
        self._result_event.wait(timeout=timeout)
        return self._result_value  # 'win' / 'loss' / None(超时)

    def report_result(self, result: str):
        """
        外部调用：告知本轮是赢(win)还是输(loss)
        前端按钮触发此方法
        """
        if result in ('win', 'loss'):
            self._result_value = result
            if hasattr(self, '_result_event'):
                self._result_event.set()

    def get_status(self) -> dict:
        """获取当前状态（供API查询）"""
        return {
            "running": self.running,
            "status": self.status,
            "current_round": self.current_round,
            "current_balance": round(self.current_balance, 3),
            "log": self.log[-20:],  # 最近20条日志
        }


# ========== 注册码验证 ==========
def _load_licenses_db() -> dict:
    """加载注册码数据库：优先 GitHub 在线，没有就读本地文件"""
    # 1. 尝试 GitHub 在线（仅配置了 URL 时）
    if LICENSE_URL:
        try:
            resp = requests.get(LICENSE_URL, timeout=8)
            if resp.status_code == 200:
                return resp.json()
        except Exception:
            pass
    # 2. 读本地 license_repo/licenses.json
    if os.path.exists(LICENSE_LOCAL):
        with open(LICENSE_LOCAL, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}


def verify_license(code: str) -> dict:
    """
    验证注册码：优先本地文件，配置了 GitHub URL 则在线验证
    返回 {"ok": True/False, "message": "..."}
    """
    if not code or len(code.strip()) < 5:
        return {"ok": False, "message": "注册码格式不正确"}

    # 先检查本地缓存（24小时内不重复验证）
    cache = _load_license_cache()
    if cache.get("code") == code and cache.get("verified"):
        cached_time = cache.get("time", 0)
        if time.time() - cached_time < 86400:  # 24小时
            return {"ok": True, "message": "已激活（缓存）"}

    try:
        licenses = _load_licenses_db()
        if not licenses:
            return {"ok": False, "message": "无法加载注册码数据库"}

        if code not in licenses:
            return {"ok": False, "message": "注册码无效"}

        entry = licenses[code]
        if not entry.get("active", False):
            return {"ok": False, "message": "注册码已失效"}

        if entry.get("used", 0) >= entry.get("limit", 0):
            return {"ok": False, "message": f"注册码已达激活上限（{entry['limit']}次）"}

        _save_license_cache(code)
        return {"ok": True, "message": f"激活成功！"}

    except Exception as e:
        return {"ok": False, "message": f"验证出错：{str(e)}"}


def _load_license_cache() -> dict:
    if os.path.exists(LICENSE_FILE):
        try:
            with open(LICENSE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def _save_license_cache(code: str):
    try:
        with open(LICENSE_FILE, 'w') as f:
            json.dump({"code": code, "verified": True, "time": time.time()}, f)
    except Exception:
        pass


# ========== 全局实例 ==========
trader = AutoTrader()


# ========== 测试入口 ==========
if __name__ == "__main__":
    print("=== 复利引擎 自动下单模块测试 ===\n")

    # 测试复利计划计算
    plan = calculate_compound_plan(10.0, 0.8, 5, "binance")
    print("币安 复利计划（初始10u，赔率0.8，5连赢）：")
    for p in plan:
        print(f"  第{p['round']}注：下注 {p['bet']}u | 余额 {p['balance_before']}u → {p['expected_balance']}u")

    print()
    plan2 = calculate_compound_plan(10.0, 0.85, 5, "altcoin")
    print("山寨所 复利计划（初始10u，赔率0.85，5连赢）：")
    for p in plan2:
        print(f"  第{p['round']}注：下注 {p['bet']}u | 余额 {p['balance_before']}u → {p['expected_balance']}u")
