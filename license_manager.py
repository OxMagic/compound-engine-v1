"""
复利引擎 - 注册码管理工具
用法:
  python license_manager.py                        # 查看所有注册码
  python license_manager.py --gen 5                # 批量生成5个码
  python license_manager.py --gen 5 --limit 100    # 批量生成，限100次
  python license_manager.py --gen 1 --type vip --limit 50
  python license_manager.py --info CE-XXXX-XXXX    # 查看某个码详情
  python license_manager.py --disable CE-XXXX-XXXX # 禁用某个码
  python license_manager.py --enable CE-XXXX-XXXX  # 启用某个码
  python license_manager.py --reset CE-XXXX-XXXX   # 重置使用次数
  python license_manager.py --delete CE-XXXX-XXXX  # 删除某个码
  python license_manager.py --note CE-XXXX-XXXX "备注内容"  # 修改备注
"""

import json
import sys
import os
import string
import random
from datetime import datetime

# Windows 控制台 UTF-8 支持
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

LICENSE_FILE = os.path.join(os.path.dirname(__file__), "license_repo", "licenses.json")

# ============ 工具函数 ============

def load_licenses():
    """加载 licenses.json"""
    if not os.path.exists(LICENSE_FILE):
        return {}
    with open(LICENSE_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def save_licenses(data):
    """保存到 licenses.json"""
    os.makedirs(os.path.dirname(LICENSE_FILE), exist_ok=True)
    with open(LICENSE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"  ✅ 已保存到 {LICENSE_FILE}")


def generate_code(prefix="CE"):
    """生成一个注册码: CE-XXXX-XXXX"""
    chars = string.ascii_uppercase + string.digits
    part1 = "".join(random.choices(chars, k=4))
    part2 = "".join(random.choices(chars, k=4))
    return f"{prefix}-{part1}-{part2}"


def check_exists(data, code):
    """检查码是否已存在"""
    if code in data:
        print(f"  ⚠️  注册码 {code} 已存在！")
        return True
    return False


# ============ 命令实现 ============

def cmd_list(data):
    """列出所有注册码"""
    if not data:
        print("  📭 暂无注册码")
        return

    print(f"\n{'='*70}")
    print(f"  复利引擎 注册码管理  |  共 {len(data)} 个码")
    print(f"{'='*70}\n")

    for code, info in data.items():
        status = "🟢 启用" if info.get("active", True) else "🔴 禁用"
        limit = info.get("limit", 0)
        used = info.get("used", 0)
        limit_str = "∞" if limit >= 99999 else str(limit)
        note = info.get("note", "")
        expires = info.get("expires_at", "")

        print(f"  📋 {code}")
        print(f"     状态: {status}  |  使用: {used}/{limit_str}")
        if expires:
            print(f"     过期: {expires}")
        if note:
            print(f"     备注: {note}")
        print()


def cmd_gen(data, count, limit, code_type, note_prefix):
    """批量生成注册码"""
    print(f"\n  🔨 生成 {count} 个注册码...")
    print(f"     类型: {code_type}  |  限制: {limit}次\n")

    type_prefixes = {
        "free": "CE-FREE",
        "vip": "CE-VIP",
        "trial": "CE-TRIAL",
        "admin": "CE-OXMAGIC",
        "custom": "CE",
    }
    prefix = type_prefixes.get(code_type, "CE")

    generated = []
    for i in range(count):
        # 确保不重复
        for _ in range(100):
            code = generate_code(prefix)
            if code not in data and code not in [g["code"] for g in generated]:
                break

        note = f"{note_prefix} #{i+1}" if note_prefix else f"{code_type}码 #{i+1}"
        info = {
            "limit": limit,
            "used": 0,
            "active": True,
            "note": note,
        }
        if code_type == "trial":
            # 试用码默认30天过期
            from datetime import timedelta
            exp = (datetime.now() + timedelta(days=30)).strftime("%Y-%m-%d")
            info["expires_at"] = exp

        data[code] = info
        generated.append({"code": code, "info": info})
        print(f"  ✅ {code}  ({note})")

    save_licenses(data)

    # 汇总
    print(f"\n  📊 汇总:")
    print(f"     生成: {count} 个")
    print(f"     总计: {len(data)} 个注册码")

    return generated


def cmd_info(data, code):
    """查看某个注册码详情"""
    if code not in data:
        print(f"  ❌ 注册码 {code} 不存在")
        return

    info = data[code]
    limit = info.get("limit", 0)
    used = info.get("used", 0)
    limit_str = "∞" if limit >= 99999 else str(limit)

    print(f"\n  📋 注册码详情: {code}")
    print(f"  {'─'*40}")
    print(f"  状态:     {'🟢 启用' if info.get('active', True) else '🔴 禁用'}")
    print(f"  使用次数: {used} / {limit_str}")
    remaining = (limit - used) if limit < 99999 else -1
    print(f"  剩余:     {'∞' if remaining < 0 else str(remaining)}")
    print(f"  备注:     {info.get('note', '无')}")
    expires = info.get("expires_at", "")
    if expires:
        print(f"  过期时间: {expires}")


def cmd_disable(data, code):
    """禁用注册码"""
    if code not in data:
        print(f"  ❌ 注册码 {code} 不存在")
        return
    data[code]["active"] = False
    save_licenses(data)
    print(f"  🔴 已禁用 {code}")


def cmd_enable(data, code):
    """启用注册码"""
    if code not in data:
        print(f"  ❌ 注册码 {code} 不存在")
        return
    data[code]["active"] = True
    save_licenses(data)
    print(f"  🟢 已启用 {code}")


def cmd_reset(data, code):
    """重置使用次数"""
    if code not in data:
        print(f"  ❌ 注册码 {code} 不存在")
        return
    old_used = data[code].get("used", 0)
    data[code]["used"] = 0
    save_licenses(data)
    print(f"  🔄 已重置 {code} 的使用次数 ({old_used} → 0)")


def cmd_delete(data, code):
    """删除注册码"""
    if code not in data:
        print(f"  ❌ 注册码 {code} 不存在")
        return
    confirm = input(f"  ⚠️  确认删除 {code}？(y/N): ").strip().lower()
    if confirm != "y":
        print("  已取消")
        return
    del data[code]
    save_licenses(data)
    print(f"  🗑️  已删除 {code}")


def cmd_note(data, code, note):
    """修改备注"""
    if code not in data:
        print(f"  ❌ 注册码 {code} 不存在")
        return
    data[code]["note"] = note
    save_licenses(data)
    print(f"  ✏️  {code} 备注已更新: {note}")


# ============ 参数解析 ============

def parse_args():
    args = sys.argv[1:]
    result = {
        "command": "list",
        "code": None,
        "count": 1,
        "limit": 99999,
        "type": "free",
        "note": "",
    }

    i = 0
    while i < len(args):
        arg = args[i]
        if arg in ("--gen",):
            result["command"] = "gen"
            if i + 1 < len(args) and args[i + 1].isdigit():
                result["count"] = int(args[i + 1])
                i += 1
        elif arg == "--limit":
            if i + 1 < len(args) and args[i + 1].isdigit():
                result["limit"] = int(args[i + 1])
                i += 1
        elif arg == "--type":
            if i + 1 < len(args):
                result["type"] = args[i + 1].lower()
                i += 1
        elif arg == "--note":
            if i + 1 < len(args):
                result["note"] = args[i + 1]
                i += 1
        elif arg == "--info":
            result["command"] = "info"
            if i + 1 < len(args):
                result["code"] = args[i + 1].upper()
                i += 1
        elif arg == "--disable":
            result["command"] = "disable"
            if i + 1 < len(args):
                result["code"] = args[i + 1].upper()
                i += 1
        elif arg == "--enable":
            result["command"] = "enable"
            if i + 1 < len(args):
                result["code"] = args[i + 1].upper()
                i += 1
        elif arg == "--reset":
            result["command"] = "reset"
            if i + 1 < len(args):
                result["code"] = args[i + 1].upper()
                i += 1
        elif arg == "--delete":
            result["command"] = "delete"
            if i + 1 < len(args):
                result["code"] = args[i + 1].upper()
                i += 1
        elif arg in ("--note", "--set-note"):
            result["command"] = "note"
            if i + 1 < len(args):
                result["code"] = args[i + 1].upper()
                i += 1
            # 等下个参数作为备注内容
            # 其实在解析阶段不好区分，换个方式
        elif arg in ("--help", "-h"):
            print(__doc__)
            sys.exit(0)
        else:
            # 可能是直接传的注册码（查看/禁用等）
            if result["command"] == "list" and "-" in arg:
                result["command"] = "info"
                result["code"] = arg.upper()
        i += 1

    return result


# ============ 主入口 ============

def main():
    args = sys.argv[1:]
    data = load_licenses()

    if not args:
        cmd_list(data)
        return
    if args[0] in ("--help", "-h", "help"):
        print(__doc__)
        return

    cmd = args[0]

    if cmd == "list":
        cmd_list(data)

    elif cmd == "gen":
        count = 1
        limit = 99999
        code_type = "free"
        note_prefix = ""

        i = 1
        while i < len(args):
            if args[i] == "--limit" and i + 1 < len(args):
                limit = int(args[i + 1])
                i += 2
            elif args[i] == "--type" and i + 1 < len(args):
                code_type = args[i + 1].lower()
                i += 2
            elif args[i] == "--note" and i + 1 < len(args):
                note_prefix = args[i + 1]
                i += 2
            else:
                if args[i].isdigit():
                    count = int(args[i])
                i += 1

        cmd_gen(data, count, limit, code_type, note_prefix)

    elif cmd == "info":
        if len(args) < 2:
            print("  ❌ 请提供注册码: python license_manager.py info CE-XXXX-XXXX")
            return
        cmd_info(data, args[1].upper())

    elif cmd == "disable":
        if len(args) < 2:
            print("  ❌ 请提供注册码: python license_manager.py disable CE-XXXX-XXXX")
            return
        cmd_disable(data, args[1].upper())

    elif cmd == "enable":
        if len(args) < 2:
            print("  ❌ 请提供注册码: python license_manager.py enable CE-XXXX-XXXX")
            return
        cmd_enable(data, args[1].upper())

    elif cmd == "reset":
        if len(args) < 2:
            print("  ❌ 请提供注册码: python license_manager.py reset CE-XXXX-XXXX")
            return
        cmd_reset(data, args[1].upper())

    elif cmd == "delete":
        if len(args) < 2:
            print("  ❌ 请提供注册码: python license_manager.py delete CE-XXXX-XXXX")
            return
        cmd_delete(data, args[1].upper())

    elif cmd == "note":
        if len(args) < 3:
            print('  ❌ 请提供注册码和备注: python license_manager.py note CE-XXXX-XXXX "备注内容"')
            return
        cmd_note(data, args[1].upper(), " ".join(args[2:]))

    elif "-" in args[0]:
        # 直接传注册码，查看详情
        cmd_info(data, args[0].upper())

    else:
        print(f"  ❌ 未知命令: {cmd}")
        print(__doc__)


if __name__ == "__main__":
    main()
