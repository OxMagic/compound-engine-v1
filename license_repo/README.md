# 复利引擎 注册码管理说明

## 这个仓库是什么

这是一个**公开仓库**，只存放注册码列表（`licenses.json`）。
软件启动时会请求这个文件来验证用户输入的注册码是否有效。

> ⚠️ 这个仓库必须是 **Public（公开）**，否则软件读不到。
> 代码逻辑（compound-engine 仓库）保持 Private，安全。

---

## 如何管理注册码

直接编辑 `licenses.json`：

```json
{
  "你的注册码": {
    "limit": 100,    ← 这个码最多激活多少次
    "used": 0,       ← 已经被激活了多少次（手动更新）
    "active": true,  ← false = 立即作废
    "note": "备注"   ← 自己看的，软件不读这个
  }
}
```

### 常用操作

**发布新码：**
```json
"CE-2026-NEW": {"limit": 100, "used": 0, "active": true, "note": "4月发布"}
```

**作废某个码（比如有人传播）：**
```json
"CE-2026-ALPHA-FREE": {"limit": 200, "used": 47, "active": false, "note": "已作废"}
```

**限量码（比如给50个人）：**
```json
"CE-2026-LIMIT50": {"limit": 50, "used": 0, "active": true, "note": "限50人"}
```

---

## 注意事项

1. `used` 字段目前是**手动维护**的（软件本地不回写GitHub）
   - 如果要精确控制激活次数，需要用 GitHub API 或手动更新
   - 如果只是粗略限制，定期看一下用户反馈再更新即可

2. 软件有**24小时本地缓存**，用户验证成功后24小时内不会重复请求

3. 如果 GitHub 访问慢（国内），可以考虑用 jsDelivr CDN 加速：
   ```
   https://cdn.jsdelivr.net/gh/YOUR_USERNAME/compound-engine-license@main/licenses.json
   ```

---

## 创建步骤

1. GitHub → New Repository → 名字填 `compound-engine-license`
2. 选 **Public**
3. 上传 `licenses.json`
4. 复制 Raw 链接，填到 `auto_trade.py` 的 `LICENSE_URL` 变量里
