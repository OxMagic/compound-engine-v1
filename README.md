# 复利引擎 Compound Engine

> BTC事件合约预测辅助工具 | 20+模型集成投票 | 138特征 | 实时K线 | 在线学习

[![powered by](https://img.shields.io/badge/powered%20by-ML%20%2B%20Rule%20Ensemble-blue)](https://github.com/OxMagic/compound-engine-v1)

## 功能特性

- 📊 **实时BTC K线图**（1分钟周期，WebSocket推送）
- 🤖 **20+模型集成投票** — 11种ML模型(RF×3, ET, GB, MLP, LDA, QDA, NB, KNN, LR, LGB) + 8个技术指标规则模型
- 📈 **138维特征工程** — 技术指标 + 订单簿 + OI + 资金费率 + 恐惧贪婪指数 + Hurst/分形维度 + 时间周期特征
- 🧠 **在线学习** — 自动结算 + 自适应权重(滑动窗口50次) + 智能重训(胜率滑坡/方向偏差/置信度下降三重检测)
- ⏱️ **多时间框架** — 5min / 10min / 30min 独立模型，共51个pkl
- 📋 **综合进场评分** — 100分制S/A/B/C/D等级（置信度+一致性+趋势+订单簿+情绪+整数位）
- 🔬 **SHAP特征重要性** — 每次训练后自动计算，可查看Top-N特征排行
- 🎯 **趋势过滤** — EMA21/55 + MACD + ADX + 一致性过滤(<55%跳过)
- 🔊 **语音播报** — Web Speech API，开单/结算/连败暂停自动语音提醒
- 📊 **回测系统** — 历史K线滑动窗口模拟，盈亏曲线+置信度分桶胜率
- 📡 **实时辅助数据** — Binance订单簿、合约OI、资金费率、Alternative.me恐惧贪婪指数

## 使用方法

### 本地运行
```bash
pip install -r requirements.txt
python compound_engine_server.py
# 浏览器自动打开 http://localhost:7788
```

### EXE运行
下载 [Releases](https://github.com/OxMagic/compound-engine-v1/releases) 中的压缩包，解压后双击 `启动复利引擎.exe`

### Render部署
1. Fork本仓库
2. 在 [Render](https://render.com) 创建 Web Service，连接GitHub
3. Build Command: `pip install -r requirements.txt`
4. Start Command: `gunicorn compound_engine_server:app --workers 1 --timeout 120`
5. 环境变量: `PORT=10000`

## 项目结构

```
├── compound_engine_server.py   # 主后端（Flask，所有API）
├── 复利引擎.html               # 前端（单文件，机甲风格）
├── launcher.py                 # EXE启动器（防多开+浏览器管理）
├── Launcher.cs                 # C#原生GUI启动器（零闪动）
├── auto_trade.py               # 自动下单（坐标标定+复利模式）
├── license_manager.py          # 注册码管理工具
├── train_multi_timeframe.py    # 多时间框架训练脚本
├── models/                     # 51个训练好的模型
│   ├── btc_model_*.pkl         # 根目录22个（含10min周期）
│   ├── 5min/                   # 5分钟周期17个模型
│   ├── 30min/                  # 30分钟周期17个模型
│   └── _meta.json              # 训练元数据
├── lib/                        # 本地化前端依赖（字体+Chart.js）
├── assets/                     # Logo等静态资源
├── requirements.txt
├── Procfile
└── build.sh                    # Render构建脚本
```

## 投票机制

- **ML模型权重**: 60%（自适应权重，滑动窗口50次自动调整）
- **规则模型权重**: 40%（趋势/均值回归/动量/量能/波动/突破/形态/情绪）
- **高置信度加成**: 一致性>80%时额外+10%权重
- **动态阈值**: 连胜放宽 / 连败收紧 / 连败5次暂停

## 关于

@OxMagic_

## License

AGPL-3.0
