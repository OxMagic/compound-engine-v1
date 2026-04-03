"""
复利引擎 - 多时间框架模型训练脚本
===================================
为 5min / 10min / 30min 三个预测周期分别训练独立的ML模型。

用法:
    python train_multi_timeframe.py           # 训练全部3个周期
    python train_multi_timeframe.py 5         # 只训练5分钟周期
    python train_multi_timeframe.py 10 30     # 训练10分钟和30分钟

训练产出:
    models/5min/btc_model_{name}.pkl          # 5分钟周期模型
    models/10min/btc_model_{name}.pkl         # 10分钟周期模型（默认/现有）
    models/30min/btc_model_{name}.pkl         # 30分钟周期模型

作者: @OxMagic_
"""

import os
import sys
import io

# 修复 Windows 控制台 GBK 编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import time
import pickle
import signal
import threading
import numpy as np
import pandas as pd
from datetime import datetime

# ========== 路径 ==========
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = SCRIPT_DIR
MODEL_BASE = os.path.join(BASE_DIR, 'models')

# ========== 配置 ==========
PERIODS = [5, 10, 30]       # 预测周期（分钟/K线根数）
DATA_LIMIT = 1000           # 训练K线数量（1分钟K线）
SYMBOL = "BTCUSDT"
INTERVAL = "1m"

# ========== 信号处理（Ctrl+C 安全退出） ==========
_stop_flag = False

def _sig_handler(sig, frame):
    global _stop_flag
    print("\n[!] 收到中断信号，训练将在当前模型完成后停止...")
    _stop_flag = True

signal.signal(signal.SIGINT, _sig_handler)

# ================================================================
#  从 compound_engine_server.py 复制的核心函数（保持完全一致）
# ================================================================

def safe_get(url, retries=2, timeout=20, **kwargs):
    """带重试的HTTP请求"""
    import requests
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"  [WARN] safe_get failed: {e}")
    return None


def get_klines(limit=1000, interval="1m"):
    """获取Binance K线数据"""
    url = "https://api.binance.com/api/v3/klines"
    r = safe_get(url, params={"symbol": SYMBOL, "interval": interval, "limit": limit})
    if r is None:
        raise Exception("K线获取失败，请检查网络（VPN）")
    data = r.json()
    cols = ['open_time','open','high','low','close','volume','close_time',
            'quote_asset_volume','number_of_trades','taker_buy_base','taker_buy_quote','ignore']
    df = pd.DataFrame(data, columns=cols)
    df = df[['open_time','open','high','low','close','volume']].astype(float)
    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
    return df


def add_period_label(df, period):
    """
    根据周期（分钟数，即K线根数）生成 future label
    与 compound_engine_server.py 完全一致
    """
    df = df.copy()
    future_close = df['close'].shift(-period)
    future_ret = future_close / df['close'] - 1
    min_move = 0.0005
    label = pd.Series(np.nan, index=df.index)
    label[future_ret > min_move]  = 1
    label[future_ret < -min_move] = 0
    df['label'] = label
    return df.dropna(subset=['label'])


def calculate_features(df):
    """
    技术指标特征工程 — 与 compound_engine_server.py 完全一致
    111+ 个特征列
    """
    df = df.copy()

    # --- 收益率 ---
    for p in [1, 3, 5, 10, 15, 20]:
        df[f'ret{p}'] = df['close'].pct_change(p)

    # --- 移动平均 ---
    for w in [3, 5, 8, 13, 21, 34, 55, 89, 144]:
        df[f'ma{w}'] = df['close'].rolling(w).mean()
        df[f'ma{w}_diff'] = (df['close'] - df[f'ma{w}']) / (df[f'ma{w}'] + 1e-9)

    # --- EMA ---
    for span in [8, 13, 21, 34]:
        df[f'ema{span}'] = df['close'].ewm(span=span, adjust=False).mean()
        df[f'ema{span}_diff'] = (df['close'] - df[f'ema{span}']) / (df[f'ema{span}'] + 1e-9)

    # --- MACD ---
    ema12 = df['close'].ewm(span=12, adjust=False).mean()
    ema26 = df['close'].ewm(span=26, adjust=False).mean()
    df['macd'] = ema12 - ema26
    df['macd_sig'] = df['macd'].ewm(span=9, adjust=False).mean()
    df['macd_hist'] = df['macd'] - df['macd_sig']
    df['macd_signal'] = np.where(df['macd'] > df['macd_sig'], 1, -1)
    df['macd_hist_norm'] = df['macd_hist'] / (df['close'] + 1e-9)

    # --- RSI ---
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    RS = roll_up / (roll_down + 1e-9)
    df['rsi'] = 100 - (100 / (1 + RS))
    df['rsi_ma'] = df['rsi'].rolling(5).mean()
    df['rsi_slope'] = df['rsi'].diff(3)

    # --- 随机指标 ---
    low_min = df['low'].rolling(14).min()
    high_max = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

    # --- 布林带 ---
    for period in [20, 50]:
        ma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = ma + 2 * std
        lower = ma - 2 * std
        df[f'bbw_{period}'] = (upper - lower) / (ma + 1e-9)
        df[f'bb_position_{period}'] = (df['close'] - lower) / (upper - lower + 1e-9)
        df[f'bb_squeeze_{period}'] = (std / ma).rolling(5).mean()

    df['bbw'] = df.get('bbw_20', df['close'].rolling(20).std() / (df['close'].rolling(20).mean() + 1e-9))
    df['bb_position'] = df.get('bb_position_20', (df['close'] - df['close'].rolling(20).min()) / (df['close'].rolling(20).max() - df['close'].rolling(20).min() + 1e-9))
    df['bb_squeeze'] = df.get('bb_squeeze_20', (df['close'].rolling(20).std() / (df['close'].rolling(20).mean())).rolling(5).mean())

    # --- ATR ---
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr14'] = tr.rolling(14).mean()
    df['atr_percent'] = df['atr14'] / df['close']
    df['atr_ratio'] = df['atr14'] / (df['atr14'].rolling(20).mean() + 1e-9)

    # --- 成交量 ---
    df['vol_chg'] = df['volume'].pct_change()
    df['vol_ma21'] = df['volume'].rolling(21).mean()
    df['vol_over_ma'] = df['volume'] / (df['vol_ma21'] + 1e-9)
    df['vol_ratio'] = df['volume'].rolling(10).mean() / (df['volume'].rolling(50).mean() + 1e-9)

    # --- 价格动量 ---
    df['price_accel'] = df['close'].diff().diff()
    df['price_velocity'] = df['close'].diff()

    # --- OBV ---
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma_diff'] = df['obv'] - df['obv'].rolling(20).mean()

    # --- VWAP ---
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (typical_price * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    df['vwap_diff'] = df['close'] - df['vwap']
    df['price_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-9)

    # --- Williams %R ---
    df['williams_r'] = 100 * ((df['high'].rolling(14).max() - df['close']) /
                              (df['high'].rolling(14).max() - df['low'].rolling(14).min() + 1e-9))

    # --- CCI ---
    tp = (df['high'] + df['low'] + df['close']) / 3
    sma_tp = tp.rolling(20).mean()
    mad_tp = tp.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    df['cci'] = (tp - sma_tp) / (0.015 * mad_tp + 1e-9)

    # --- ADX ---
    high_diff = df['high'].diff()
    low_diff = df['low'].diff()
    plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
    minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)
    tr = df['high'] - df['low']
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/14).mean() / (tr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/14).mean() / (tr + 1e-9)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    df['adx'] = pd.Series(dx, index=df.index).ewm(alpha=1/14).mean()
    df['adx_trend'] = np.where(df['adx'] > 25, 1, np.where(df['adx'] < 20, -1, 0))

    # --- 价格-成交量相关性 ---
    df['price_vol_corr'] = df['close'].rolling(20).corr(df['volume'])
    df['price_vol_corr_ma'] = df['price_vol_corr'].rolling(5).mean()

    # --- Hurst 指数 ---
    def _hurst(series):
        lags = range(2, min(80, len(series) // 2))
        if len(lags) < 5:
            return 0.5
        tau = [np.std(np.diff(series.values, n=lag)) for lag in lags]
        log_lags = np.log(list(lags))
        log_tau = np.log(tau)
        valid = np.isfinite(log_lags) & np.isfinite(log_tau)
        if valid.sum() < 5:
            return 0.5
        try:
            slope = np.polyfit(log_lags[valid], log_tau[valid], 1)[0]
            return float(np.clip(slope, 0.01, 0.99))
        except:
            return 0.5

    df['hurst'] = df['close'].rolling(100).apply(_hurst, raw=False)

    # --- 分形维度 ---
    def _fractal_dim(series):
        try:
            n = len(series)
            if n < 20:
                return 1.5
            max_k = int(np.floor(np.log2(n))) - 1
            if max_k < 2:
                return 1.5
            sizes = [2**k for k in range(2, max_k + 1)]
            vals = []
            for size in sizes:
                boxes = int(np.floor(n / size))  # floor 而非 ceil，确保能整除
                if boxes < 1:
                    continue
                usable = boxes * size
                if usable > n:
                    boxes -= 1
                    usable = boxes * size
                if boxes < 1 or usable < size:
                    continue
                reshaped = series.values[:usable].reshape(boxes, size)
                rng = reshaped.max(axis=1) - reshaped.min(axis=1)
                s = rng.sum() * (n / usable)
                if s > 0:
                    vals.append((np.log(1.0 / size), np.log(s)))
            if len(vals) < 2:
                return 1.5
            xs, ys = zip(*vals)
            slope = np.polyfit(list(xs), list(ys), 1)[0]
            return float(np.clip(-slope, 1.0, 2.0))
        except Exception:
            return 1.5

    df['fractal_dim'] = df['close'].rolling(200).apply(_fractal_dim, raw=False)

    # --- 市场体制 ---
    ma_short = df['close'].rolling(20).mean()
    ma_mid = df['close'].rolling(50).mean()
    ma_long = df['close'].rolling(100).mean()
    df['market_regime'] = np.where(
        ma_short > ma_mid, np.where(ma_mid > ma_long, 2, 1),
        np.where(ma_short < ma_mid, np.where(ma_mid < ma_long, -2, -1), 0)
    )

    # --- 趋势强度 ---
    df['trend_strength'] = (
        (df['close'] - df['ma21']) / (df['atr14'] + 1e-9)
        if 'ma21' in df.columns and 'atr14' in df.columns else 0
    )

    # --- 波动率体制 ---
    df['volatility_regime'] = df['atr_percent'].rolling(50).apply(
        lambda x: 1 if x.iloc[-1] > x.mean() * 1.2 else (-1 if x.iloc[-1] < x.mean() * 0.8 else 0), raw=False
    )

    # --- 成交量体制 ---
    df['volume_regime'] = df['vol_ratio'].rolling(50).apply(
        lambda x: 1 if x.iloc[-1] > x.mean() * 1.3 else (-1 if x.iloc[-1] < x.mean() * 0.7 else 0), raw=False
    )

    return df


def get_feature_cols(df):
    """获取特征列名（与 compound_engine_server.py 一致）"""
    exclude = {'open_time', 'open', 'high', 'low', 'close', 'volume', 'label'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]


# ================================================================
#  模型训练核心
# ================================================================

def _train_single_model(model, X_train, y_train, X_val, y_val, timeout=120):
    """带超时的单模型训练"""
    import threading

    result = {'model': None, 'acc': 0.0}

    def _target():
        try:
            model.fit(X_train, y_train)
            from sklearn.metrics import accuracy_score
            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            result['model'] = model
            result['acc'] = acc
        except Exception as e:
            print(f"      训练异常: {e}")

    t = threading.Thread(target=_target, daemon=True)
    t.start()
    t.join(timeout=timeout)
    if t.is_alive():
        print(f"      超时（{timeout}s），跳过")
    return result


def train_for_period(period, df_raw):
    """
    为指定周期训练全部模型
    返回: {model_name: model_obj}, scaler, feature_cols, 训练统计
    """
    global _stop_flag
    if _stop_flag:
        return None

    print(f"\n{'='*60}")
    print(f"  训练周期: {period} 分钟")
    print(f"{'='*60}")

    # 标签生成（不同周期 → 不同标签 → 不同模型学到的模式）
    print(f"  [1/5] 生成 {period}min 标签...")
    df_labeled = add_period_label(df_raw, period)
    print(f"        有效样本: {len(df_labeled)}（去噪音后）")

    # 特征工程
    print(f"  [2/5] 计算技术指标特征...")
    df_feat = calculate_features(df_labeled)
    feature_cols = get_feature_cols(df_feat)
    df_feat = df_feat.dropna(subset=feature_cols + ['label'])
    print(f"        特征数: {len(feature_cols)}，最终样本: {len(df_feat)}")

    # 标签分布
    label_counts = df_feat['label'].value_counts()
    total_labels = len(df_feat)
    up_pct = label_counts.get(1, 0) / total_labels * 100
    down_pct = label_counts.get(0, 0) / total_labels * 100
    print(f"        标签分布: UP={up_pct:.1f}% DOWN={down_pct:.1f}%")

    if len(df_feat) < 200:
        print(f"  ⚠ 样本不足（{len(df_feat)} < 200），跳过")
        return None

    X = df_feat[feature_cols].values
    y = df_feat['label'].values

    # 时序分割（不用随机分割，避免未来数据泄露）
    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    print(f"  [3/5] 数据标准化 (RobustScaler)...")
    from sklearn.preprocessing import RobustScaler
    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    # 模型定义
    print(f"  [4/5] 训练模型...")

    model_defs = [
        ('rf1',  'RandomForestClassifier(n_estimators=200, max_depth=8, class_weight=balanced)'),
        ('rf2',  'RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, class_weight=balanced)'),
        ('rf3',  'RandomForestClassifier(n_estimators=150, max_features=sqrt, class_weight=balanced)'),
        ('et',   'ExtraTreesClassifier(n_estimators=200, class_weight=balanced)'),
        ('gb',   'GradientBoostingClassifier(n_estimators=100, max_depth=4)'),
        ('mlp',  'MLPClassifier(hidden_layer_sizes=(128,64,32), max_iter=500, early_stopping=True)'),
        ('lda',  'CalibratedClassifierCV(LinearDiscriminantAnalysis())'),
        ('qda',  'QuadraticDiscriminantAnalysis()'),
        ('nb',   'GaussianNB()'),
        ('knn',  'CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=15))'),
        ('lr',   'LogisticRegression(max_iter=500, class_weight=balanced)'),
    ]

    # 动态导入 + 实例化
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV

    model_instances = [
        ('rf1',  RandomForestClassifier(n_estimators=200, max_depth=8, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('rf2',  RandomForestClassifier(n_estimators=100, max_depth=6, min_samples_leaf=5, class_weight='balanced', random_state=43, n_jobs=-1)),
        ('rf3',  RandomForestClassifier(n_estimators=150, max_features='sqrt', class_weight='balanced', random_state=44, n_jobs=-1)),
        ('et',   ExtraTreesClassifier(n_estimators=200, class_weight='balanced', random_state=42, n_jobs=-1)),
        ('gb',   GradientBoostingClassifier(n_estimators=100, max_depth=4, random_state=42)),
        ('mlp',  MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=500, random_state=42, early_stopping=True, validation_fraction=0.15)),
        ('lda',  CalibratedClassifierCV(LinearDiscriminantAnalysis(), cv=3, method='isotonic')),
        ('qda',  QuadraticDiscriminantAnalysis()),
        ('nb',   GaussianNB()),
        ('knn',  CalibratedClassifierCV(KNeighborsClassifier(n_neighbors=15, n_jobs=-1), cv=3, method='isotonic')),
        ('lr',   LogisticRegression(max_iter=500, class_weight='balanced', random_state=42, n_jobs=-1)),
    ]

    # 可选模型: LightGBM
    try:
        from lightgbm import LGBMClassifier
        model_instances.append(('lgb1', LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)))
        model_instances.append(('lgb2', LGBMClassifier(n_estimators=100, num_leaves=31, random_state=43, n_jobs=-1, verbose=-1)))
    except ImportError:
        print("        LightGBM 未安装，跳过 lgb1/lgb2")

    # 可选模型: XGBoost
    try:
        import xgboost as xgb
        model_instances.append(('xgb1', xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss')))
        model_instances.append(('xgb2', xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=43, n_jobs=-1, verbosity=0, eval_metric='logloss')))
    except ImportError:
        print("        XGBoost 未安装，跳过 xgb1/xgb2")

    # 可选模型: CatBoost
    try:
        from catboost import CatBoostClassifier
        model_instances.append(('cat1', CatBoostClassifier(iterations=200, random_state=42, verbose=False)))
        model_instances.append(('cat2', CatBoostClassifier(iterations=100, depth=4, random_state=43, verbose=False)))
    except ImportError:
        print("        CatBoost 未安装，跳过 cat1/cat2")

    # 逐个训练
    trained = {}
    stats = {}
    total = len(model_instances)

    for i, (name, model) in enumerate(model_instances):
        if _stop_flag:
            print(f"\n  ⚠ 收到停止信号，终止训练")
            break

        print(f"    [{i+1:2d}/{total}] {name:6s} ...", end='', flush=True)
        try:
            res = _train_single_model(model, X_train_s, y_train, X_val_s, y_val, timeout=120)
            if res['model'] is not None:
                trained[name] = res['model']
                stats[name] = res['acc']
                print(f" ✓ acc={res['acc']:.4f}")
            else:
                stats[name] = 0.0
                print(f" ✗ 失败")
        except Exception as e:
            stats[name] = 0.0
            print(f" ✗ 异常: {e}")

    if not trained:
        print(f"  ⚠ 没有成功训练的模型，跳过")
        return None

    # 训练统计
    avg_acc = np.mean(list(stats.values()))
    print(f"\n  训练统计:")
    print(f"    成功: {len(trained)}/{total}")
    print(f"    平均验证准确率: {avg_acc:.4f}")
    top3 = sorted(stats.items(), key=lambda x: -x[1])[:3]
    print(f"    Top 3: {', '.join(f'{n}={a:.4f}' for n, a in top3)}")

    return {
        'models': trained,
        'scaler': scaler,
        'features': feature_cols,
        'stats': stats,
        'avg_acc': avg_acc,
        'samples': len(df_feat),
        'label_dist': {'up': up_pct, 'down': down_pct},
    }


def save_period_models(period, train_result):
    """保存指定周期的所有模型"""
    model_dir = os.path.join(MODEL_BASE, f'{period}min')
    os.makedirs(model_dir, exist_ok=True)

    saved_count = 0
    for name, model in train_result['models'].items():
        path = os.path.join(model_dir, f'btc_model_{name}.pkl')
        try:
            with open(path, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'scaler': train_result['scaler'],
                    'features': train_result['features'],
                    'period': period,
                    'trained_at': datetime.now().isoformat(),
                }, f)
            saved_count += 1
        except Exception as e:
            print(f"    保存 {name} 失败: {e}")

    # 保存训练元数据
    meta_path = os.path.join(model_dir, '_meta.json')
    import json
    meta = {
        'period': period,
        'trained_at': datetime.now().isoformat(),
        'model_count': saved_count,
        'feature_count': len(train_result['features']),
        'sample_count': train_result['samples'],
        'avg_accuracy': train_result['avg_acc'],
        'label_distribution': train_result['label_dist'],
        'models': {name: {'acc': acc} for name, acc in train_result['stats'].items()},
    }
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    return saved_count, model_dir


# ================================================================
#  主流程
# ================================================================

def main():
    global _stop_flag

    print("╔══════════════════════════════════════════════════════╗")
    print("║     复利引擎 - 多时间框架模型训练器                  ║")
    print("║     @OxMagic_                                       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据量: {DATA_LIMIT} 根 1分钟K线")
    print(f"  Python: {sys.version.split()[0]}")

    # 解析命令行参数
    if len(sys.argv) > 1:
        try:
            periods = [int(x) for x in sys.argv[1:]]
            print(f"  周期: {periods} (命令行指定)")
        except ValueError:
            print(f"  ⚠ 无效参数，使用默认周期: {PERIODS}")
            periods = PERIODS
    else:
        periods = PERIODS
        print(f"  周期: {periods}")

    # Step 1: 获取K线数据（只需获取一次，所有周期共享）
    print(f"\n{'─'*60}")
    print(f"  获取 {DATA_LIMIT} 根1分钟K线数据（需VPN连接Binance）...")
    print(f"{'─'*60}")
    t0 = time.time()

    try:
        df_raw = get_klines(DATA_LIMIT)
    except Exception as e:
        print(f"\n  ❌ K线获取失败: {e}")
        print(f"  请确认:")
        print(f"    1. VPN 已开启")
        print(f"    2. 能正常访问 api.binance.com")
        sys.exit(1)

    print(f"  ✓ 获取 {len(df_raw)} 根K线，耗时 {time.time()-t0:.1f}s")
    print(f"  时间范围: {df_raw['open_time'].iloc[0]} ~ {df_raw['open_time'].iloc[-1]}")

    # Step 2: 逐周期训练
    results = {}
    for period in periods:
        if _stop_flag:
            break

        t_start = time.time()
        train_result = train_for_period(period, df_raw)

        if train_result is None:
            continue

        # 保存模型
        print(f"  [5/5] 保存模型...")
        saved_count, model_dir = save_period_models(period, train_result)
        elapsed = time.time() - t_start

        results[period] = {
            'saved_count': saved_count,
            'model_dir': model_dir,
            'avg_acc': train_result['avg_acc'],
            'samples': train_result['samples'],
            'elapsed': elapsed,
        }

        print(f"  ✓ 完成！{saved_count} 个模型 → {model_dir}（耗时 {elapsed:.1f}s）")

    # Step 3: 总结
    print(f"\n{'═'*60}")
    print(f"  训练总结")
    print(f"{'═'*60}")
    total_models = 0
    total_time = 0

    for period, info in results.items():
        print(f"  {period:3d}min: {info['saved_count']:2d} 个模型, "
              f"准确率={info['avg_acc']:.4f}, "
              f"{info['samples']} 样本, "
              f"耗时 {info['elapsed']:.1f}s")
        print(f"         → {info['model_dir']}")
        total_models += info['saved_count']
        total_time += info['elapsed']

    print(f"\n  总计: {len(results)} 个周期, {total_models} 个模型, 耗时 {total_time:.1f}s")

    if not results:
        print(f"\n  ❌ 没有成功训练的周期")
        sys.exit(1)

    print(f"\n  ✅ 所有模型已保存到 models/ 目录下对应子文件夹")
    print(f"  下一步: 在 compound_engine_server.py 中集成多周期预测逻辑")


if __name__ == '__main__':
    main()
