"""
复利引擎后端服务 - compound_engine_server.py
加载真实21个sklearn模型，提供预测API
"""

import os
import sys
import pickle
import threading
import time
import math
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ========== 路径配置 ==========
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
# 日志写在项目目录下，兼容云端（不用Windows绝对路径）
LOG_CSV   = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'compound_engine_log.csv')

SYMBOL    = "BTCUSDT"
INTERVAL  = "1m"
DATA_LIMIT = 1000  # 使用1000根K线训练，样本更充足（Binance API上限）

# ========== K线缓存（60秒内复用，避免重复请求Binance） ==========
_klines_cache = {'df': None, 'ts': 0, 'limit': 0}
_klines_cache_lock = threading.Lock()

# ========== 模型文件映射 ==========
MODEL_FILES = {
    'rf1':  'btc_model_rf1.pkl',
    'rf2':  'btc_model_rf2.pkl',
    'rf3':  'btc_model_rf3.pkl',
    'lgb1': 'btc_model_lgb1.pkl',
    'lgb2': 'btc_model_lgb2.pkl',
    'xgb1': 'btc_model_xgb1.pkl',
    'xgb2': 'btc_model_xgb2.pkl',
    'cat1': 'btc_model_cat1.pkl',
    'cat2': 'btc_model_cat2.pkl',
    'ada':  'btc_model_ada.pkl',
    'gb':   'btc_model_gb.pkl',
    'et':   'btc_model_et.pkl',
    'svm1': 'btc_model_svm1.pkl',
    'svm2': 'btc_model_svm2.pkl',
    'knn':  'btc_model_knn.pkl',
    'mlp':  'btc_model_mlp.pkl',
    'lda':  'btc_model_lda.pkl',
    'qda':  'btc_model_qda.pkl',
    'nb':   'btc_model_nb.pkl',
    'hgb':  'btc_model_hgb.pkl',
    'ensemble': 'btc_model_ensemble.pkl',
}

# ========== Flask App ==========
app = Flask(__name__)
CORS(app)

# ========== 全局状态 ==========
state = {
    'models': {},
    'scalers': {},
    'features': [],
    'per_model_features': {},   # 每个模型自己保存的特征列
    'loaded': False,
    'training': False,
    'train_progress': 0,
    'train_status': '',
    'last_prediction': None,
    'last_price': None,
    'last_signal': None,   # 'up' / 'down'，供自动下单复利模块使用
    # ===== 自适应权重（在线学习）=====
    'adaptive_weights': {},     # {model_name: float}，动态权重，初始1.0
    'model_perf_window': {},    # {model_name: deque(maxlen=50)}，近50次正确/错误
    'pending_model_preds': None,# 最近一次预测的各模型结果，等待结果回调更新权重
}
state_lock = threading.Lock()

# ========== 自适应权重工具函数 ==========
from collections import deque

ADAPTIVE_WINDOW = 50      # 统计近50次
ADAPTIVE_MIN_W  = 0.1     # 最低权重
ADAPTIVE_MAX_W  = 3.0     # 最高权重
ADAPTIVE_LR     = 0.15    # 学习率（每次更新的步长）

def get_adaptive_weight(model_name: str) -> float:
    """获取模型当前动态权重，未初始化则返回1.0"""
    with state_lock:
        return state['adaptive_weights'].get(model_name, 1.0)

def update_adaptive_weights(model_preds: dict, actual: int):
    """
    根据本轮实际结果更新所有模型权重
    model_preds: {name: {'pred': 0/1, 'conf': float}}
    actual: 1=涨赢了 / 0=跌赢了
    """
    with state_lock:
        for name, r in model_preds.items():
            correct = int(r['pred'] == actual)
            # 维护近期窗口
            if name not in state['model_perf_window']:
                state['model_perf_window'][name] = deque(maxlen=ADAPTIVE_WINDOW)
            state['model_perf_window'][name].append(correct)
            # 用近期胜率重算权重
            window = state['model_perf_window'][name]
            if len(window) >= 5:  # 至少5条数据再调整
                win_rate = sum(window) / len(window)
                # 胜率 -> 权重：50%对应1.0，75%对应2.5，25%对应0.1
                new_w = ADAPTIVE_MIN_W + (ADAPTIVE_MAX_W - ADAPTIVE_MIN_W) * max(0, (win_rate - 0.3) / 0.5)
                new_w = max(ADAPTIVE_MIN_W, min(ADAPTIVE_MAX_W, new_w))
                state['adaptive_weights'][name] = round(new_w, 3)

# ========== 模型加载 ==========
def load_all_models():
    loaded = {}
    scalers = {}
    features_found = []
    per_model_features = {}   # 每个模型自己的特征列表

    for name, fname in MODEL_FILES.items():
        path = os.path.join(MODEL_DIR, fname)
        if os.path.exists(path):
            try:
                with open(path, 'rb') as f:
                    data = pickle.load(f)
                if isinstance(data, dict):
                    loaded[name] = data.get('model')
                    scalers[name] = data.get('scaler')
                    m_features = data.get('features', [])
                    if m_features:
                        per_model_features[name] = m_features
                    if not features_found and m_features:
                        features_found = m_features
                else:
                    loaded[name] = data
            except Exception as e:
                print(f"[WARN] 加载模型 {name} 失败: {e}")
    
    with state_lock:
        state['models'] = loaded
        state['scalers'] = scalers
        state['features'] = features_found
        state['per_model_features'] = per_model_features
        state['loaded'] = True
    
    print(f"[INFO] 已加载 {len(loaded)} 个模型: {list(loaded.keys())}")
    return loaded

# ========== 数据获取 ==========
def safe_get(url, retries=2, **kwargs):
    """带重试的HTTP请求，超时20秒"""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=20, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"[WARN] safe_get failed after {retries+1} attempts: {e}")
    return None

def get_klines(limit=500, interval=None):
    if interval is None:
        interval = INTERVAL
    # 60秒内若已有足量缓存，直接复用（大幅减少Binance请求次数）
    cache_key = interval
    with _klines_cache_lock:
        now = time.time()
        if (_klines_cache.get('df_' + cache_key) is not None
                and now - _klines_cache.get('ts_' + cache_key, 0) < 60
                and _klines_cache.get('limit_' + cache_key, 0) >= limit):
            print(f"[CACHE] 复用K线缓存 {interval}（{int(now - _klines_cache['ts_' + cache_key])}秒前）")
            return _klines_cache['df_' + cache_key].tail(limit).copy()

    sources = [
        ("binance", f"https://api.binance.com/api/v3/klines?symbol={SYMBOL}&interval={interval}&limit={limit}"),
        ("okx",     None),
        ("bybit",   None),
    ]
    for source, url in sources:
        try:
            if source == "binance":
                r = safe_get(url)
                if r is None: continue
                data = r.json()
                df = pd.DataFrame(data, columns=[
                    'open_time','open','high','low','close','volume',
                    'close_time','quote_asset_volume','number_of_trades',
                    'taker_buy_base','taker_buy_quote','ignore'])
                df = df[['open_time','open','high','low','close','volume']].astype(float)
                df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                # 更新缓存
                with _klines_cache_lock:
                    _klines_cache['df_' + cache_key] = df.copy()
                    _klines_cache['ts_' + cache_key] = time.time()
                    _klines_cache['limit_' + cache_key] = limit
                return df
            elif source == "okx":
                okx_bar = {'1m':'1m','5m':'5m','15m':'15m','1h':'1H'}.get(interval, '1m')
                r = safe_get("https://www.okx.com/api/v5/market/history-candles",
                             params={"instId": "BTC-USDT", "bar": okx_bar, "limit": limit})
                if r is None: continue
                j = r.json()
                if 'data' in j:
                    arr = j['data']
                    df = pd.DataFrame(arr, columns=['open_time','open','high','low','close','volume'])
                    df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
                    df = df[['open_time','open','high','low','close','volume']].astype(float)
                    with _klines_cache_lock:
                        _klines_cache['df_' + cache_key] = df.copy()
                        _klines_cache['ts_' + cache_key] = time.time()
                        _klines_cache['limit_' + cache_key] = limit
                    return df
        except Exception:
            continue
    return None

def get_price():
    sources = [
        f"https://api.binance.com/api/v3/ticker/price?symbol={SYMBOL}",
        f"https://api.binance.com/api/v3/ticker/24hr?symbol={SYMBOL}",
    ]
    for url in sources:
        try:
            r = safe_get(url)
            if r:
                j = r.json()
                p = float(j.get('price') or j.get('lastPrice', 'nan'))
                if not math.isnan(p):
                    return p
        except Exception:
            continue
    return None

# ========== 特征工程（完整版，兼容原版 pkl 特征集） ==========
def calculate_features(df):
    df = df.copy()

    # --- 收益率 ---
    for period in [1, 3, 5, 10, 15, 20]:
        df[f'ret{period}'] = df['close'].pct_change(period)

    # --- 移动均线 ---
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

    # --- RSI（原版命名：rsi / rsi_ma / rsi_slope） ---
    delta = df['close'].diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    rs14 = up.rolling(14).mean() / (down.rolling(14).mean() + 1e-9)
    df['rsi'] = 100 - 100 / (1 + rs14)
    df['rsi_ma'] = df['rsi'].rolling(5).mean()
    df['rsi_slope'] = df['rsi'].diff(3)
    # 多周期 RSI
    for w in [6, 21]:
        rs = up.rolling(w).mean() / (down.rolling(w).mean() + 1e-9)
        df[f'rsi{w}'] = 100 - 100 / (1 + rs)

    # --- Stochastic (14,3) ---
    low14  = df['low'].rolling(14).min()
    high14 = df['high'].rolling(14).max()
    df['stoch_k'] = 100 * (df['close'] - low14) / (high14 - low14 + 1e-9)
    df['stoch_d'] = df['stoch_k'].rolling(3).mean()
    df['stoch_diff'] = df['stoch_k'] - df['stoch_d']

    # --- Bollinger Bands（原版命名：bbw / bb_position / bb_squeeze + 带数字版） ---
    for w in [5, 10, 20]:
        mid = df['close'].rolling(w).mean()
        std = df['close'].rolling(w).std()
        df[f'bb_upper{w}'] = mid + 2 * std
        df[f'bb_lower{w}'] = mid - 2 * std
        df[f'bb_width{w}']  = (df[f'bb_upper{w}'] - df[f'bb_lower{w}']) / (mid + 1e-9)
        df[f'bb_pos{w}']    = (df['close'] - df[f'bb_lower{w}']) / (df[f'bb_upper{w}'] - df[f'bb_lower{w}'] + 1e-9)
        df[f'bb_squeeze{w}'] = (std / mid).rolling(5).mean()
    # 原版无数字版别名
    df['bbw']         = df['bb_width20']
    df['bb_position'] = df['bb_pos20']
    df['bb_squeeze']  = df['bb_squeeze20']

    # --- ATR ---
    hl  = df['high'] - df['low']
    hpc = (df['high'] - df['close'].shift()).abs()
    lpc = (df['low']  - df['close'].shift()).abs()
    tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
    df['atr14']      = tr.rolling(14).mean()
    df['atr_percent'] = df['atr14'] / (df['close'] + 1e-9)
    df['atr_ratio']   = df['atr14'] / (df['atr14'].rolling(20).mean() + 1e-9)

    # --- 成交量 ---
    for w in [5, 10, 20]:
        df[f'vol_ma{w}']    = df['volume'].rolling(w).mean()
        df[f'vol_ratio{w}'] = df['volume'] / (df[f'vol_ma{w}'] + 1e-9)
    df['vol_chg']    = df['volume'].pct_change()
    df['vol_ma21']   = df['volume'].rolling(21).mean()
    df['vol_over_ma'] = df['volume'] / (df['vol_ma21'] + 1e-9)
    df['vol_ratio']  = df['volume'].rolling(10).mean() / (df['volume'].rolling(50).mean() + 1e-9)

    # --- 价格加速度 / 速度 ---
    df['price_accel']    = df['close'].diff().diff()
    df['price_velocity'] = df['close'].diff()

    # --- OBV ---
    df['obv'] = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    df['obv_ma_diff'] = df['obv'] - df['obv'].rolling(20).mean()

    # --- VWAP ---
    tp = (df['high'] + df['low'] + df['close']) / 3
    df['vwap'] = (tp * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-9)
    df['vwap_diff'] = df['close'] - df['vwap']
    df['price_vwap_ratio'] = df['close'] / (df['vwap'] + 1e-9)

    # --- Williams %R ---
    df['williams_r'] = 100 * (high14 - df['close']) / (high14 - low14 + 1e-9)

    # --- CCI ---
    cci_tp = (df['high'] + df['low'] + df['close']) / 3
    cci_ma = cci_tp.rolling(20).mean()
    cci_md = cci_tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    df['cci'] = (cci_tp - cci_ma) / (0.015 * cci_md + 1e-9)

    # --- K线形态 ---
    df['hl_range']     = (df['high'] - df['low']) / (df['close'] + 1e-9)
    df['oc_range']     = (df['close'] - df['open']) / (df['open'] + 1e-9)
    df['upper_shadow'] = (df['high'] - df[['open','close']].max(axis=1)) / (df['close'] + 1e-9)
    df['lower_shadow'] = (df[['open','close']].min(axis=1) - df['low']) / (df['close'] + 1e-9)

    # --- Lag 特征 ---
    for lag in [1, 2, 3, 5]:
        df[f'close_lag{lag}']  = df['close'].shift(lag)
        df[f'volume_lag{lag}'] = df['volume'].shift(lag)

    # --- ADX (14) ---
    plus_dm  = df['high'].diff().clip(lower=0)
    minus_dm = (-df['low'].diff()).clip(lower=0)
    # 当 +DM > -DM 时才计数
    plus_dm  = plus_dm.where(plus_dm > minus_dm, 0)
    minus_dm = minus_dm.where(minus_dm > plus_dm, 0)
    atr14_adx = tr.rolling(14).mean()
    plus_di  = 100 * plus_dm.rolling(14).mean()  / (atr14_adx + 1e-9)
    minus_di = 100 * minus_dm.rolling(14).mean() / (atr14_adx + 1e-9)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
    df['adx'] = dx.rolling(14).mean()
    df['adx_trend'] = (df['adx'] > 25).astype(int)

    # --- 价量相关性 ---
    df['price_vol_corr']    = df['close'].rolling(20).corr(df['volume'])
    df['price_vol_corr_ma'] = df['price_vol_corr'].rolling(5).mean()

    # --- 市场制度指标 ---
    ret20_std = df['ret1'].rolling(20).std()
    df['volatility_regime'] = (ret20_std > ret20_std.rolling(100).mean()).astype(int)
    df['trend_strength']    = df['adx'] / 100.0
    df['market_regime']     = np.where(df['ma21_diff'] > 0, 1, -1)
    df['volume_regime']     = (df['vol_over_ma'] > 1.5).astype(int)

    # --- hurts / fractal_dim（占位，填0） ---
    df['hurts']       = 0.0
    df['fractal_dim'] = 0.0

    # --- orderbook / OI / funding（无额外API，填0） ---
    df['orderbook_imbalance'] = 0.0
    df['orderbook_pressure']  = 0.0
    df['open_interest']       = 0.0
    df['funding_rate']        = 0.0

    # 默认 label（predict 不使用，训练会被 add_period_label 覆盖）
    df['label'] = (df['close'].shift(-1) > df['close']).astype(int)
    return df



def add_period_label(df, period):
    """
    根据周期（分钟数，即K线根数）生成 future label
    策略：未来 period 根后的价格相对当前涨跌
        - 涨幅 > +0.05% → label=1（涨）
        - 跌幅 > -0.05% → label=0（跌）
        - 幅度过小的噪音行情丢弃
    不将 future_ret 保留为特征列，彻底杜绝特征泄露
    """
    df = df.copy()
    future_close = df['close'].shift(-period)
    future_ret = future_close / df['close'] - 1
    # 只保留有明确方向的样本（过滤±0.05%以内的噪音）
    min_move = 0.0005
    label = pd.Series(np.nan, index=df.index)
    label[future_ret > min_move]  = 1
    label[future_ret < -min_move] = 0
    df['label'] = label
    # 不将 future_ret 写入 df，避免被 get_feature_cols 拾取
    return df.dropna(subset=['label'])

def get_feature_cols(df):
    exclude = {'open_time', 'open', 'high', 'low', 'close', 'volume', 'label'}
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.int64, float, int]]

# ========== 整数触发检测 ==========
def is_near_integer(price, tol=0.1):
    tol_abs = price * tol / 100
    for base in [50, 100, 25, 10]:
        nearest = round(price / base) * base
        if abs(price - nearest) <= tol_abs:
            return True, str(int(nearest))
    return False, None

# ========== 核心预测 ==========
def run_prediction(df, price, precomputed=False):
    """
    df: K线 DataFrame
    precomputed: True 时表示 df 已经过 add_period_label 处理，
                 直接在其上补充技术指标，不覆盖 label
    """
    with state_lock:
        models = state['models']
        scalers = state['scalers']
        saved_features = state['features']
        per_model_features_map = state.get('per_model_features', {})
    
    if not models:
        return None, 0.0, {}

    if precomputed:
        # df 已有 label 列，只补充技术指标特征，保留 label
        saved_label = df['label'].copy() if 'label' in df.columns else None
        df_feat = calculate_features(df)
        if saved_label is not None:
            df_feat['label'] = saved_label.values[:len(df_feat)] if len(saved_label) >= len(df_feat) else saved_label
    else:
        df_feat = calculate_features(df)

    # label列仅作标签不作特征
    LEAK_COLS = {'label', 'future_ret', 'future_close', 'future_high', 'future_low'}

    # 全局feature_cols仅作备用
    global_feature_cols = get_feature_cols(df_feat)

    votes_up = 0
    votes_down = 0
    model_results = {}
    conf_sum = 0.0
    n_voted = 0
    # 记录各模型原始方向（用于一致性计算）
    up_count_raw = 0
    dn_count_raw = 0

    for name, model in models.items():
        if model is None:
            continue
        try:
            scaler = scalers.get(name)
            # 每个模型用自己pkl里保存的特征列，若没有则用全局
            per_model_features = per_model_features_map.get(name)
            if per_model_features:
                feat_cols = [c for c in per_model_features
                             if c not in LEAK_COLS and c in df_feat.columns]
            elif saved_features:
                feat_cols = [c for c in saved_features
                             if c not in LEAK_COLS and c in df_feat.columns]
            else:
                feat_cols = global_feature_cols

            row = df_feat[feat_cols].dropna().iloc[-1:] if len(df_feat) > 0 else None
            if row is None or row.empty:
                continue
            Xs = row.values
            if scaler:
                Xs = scaler.transform(Xs)
            
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(Xs)[0]
                pred = int(np.argmax(proba))
                conf = float(proba[pred])
            else:
                pred = int(model.predict(Xs)[0])
                conf = 0.6
            
            model_results[name] = {'pred': pred, 'conf': conf}
            
            # 动态自适应权重 × 高置信度加成（二者相乘）
            adaptive_w = get_adaptive_weight(name)
            conf_bonus = 1.5 if conf > 0.65 else 1.0
            weight = adaptive_w * conf_bonus
            if pred == 1:
                votes_up += conf * weight
                up_count_raw += 1
            else:
                votes_down += conf * weight
                dn_count_raw += 1
            conf_sum += conf
            n_voted += 1
        except Exception as e:
            print(f"[PRED ERR] {name}: {e}")

    if n_voted == 0:
        return None, 0.0, {}

    total_votes = votes_up + votes_down
    if total_votes == 0:
        return None, 0.0, {}

    final_pred = 1 if votes_up >= votes_down else 0
    final_conf = votes_up / total_votes if final_pred == 1 else votes_down / total_votes
    
    # 一致性指数：方向一致的模型比例（0~1，越高越可靠）
    consensus = up_count_raw / n_voted if final_pred == 1 else dn_count_raw / n_voted
    # 将一致性融入最终置信度（低一致性时拉低置信度）
    # 一致性<50%不可能（因为final_pred就是多数方向），最低约50%
    # 一致性>80%才算强信号
    final_conf = final_conf * (0.5 + 0.5 * consensus)

    return final_pred, float(final_conf), model_results


# ========== 趋势过滤器 ==========
def get_trend_filter(df):
    """
    综合趋势过滤，返回:
      trend_dir: 1=趋势向上 / -1=趋势向下 / 0=无明确趋势
      trend_strength: 0.0~1.0（趋势强度，越大越可信）
      details: dict，调试用
    """
    if df is None or len(df) < 55:
        return 0, 0.0, {}

    try:
        c = df['close'].astype(float)
        h = df['high'].astype(float)
        l = df['low'].astype(float)

        # --- EMA 方向 ---
        ema21 = c.ewm(span=21, adjust=False).mean().iloc[-1]
        ema55 = c.ewm(span=55, adjust=False).mean().iloc[-1]
        price_now = c.iloc[-1]
        ema_dir = 1 if (price_now > ema21 > ema55) else (-1 if (price_now < ema21 < ema55) else 0)

        # --- MACD ---
        ema12 = c.ewm(span=12, adjust=False).mean()
        ema26 = c.ewm(span=26, adjust=False).mean()
        macd  = ema12 - ema26
        macd_sig = macd.ewm(span=9, adjust=False).mean()
        macd_dir = 1 if macd.iloc[-1] > macd_sig.iloc[-1] else -1

        # --- ADX（趋势有无，>20才有效） ---
        plus_dm  = h.diff().clip(lower=0)
        minus_dm = (-l.diff()).clip(lower=0)
        plus_dm2  = plus_dm.where(plus_dm > minus_dm, 0.0)
        minus_dm2 = minus_dm.where(minus_dm > plus_dm, 0.0)
        hl  = h - l
        hpc = (h - c.shift()).abs()
        lpc = (l - c.shift()).abs()
        tr  = pd.concat([hl, hpc, lpc], axis=1).max(axis=1)
        atr14 = tr.rolling(14).mean()
        plus_di  = 100 * plus_dm2.rolling(14).mean()  / (atr14 + 1e-9)
        minus_di = 100 * minus_dm2.rolling(14).mean() / (atr14 + 1e-9)
        dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-9)
        adx = dx.rolling(14).mean().iloc[-1]
        has_trend = adx > 20  # ADX>20认为有趋势

        # --- 综合投票 ---
        score = ema_dir + macd_dir  # -2 ~ +2
        if score >= 2:
            trend_dir = 1
        elif score <= -2:
            trend_dir = -1
        elif score == 1:
            trend_dir = 1
        elif score == -1:
            trend_dir = -1
        else:
            trend_dir = 0

        # 强度 = ADX归一化（最大100）
        trend_strength = min(adx / 50.0, 1.0) if has_trend else 0.2

        details = {
            'ema_dir': int(ema_dir), 'macd_dir': int(macd_dir),
            'adx': round(float(adx), 1), 'has_trend': bool(has_trend),
            'score': int(score),
        }
        return int(trend_dir), float(trend_strength), details
    except Exception as e:
        print(f"[TREND] 计算失败: {e}")
        return 0, 0.0, {}

# ========== 训练函数 ==========
def train_models_thread(period=10):
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
    from sklearn.preprocessing import RobustScaler
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
    from sklearn.naive_bayes import GaussianNB
    from sklearn.calibration import CalibratedClassifierCV

    def update_progress(p, msg):
        with state_lock:
            state['train_progress'] = p
            state['train_status'] = msg
        print(f"[TRAIN] {p}% {msg}")

    update_progress(5, "获取K线数据...")
    df = get_klines(DATA_LIMIT)
    if df is None or len(df) < 200:
        with state_lock:
            state['training'] = False
            state['train_status'] = '数据获取失败'
        return

    update_progress(15, f"计算技术指标特征 (周期={period}min)...")
    # 先用 add_period_label 生成对应周期的 label，再计算技术指标
    df_labeled = add_period_label(df, period)
    df_feat = calculate_features(df_labeled)
    feature_cols = get_feature_cols(df_feat)
    df_feat = df_feat.dropna(subset=feature_cols + ['label'])
    
    X = df_feat[feature_cols].values
    y = df_feat['label'].values

    split_idx = int(len(X) * 0.8)
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    scaler = RobustScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    update_progress(25, "训练基础模型组...")

    model_defs = [
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

    # 尝试加载可选模型
    try:
        from lightgbm import LGBMClassifier
        model_defs.append(('lgb1', LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)))
        model_defs.append(('lgb2', LGBMClassifier(n_estimators=100, num_leaves=31, random_state=43, n_jobs=-1, verbose=-1)))
    except Exception:
        pass
    try:
        import xgboost as xgb
        model_defs.append(('xgb1', xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss')))
        model_defs.append(('xgb2', xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=43, n_jobs=-1, verbosity=0, eval_metric='logloss')))
    except Exception:
        pass
    try:
        from catboost import CatBoostClassifier
        model_defs.append(('cat1', CatBoostClassifier(iterations=200, random_state=42, verbose=False)))
        model_defs.append(('cat2', CatBoostClassifier(iterations=100, depth=4, random_state=43, verbose=False)))
    except Exception:
        pass

    trained = {}
    total = len(model_defs)
    for i, (name, model) in enumerate(model_defs):
        pct = 25 + int(55 * i / total)
        update_progress(pct, f"训练 {name.upper()} ({i+1}/{total})...")
        try:
            model.fit(X_train_s, y_train)
            acc = float(np.mean(model.predict(X_val_s) == y_val))
            trained[name] = model
            print(f"  {name}: val_acc={acc:.3f}")
        except Exception as e:
            print(f"  {name} 失败: {e}")

    update_progress(82, "保存模型文件...")
    os.makedirs(MODEL_DIR, exist_ok=True)
    for name, model in trained.items():
        path = os.path.join(MODEL_DIR, f'btc_model_{name}.pkl')
        try:
            with open(path, 'wb') as f:
                pickle.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, f)
        except Exception as e:
            print(f"  保存 {name} 失败: {e}")

    update_progress(95, "重新加载模型到内存...")
    new_models = {}
    new_scalers = {}
    for name in trained:
        path = os.path.join(MODEL_DIR, f'btc_model_{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            new_models[name] = d.get('model')
            new_scalers[name] = d.get('scaler')

    with state_lock:
        state['models'] = new_models
        state['scalers'] = new_scalers
        state['features'] = feature_cols
        state['loaded'] = True
        state['training'] = False
        state['train_progress'] = 100
        state['train_status'] = f'训练完成，{len(new_models)} 个模型已加载'

    print(f"[TRAIN] 完成，训练了 {len(trained)} 个模型")

# ========== API 路由 ==========

@app.route('/', methods=['GET'])
def index():
    """直接提供前端HTML页面，用户访问根路径就能用"""
    html_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '复利引擎.html')
    if os.path.exists(html_path):
        return send_file(html_path)
    return "复利引擎 - HTML文件未找到", 404

@app.route('/lib/<path:filename>', methods=['GET'])
def serve_lib(filename):
    """提供 lib/ 目录下的静态文件（lightweight-charts.js 等）"""
    lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'lib')
    file_path = os.path.join(lib_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    return "文件未找到", 404

@app.route('/api/status', methods=['GET'])
def api_status():
    with state_lock:
        n = len([m for m in state['models'].values() if m is not None])
        return jsonify({
            'loaded': state['loaded'],
            'model_count': n,
            'training': state['training'],
            'train_progress': state['train_progress'],
            'train_status': state['train_status'],
        })

@app.route('/api/price', methods=['GET'])
def api_price():
    p = get_price()
    if p is None:
        return jsonify({'error': '获取价格失败'}), 503
    near, key = is_near_integer(p)
    return jsonify({'price': p, 'near_integer': near, 'key_level': key})

@app.route('/api/predict', methods=['GET'])
def api_predict():
    with state_lock:
        if not state['loaded'] or not state['models']:
            return jsonify({'error': '模型未加载'}), 503

    # 获取周期参数（前端传来的分钟数，默认10）
    try:
        period = int(request.args.get('period', 10))
    except Exception:
        period = 10
    # 只允许 5 / 10 / 30
    if period not in (5, 10, 30):
        period = 10

    price = get_price()
    if price is None:
        return jsonify({'error': '无法获取价格'}), 503

    # 固定拉取300根，足够计算所有特征，且响应最快
    klines_need = 300
    df = get_klines(klines_need)
    if df is None or len(df) < 100:
        return jsonify({'error': '无法获取K线数据'}), 503

    # 直接用原始 df 做特征提取预测（period 仅影响训练时标签，不影响推理）
    # 真正的周期感知：不同 period 会在重训时用对应的 add_period_label 生成标签
    pred, conf, model_results = run_prediction(df, price)
    if pred is None:
        return jsonify({'error': '预测失败，模型无输出'}), 503

    # ---- 趋势过滤 ----
    trend_dir, trend_strength, trend_details = get_trend_filter(df)
    # 与趋势方向一致 → 置信度加权提升
    # 与趋势方向相反 → 置信度大幅折扣（逆势信号可靠性低）
    pred_dir = 1 if pred == 1 else -1
    if trend_dir != 0 and pred_dir != trend_dir:
        # 逆趋势：置信度折扣50%，且标记警告
        conf_adjusted = conf * (1.0 - 0.5 * trend_strength)
        against_trend = True
    else:
        conf_adjusted = conf
        against_trend = False

    near, key = is_near_integer(price)

    # 各模型票数统计
    up_count = sum(1 for r in model_results.values() if r['pred'] == 1)
    dn_count = sum(1 for r in model_results.values() if r['pred'] == 0)
    total_models = len(model_results)
    # 一致性：方向一致的模型占比
    agreement_rate = up_count / total_models if pred == 1 else dn_count / total_models if total_models > 0 else 0.5

    # 按模型分组汇总（用于前端5个仪表）
    group_map = {
        'ml': ['rf1','rf2','rf3','et','gb','hgb'],
        'boost': ['lgb1','lgb2','xgb1','xgb2','ada'],
        'deep': ['mlp','cat1','cat2'],
        'stat': ['svm1','svm2','lda','qda','nb','lr'],
        'knn':  ['knn','ensemble'],
    }
    group_confs = {}
    for g, names in group_map.items():
        preds = [model_results[n]['pred'] for n in names if n in model_results]
        confs = [model_results[n]['conf'] for n in names if n in model_results]
        if preds:
            up = sum(p == 1 for p in preds)
            group_confs[g] = round(up / len(preds) * 100, 1)
        else:
            group_confs[g] = 50.0

    result = {
        'price': price,
        'prediction': int(pred),      # 1=涨 0=跌
        'confidence': round(conf_adjusted, 4),
        'confidence_raw': round(conf, 4),  # 未经趋势调整的原始置信度
        'agreement_rate': round(agreement_rate, 3),  # 模型方向一致性（越高越可靠）
        'up_votes': up_count,
        'down_votes': dn_count,
        'total_models': total_models,
        'near_integer': near,
        'key_level': key,
        'group_confs': group_confs,   # 前端5仪表用
        'direction': 'UP' if pred == 1 else 'DOWN',
        'trend_dir': trend_dir,           # -1/0/1
        'trend_strength': round(trend_strength, 3),
        'against_trend': against_trend,   # True=逆趋势预测，警告
        'trend_details': trend_details,
    }
    with state_lock:
        state['last_prediction'] = result
        state['last_price'] = price
        state['last_signal'] = 'up' if pred == 1 else 'down'  # 供复利模块使用
        state['pending_model_preds'] = model_results  # 供在线学习权重更新使用
    return jsonify(result)

@app.route('/api/train', methods=['POST'])
def api_train():
    with state_lock:
        if state['training']:
            return jsonify({'status': 'already_training', 'message': '训练进行中'}), 409
        state['training'] = True
        state['train_progress'] = 0
        state['train_status'] = '准备开始训练...'

    try:
        data = request.get_json(silent=True) or {}
        period = int(data.get('period', 10))
        if period not in (5, 10, 30):
            period = 10
    except Exception:
        period = 10

    t = threading.Thread(target=train_models_thread, args=(period,), daemon=True)
    t.start()
    return jsonify({'status': 'started', 'message': f'训练已开始 (period={period}min)'})

@app.route('/api/train/progress', methods=['GET'])
def api_train_progress():
    with state_lock:
        return jsonify({
            'training': state['training'],
            'progress': state['train_progress'],
            'status': state['train_status'],
        })

@app.route('/api/klines', methods=['GET'])
def api_klines():
    limit = int(request.args.get('limit', 200))
    interval = request.args.get('interval', INTERVAL)
    # 只允许合法的interval，避免乱传
    if interval not in ('1m', '3m', '5m', '15m', '30m', '1h', '4h'):
        interval = '1m'
    df = get_klines(min(limit, 500), interval)
    if df is None:
        return jsonify({'error': '获取K线失败'}), 503
    rows = df.tail(limit).copy()
    # open_time 统一转为毫秒时间戳（前端 lightweight-charts 需要秒级）
    rows['t'] = (rows['open_time'].astype('int64') // 10**9).astype(int)  # ns→秒（lightweight-charts用秒）
    result = []
    for _, row in rows.iterrows():
        result.append({
            't': int(row['t']),          # ms timestamp
            'o': float(row['open']),
            'h': float(row['high']),
            'l': float(row['low']),
            'c': float(row['close']),
            'v': float(row['volume']),
        })
    return jsonify(result)

# ========== 启动 ==========
def main():
    print("=" * 50)
    print("复利引擎后端服务 v2.0")
    print(f"模型目录: {MODEL_DIR}")
    print("=" * 50)

    # 后台加载模型
    t = threading.Thread(target=load_all_models, daemon=True)
    t.start()

    port = int(os.environ.get('PORT', 7788))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

# ========== 自动下单 API（仅软件版使用，云端无效） ==========
# 动态导入auto_trade，云端没有pyautogui不影响正常运行
try:
    from auto_trade import (
        trader, load_coords, save_coords, calibrate_all,
        calculate_compound_plan, format_amount, verify_license,
        PLATFORM_RULES
    )
    AUTO_TRADE_AVAILABLE = True
except ImportError:
    AUTO_TRADE_AVAILABLE = False

@app.route('/api/auto_trade/status', methods=['GET'])
def api_auto_trade_status():
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"available": False, "message": "自动下单不可用（云端模式）"})
    return jsonify({"available": True, **trader.get_status()})

@app.route('/api/auto_trade/coords', methods=['GET'])
def api_auto_trade_get_coords():
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "自动下单不可用"}), 400
    return jsonify(load_coords())

@app.route('/api/auto_trade/coords', methods=['POST'])
def api_auto_trade_save_coords():
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "自动下单不可用"}), 400
    data = request.json or {}
    coords = data.get("coords", {})
    amount = float(data.get("trade_amount", 10))
    save_coords(coords, amount)
    return jsonify({"ok": True})

@app.route('/api/auto_trade/calibrate', methods=['POST'])
def api_auto_trade_calibrate():
    """启动坐标标定流程（需要在本地运行，云端无效）"""
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "自动下单不可用"}), 400
    import threading as _threading
    def _do():
        coords = calibrate_all()
        save_coords(coords, 10)
    _threading.Thread(target=_do, daemon=True).start()
    return jsonify({"ok": True, "message": "标定已启动，请按终端提示操作"})

@app.route('/api/auto_trade/mouse_pos', methods=['GET'])
def api_auto_trade_mouse_pos():
    """获取当前鼠标坐标（用于网页版坐标标定）"""
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "自动下单不可用"}), 400
    try:
        import pyautogui
        x, y = pyautogui.position()
        return jsonify({"ok": True, "x": x, "y": y})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/auto_trade/plan', methods=['POST'])
def api_auto_trade_plan():
    """预览复利计划表"""
    data = request.json or {}
    initial = float(data.get("initial", 10))
    payout = float(data.get("payout", 0.8))
    rounds = int(data.get("rounds", 5))
    platform = data.get("platform", "binance")
    plan = calculate_compound_plan(initial, payout, rounds, platform)
    return jsonify({"ok": True, "plan": plan})

@app.route('/api/auto_trade/execute', methods=['POST'])
def api_auto_trade_execute():
    """执行单次普通下单"""
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "自动下单不可用（云端模式）"}), 400
    data = request.json or {}
    coords_data = load_coords()
    coords = coords_data.get("coords", {})
    amount = float(data.get("amount", coords_data.get("trade_amount", 10)))
    direction = data.get("direction", "up")
    result = trader.start_normal(coords, amount, direction)
    return jsonify(result)

@app.route('/api/auto_trade/compound/start', methods=['POST'])
def api_auto_trade_compound_start():
    """启动复利下单"""
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "自动下单不可用（云端模式）"}), 400
    data = request.json or {}
    coords_data = load_coords()
    coords = coords_data.get("coords", {})
    initial = float(data.get("initial", 10))
    payout = float(data.get("payout", 0.8))
    rounds = int(data.get("rounds", 5))
    platform = data.get("platform", "binance")

    # 方向函数：每次从预测API获取最新信号
    def get_direction():
        try:
            with state_lock:
                sig = state.get('last_signal', None)
            if sig in ('up', 'down'):
                return sig
        except Exception:
            pass
        return None

    result = trader.start_compound(coords, initial, payout, rounds, platform, get_direction)
    return jsonify(result)

@app.route('/api/auto_trade/compound/result', methods=['POST'])
def api_auto_trade_compound_result():
    """上报本轮结果（win/loss）"""
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "不可用"}), 400
    data = request.json or {}
    result = data.get("result", "")
    if result not in ("win", "loss"):
        return jsonify({"error": "result 必须是 win 或 loss"}), 400
    trader.report_result(result)
    return jsonify({"ok": True})

@app.route('/api/auto_trade/stop', methods=['POST'])
def api_auto_trade_stop():
    """停止自动交易"""
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"error": "不可用"}), 400
    trader.stop()
    return jsonify({"ok": True})


@app.route('/api/adaptive/update', methods=['POST'])
def api_adaptive_update():
    """
    上报本轮结果触发在线学习权重更新
    body: {"actual": 1(涨赢)/0(跌赢)}
    """
    data = request.json or {}
    actual = data.get("actual")
    if actual not in (0, 1):
        return jsonify({"error": "actual 必须是 0 或 1"}), 400
    with state_lock:
        model_preds = state.get('pending_model_preds')
    if not model_preds:
        return jsonify({"ok": False, "message": "无待更新预测记录"})
    update_adaptive_weights(model_preds, actual)
    with state_lock:
        weights = dict(state['adaptive_weights'])
    return jsonify({"ok": True, "weights": weights})


@app.route('/api/adaptive/weights', methods=['GET'])
def api_adaptive_weights():
    """查询当前各模型自适应权重"""
    with state_lock:
        weights = dict(state['adaptive_weights'])
        perf = {k: {"count": len(v), "winrate": round(sum(v)/len(v)*100, 1) if v else 0}
                for k, v in state['model_perf_window'].items()}
    return jsonify({"ok": True, "weights": weights, "performance": perf})


@app.route('/api/license/verify', methods=['POST'])
def api_license_verify():
    """验证注册码"""
    if not AUTO_TRADE_AVAILABLE:
        return jsonify({"ok": False, "message": "注册码验证仅软件版可用"})
    data = request.json or {}
    code = data.get("code", "").strip()
    result = verify_license(code)
    return jsonify(result)


if __name__ == '__main__':
    main()
