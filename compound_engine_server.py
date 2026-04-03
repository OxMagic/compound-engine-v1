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
import json
import sqlite3
import warnings
warnings.filterwarnings('ignore')

import requests
import pandas as pd
import numpy as np
from flask import Flask, jsonify, request, send_file
from flask_cors import CORS

# ========== 路径配置 ==========
# PyInstaller 打包后，数据文件在 sys._MEIPASS 下
if getattr(sys, 'frozen', False):
    _BASE_DIR = sys._MEIPASS
else:
    _BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(_BASE_DIR, 'models')
# 日志写在项目目录下，兼容云端（不用Windows绝对路径）
if getattr(sys, 'frozen', False):
    LOG_CSV = os.path.join(os.path.dirname(sys.executable), 'compound_engine_log.csv')
else:
    LOG_CSV = os.path.join(_BASE_DIR, 'compound_engine_log.csv')

# 预测日志数据库路径（SQLite，持久化存储，关机不丢）
if getattr(sys, 'frozen', False):
    DB_PATH = os.path.join(os.path.dirname(sys.executable), 'prediction_log.db')
else:
    DB_PATH = os.path.join(_BASE_DIR, 'prediction_log.db')

SYMBOL    = "BTCUSDT"
SYMBOL_FUTURES = "BTCUSDT"  # 合约API symbol（大部分交易所相同）
INTERVAL  = "1m"
DATA_LIMIT = 1000  # 使用1000根K线训练，样本更充足（Binance API上限）

# ========== K线缓存（60秒内复用，避免重复请求Binance） ==========
_klines_cache = {'df': None, 'ts': 0, 'limit': 0}
_klines_cache_lock = threading.Lock()

# ========== 外部市场数据缓存（避免频繁请求，独立于K线） ==========
_market_data_cache = {
    'orderbook': {'bids': [], 'asks': [], 'ts': 0},
    'futures':  {'oi': float('nan'), 'funding_rate': float('nan'), 'ts': 0},
    'fear_greed': {'value': 50, 'ts': 0},
}
_market_data_lock = threading.Lock()
# 各数据源缓存有效期（秒）
_OB_CACHE_TTL = 10       # 订单簿 10秒（变化快）
_FUTURES_CACHE_TTL = 60  # OI/资金费率 60秒（8小时更新一次）
_FNG_CACHE_TTL = 300     # 贪婪恐惧指数 5分钟

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

# ========== 多时间框架配置 ==========
TF_PERIODS = [5, 10, 30]  # 支持的预测周期（分钟）
ACTIVE_PERIOD = 10         # 默认活跃周期

# ========== Flask App ==========
app = Flask(__name__)
CORS(app)

# ========== 全局状态 ==========
state = {
    'models': {},               # 默认模型（根目录），向后兼容
    'scalers': {},
    'features': [],
    'per_model_features': {},   # 每个模型自己保存的特征列
    'tf_models': {},            # 多时间框架模型 {period: {name: model_obj}}
    'tf_scalers': {},           # 多时间框架scalers {period: {name: scaler}}
    'tf_features': {},          # 多时间框架特征 {period: [feature_cols]}
    'tf_per_model_features': {},# 多时间框架每模型特征 {period: {name: [cols]}}
    'loaded': False,
    'training': False,
    'train_progress': 0,
    'train_status': '',
    'last_prediction': None,
    'last_price': None,
    'last_signal': None,   # 'up' / 'down'，供自动下单复利模块使用
    # ===== 动态置信度阈值 =====
    'dynamic_conf': {
        'base_threshold': 0.65,
        'current_threshold': 0.65,
        'win_streak': 0,
        'loss_streak': 0,
    },
    # ===== 自适应权重（在线学习）=====
    'adaptive_weights': {},     # {model_name: float}，动态权重，初始1.0
    'model_perf_window': {},    # {model_name: deque(maxlen=50)}，近50次正确/错误
    'pending_model_preds': None,# 最近一次预测的各模型结果，等待结果回调更新权重
    # ===== SHAP 特征重要性（训练后更新）=====
    'shap_importance': {},      # {feature_name: mean_abs_shap_value}
    # ===== 智能重训滑动统计 =====
    'retrain_win_window': [],   # 近期结算胜负序列（1/0），用于滑坡检测
}
state_lock = threading.Lock()

# ========== 自适应权重工具函数 ==========
from collections import deque

ADAPTIVE_WINDOW = 50      # 统计近50次
ADAPTIVE_MIN_W  = 0.1     # 最低权重
ADAPTIVE_MAX_W  = 3.0     # 最高权重
ADAPTIVE_LR     = 0.15    # 学习率（每次更新的步长）

# ========== 预测日志数据库（SQLite 持久化） ==========
class PredictionDB:
    """
    预测日志持久化 + 自动补全实际结果
    - 每次预测自动记录到 SQLite
    - 后台线程定时检查到期的预测，拉价格补上实际涨跌
    - 所有数据关机不丢，重启后自动恢复
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _get_conn(self):
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        conn = self._get_conn()
        try:
            conn.executescript('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    price REAL NOT NULL,
                    direction TEXT NOT NULL,       -- 'UP' / 'DOWN'
                    confidence REAL NOT NULL,      -- 趋势调整后置信度
                    confidence_raw REAL,           -- 原始置信度
                    agreement_rate REAL,           -- 模型一致性
                    trend_dir INTEGER,             -- -1/0/1
                    trend_strength REAL,           -- 0~1
                    against_trend INTEGER,         -- 0/1
                    entry_score REAL,              -- 进场评分 0~100
                    period INTEGER DEFAULT 10,     -- 预测周期(分钟)
                    model_detail TEXT,             -- JSON: 各模型预测详情
                    -- 实际结果（预测到期后补填）
                    actual_price REAL,             -- 到期时的实际价格
                    actual_change REAL,            -- 实际涨跌幅(%)
                    actual_direction TEXT,         -- 实际方向 'UP'/'DOWN'
                    correct INTEGER,               -- 1=预测正确 0=错误 NULL=未到期
                    settled INTEGER DEFAULT 0,     -- 0=未结算 1=已结算
                    settle_source TEXT,            -- 'auto' / 'manual'
                    settle_time TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_pred_timestamp ON predictions(timestamp);
                CREATE INDEX IF NOT EXISTS idx_pred_settled ON predictions(settled);
                CREATE INDEX IF NOT EXISTS idx_pred_correct ON predictions(correct);

                -- 自动重训记录
                CREATE TABLE IF NOT EXISTS auto_retrain_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    retrain_time TEXT NOT NULL,
                    period INTEGER DEFAULT 10,
                    old_acc REAL,
                    new_acc REAL,
                    models_trained INTEGER,
                    models_kept INTEGER,
                    duration_sec REAL,
                    reason TEXT
                );

                -- 自适应权重持久化（关机保存，重启恢复）
                CREATE TABLE IF NOT EXISTS adaptive_weights (
                    model_name TEXT PRIMARY KEY,
                    weight REAL DEFAULT 1.0,
                    correct_count INTEGER DEFAULT 0,
                    total_count INTEGER DEFAULT 0,
                    updated_at TEXT
                );
            ''')
            conn.commit()
        finally:
            conn.close()

    def add_prediction(self, price, direction, confidence, confidence_raw=None,
                       agreement_rate=None, trend_dir=None, trend_strength=None,
                       against_trend=False, entry_score=None, period=10,
                       model_detail=None):
        """记录一次预测"""
        import datetime
        conn = self._get_conn()
        try:
            conn.execute('''
                INSERT INTO predictions
                    (timestamp, price, direction, confidence, confidence_raw,
                     agreement_rate, trend_dir, trend_strength, against_trend,
                     entry_score, period, model_detail)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                datetime.datetime.utcnow().isoformat(),
                price, direction, confidence, confidence_raw,
                agreement_rate, trend_dir, trend_strength,
                1 if against_trend else 0,
                entry_score, period,
                json.dumps(model_detail, default=str) if model_detail else None
            ))
            conn.commit()
            # 返回刚插入的 id
            row = conn.execute('SELECT last_insert_rowid() as id').fetchone()
            return row['id']
        finally:
            conn.close()

    def settle_prediction(self, pred_id, actual_price, settle_source='manual'):
        """结算一次预测：填入实际价格并判断对错"""
        import datetime
        conn = self._get_conn()
        try:
            row = conn.execute(
                'SELECT price, direction, period FROM predictions WHERE id = ?',
                (pred_id,)
            ).fetchone()
            if not row:
                return None
            pred_price = row['price']
            pred_dir = row['direction']
            period = row['period']
            change_pct = ((actual_price - pred_price) / pred_price) * 100
            actual_dir = 'UP' if change_pct > 0 else 'DOWN'
            correct = 1 if actual_dir == pred_dir else 0
            now = datetime.datetime.utcnow().isoformat()
            conn.execute('''
                UPDATE predictions SET
                    actual_price = ?,
                    actual_change = ?,
                    actual_direction = ?,
                    correct = ?,
                    settled = 1,
                    settle_source = ?,
                    settle_time = ?
                WHERE id = ?
            ''', (actual_price, round(change_pct, 6), actual_dir,
                  correct, settle_source, now, pred_id))
            conn.commit()
            return {'correct': correct, 'change_pct': round(change_pct, 4),
                    'actual_dir': actual_dir, 'pred_dir': pred_dir}
        finally:
            conn.close()

    def get_pending_predictions(self, current_price=None):
        """获取所有未结算且已到期的预测（当前时间 > 预测时间 + period分钟）"""
        import datetime
        conn = self._get_conn()
        try:
            now = datetime.datetime.utcnow()
            rows = conn.execute('''
                SELECT id, timestamp, price, direction, period
                FROM predictions WHERE settled = 0
            ''').fetchall()
            pending = []
            for row in rows:
                pred_time = datetime.datetime.fromisoformat(row['timestamp'])
                expiry = pred_time + datetime.timedelta(minutes=row['period'])
                if now >= expiry:
                    pending.append(dict(row))
            return pending
        finally:
            conn.close()

    def get_statistics(self, hours=24):
        """获取统计信息"""
        import datetime
        conn = self._get_conn()
        try:
            cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).isoformat()

            # 总体统计
            row = conn.execute('''
                SELECT
                    COUNT(*) as total,
                    SUM(CASE WHEN settled = 1 THEN 1 ELSE 0 END) as settled_count,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as correct_count,
                    SUM(CASE WHEN correct = 0 THEN 1 ELSE 0 END) as wrong_count,
                    AVG(CASE WHEN settled = 1 THEN confidence END) as avg_conf,
                    AVG(CASE WHEN correct = 1 THEN confidence END) as avg_conf_correct,
                    AVG(CASE WHEN correct = 0 THEN confidence END) as avg_conf_wrong,
                    AVG(CASE WHEN settled = 1 THEN agreement_rate END) as avg_agreement,
                    AVG(CASE WHEN correct = 1 THEN agreement_rate END) as avg_agreement_correct
                FROM predictions WHERE timestamp >= ?
            ''', (cutoff,)).fetchone()

            # 按方向统计
            dir_stats = conn.execute('''
                SELECT direction,
                       COUNT(*) as total,
                       SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
                FROM predictions WHERE timestamp >= ? AND settled = 1
                GROUP BY direction
            ''', (cutoff,)).fetchall()

            # 连胜连败（简化版：CTE 分组计算）
            streak = conn.execute('''
                WITH ordered AS (
                    SELECT correct, ROW_NUMBER() OVER (ORDER BY timestamp) as rn
                    FROM predictions WHERE timestamp >= ? AND settled = 1 AND correct IS NOT NULL
                ),
                groups AS (
                    SELECT correct, rn,
                           rn - ROW_NUMBER() OVER (PARTITION BY correct ORDER BY rn) as grp
                    FROM ordered
                )
                SELECT correct, COUNT(*) as length, MAX(rn) as end_idx
                FROM groups
                GROUP BY correct, grp
                ORDER BY end_idx DESC LIMIT 10
            ''', (cutoff,)).fetchall()

            # 按置信度分桶统计胜率
            conf_buckets = conn.execute('''
                SELECT
                    CASE
                        WHEN confidence >= 0.8 THEN '80%+'
                        WHEN confidence >= 0.7 THEN '70-80%'
                        WHEN confidence >= 0.6 THEN '60-70%'
                        ELSE '<60%'
                    END as conf_range,
                    COUNT(*) as total,
                    SUM(CASE WHEN correct = 1 THEN 1 ELSE 0 END) as wins
                FROM predictions WHERE timestamp >= ? AND settled = 1
                GROUP BY conf_range
            ''', (cutoff,)).fetchall()

            result = {
                'period_hours': hours,
                'total_predictions': row['total'] or 0,
                'settled': row['settled_count'] or 0,
                'correct': row['correct_count'] or 0,
                'wrong': row['wrong_count'] or 0,
                'win_rate': round((row['correct_count'] or 0) / max(1, row['settled_count'] or 0) * 100, 1),
                'avg_confidence': round(row['avg_conf'] or 0, 3),
                'avg_conf_correct': round(row['avg_conf_correct'] or 0, 3),
                'avg_conf_wrong': round(row['avg_conf_wrong'] or 0, 3),
                'avg_agreement': round(row['avg_agreement'] or 0, 3),
                'avg_agreement_correct': round(row['avg_agreement_correct'] or 0, 3),
                'direction_stats': {d['direction']: {'total': d['total'], 'wins': d['wins']}
                                    for d in dir_stats},
                'recent_streaks': [{'correct': s['correct'], 'length': s['length']}
                                   for s in streak],
                'confidence_buckets': {b['conf_range']: {'total': b['total'], 'wins': b['wins'],
                                        'win_rate': round(b['wins']/max(1,b['total'])*100, 1)}
                                       for b in conf_buckets},
            }
            return result
        finally:
            conn.close()

    def get_recent_predictions(self, limit=50, settled_only=False):
        """获取最近的预测记录"""
        conn = self._get_conn()
        try:
            extra = ' AND settled = 1' if settled_only else ''
            rows = conn.execute(f'''
                SELECT id, timestamp, price, direction, confidence, confidence_raw,
                       agreement_rate, trend_dir, entry_score, period,
                       actual_price, actual_change, actual_direction, correct, settled
                FROM predictions {extra if settled_only else ''}
                ORDER BY id DESC LIMIT ?
            ''', (limit,)).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def save_adaptive_weights(self, weights, perf_windows):
        """持久化自适应权重（关机前保存）"""
        import datetime
        conn = self._get_conn()
        try:
            now = datetime.datetime.utcnow().isoformat()
            for name, weight in weights.items():
                perf = perf_windows.get(name)
                correct_count = sum(perf) if perf else 0
                total_count = len(perf) if perf else 0
                conn.execute('''
                    INSERT INTO adaptive_weights (model_name, weight, correct_count, total_count, updated_at)
                    VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(model_name) DO UPDATE SET
                        weight = excluded.weight,
                        correct_count = excluded.correct_count,
                        total_count = excluded.total_count,
                        updated_at = excluded.updated_at
                ''', (name, weight, correct_count, total_count, now))
            conn.commit()
        finally:
            conn.close()

    def load_adaptive_weights(self):
        """加载持久化的自适应权重（启动时恢复）"""
        conn = self._get_conn()
        try:
            rows = conn.execute('SELECT model_name, weight FROM adaptive_weights').fetchall()
            return {r['model_name']: r['weight'] for r in rows}
        finally:
            conn.close()

    def get_retrain_history(self, limit=20):
        """获取自动重训历史"""
        conn = self._get_conn()
        try:
            rows = conn.execute(
                'SELECT * FROM auto_retrain_log ORDER BY id DESC LIMIT ?', (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
        finally:
            conn.close()

    def add_retrain_log(self, period, old_acc, new_acc, models_trained, models_kept,
                        duration_sec, reason='auto'):
        """记录一次自动重训"""
        import datetime
        conn = self._get_conn()
        try:
            conn.execute('''
                INSERT INTO auto_retrain_log
                    (retrain_time, period, old_acc, new_acc, models_trained, models_kept,
                     duration_sec, reason)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (datetime.datetime.utcnow().isoformat(), period, old_acc, new_acc,
                  models_trained, models_kept, duration_sec, reason))
            conn.commit()
        finally:
            conn.close()

    def get_unsettled_count(self):
        """获取未结算预测数量"""
        conn = self._get_conn()
        try:
            row = conn.execute('SELECT COUNT(*) as c FROM predictions WHERE settled = 0').fetchone()
            return row['c']
        finally:
            conn.close()


# 初始化全局数据库实例
pred_db = PredictionDB(DB_PATH)

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
def _load_models_from_dir(model_dir, name_prefix=''):
    """
    从指定目录加载所有 .pkl 模型文件
    name_prefix: 模型名称前缀，用于避免不同目录同名模型冲突
    返回: {name: model_obj}, {name: scaler}, feature_cols, {name: feature_cols}
    """
    loaded = {}
    scalers = {}
    features_found = []
    per_model_features = {}

    if not os.path.isdir(model_dir):
        return loaded, scalers, features_found, per_model_features

    for fname in sorted(os.listdir(model_dir)):
        if not fname.endswith('.pkl'):
            continue
        path = os.path.join(model_dir, fname)
        # 从文件名提取模型名: btc_model_rf1.pkl → rf1
        model_key = fname.replace('btc_model_', '').replace('.pkl', '')
        # 如果有前缀则加上: 5min_rf1
        full_name = f'{name_prefix}{model_key}' if name_prefix else model_key

        try:
            with open(path, 'rb') as f:
                data = pickle.load(f)
            if isinstance(data, dict):
                loaded[full_name] = data.get('model')
                scalers[full_name] = data.get('scaler')
                m_features = data.get('features', [])
                if m_features:
                    per_model_features[full_name] = m_features
                if not features_found and m_features:
                    features_found = m_features
            else:
                loaded[full_name] = data
        except Exception as e:
            print(f"[WARN] 加载模型 {full_name} ({path}) 失败: {e}")
    
    return loaded, scalers, features_found, per_model_features


def load_all_models():
    # 1) 加载根目录模型（原有逻辑，向后兼容）
    loaded, scalers, features_found, per_model_features = _load_models_from_dir(MODEL_DIR)
    
    # 2) 加载多时间框架模型（models/5min/, models/10min/, models/30min/）
    tf_models = {}
    tf_scalers = {}
    tf_features = {}
    tf_per_model_features = {}

    for period in TF_PERIODS:
        tf_dir = os.path.join(MODEL_DIR, f'{period}min')
        prefix = f'{period}min_'
        m, s, f, pmf = _load_models_from_dir(tf_dir, name_prefix=prefix)
        if m:
            tf_models[period] = m
            tf_scalers[period] = s
            tf_features[period] = f
            tf_per_model_features[period] = pmf
            print(f"[INFO] 已加载 {period}min 时间框架: {len(m)} 个模型")
        else:
            print(f"[INFO] {period}min 时间框架: 无可用模型（目录 {tf_dir} 不存在或为空）")

    with state_lock:
        state['models'] = loaded
        state['scalers'] = scalers
        state['features'] = features_found
        state['per_model_features'] = per_model_features
        state['tf_models'] = tf_models
        state['tf_scalers'] = tf_scalers
        state['tf_features'] = tf_features
        state['tf_per_model_features'] = tf_per_model_features
        state['loaded'] = True
    
    total_tf = sum(len(v) for v in tf_models.values())
    print(f"[INFO] 根目录模型: {len(loaded)} 个: {list(loaded.keys())}")
    print(f"[INFO] 多时间框架模型总计: {total_tf} 个")
    print(f"[INFO] 总计: {len(loaded) + total_tf} 个模型")
    if len(loaded) == 0 and total_tf == 0:
        print("[WARN] models/ 目录为空或无可用模型，请先点击「训练模型」生成模型（需VPN）")

    # ── 恢复持久化的自适应权重 ──
    try:
        saved_weights = pred_db.load_adaptive_weights()
        if saved_weights:
            with state_lock:
                state['adaptive_weights'] = saved_weights
            print(f"[INFO] 已恢复 {len(saved_weights)} 个模型的自适应权重")
    except Exception as e:
        print(f"[WARN] 恢复自适应权重失败: {e}")

    return loaded

# ========== 数据获取 ==========
def safe_get(url, retries=2, timeout=20, **kwargs):
    """带重试的HTTP请求"""
    for attempt in range(retries + 1):
        try:
            r = requests.get(url, timeout=timeout, **kwargs)
            r.raise_for_status()
            return r
        except Exception as e:
            if attempt < retries:
                time.sleep(1)
            else:
                print(f"[WARN] safe_get failed after {retries+1} attempts: {e}")
    return None

# ========== 外部市场数据获取 ==========

def fetch_orderbook():
    """
    获取 BTCUSDT 实时订单簿（Binance现货，limit=20 足够计算特征）。
    带 10 秒缓存，避免频繁请求。
    返回 (bids, asks)，每个是 [(price, qty), ...] 列表。
    """
    with _market_data_lock:
        cached = _market_data_cache['orderbook']
        if time.time() - cached['ts'] < _OB_CACHE_TTL and cached['bids']:
            return cached['bids'], cached['asks']

    sources = [
        f"https://api.binance.com/api/v3/depth?symbol={SYMBOL}&limit=20",
    ]
    for url in sources:
        try:
            r = safe_get(url, retries=1, timeout=8)
            if r is None:
                continue
            j = r.json()
            bids = [(float(p), float(q)) for p, q in j.get('bids', [])]
            asks = [(float(p), float(q)) for p, q in j.get('asks', [])]
            if bids and asks:
                with _market_data_lock:
                    _market_data_cache['orderbook'] = {'bids': bids, 'asks': asks, 'ts': time.time()}
                return bids, asks
        except Exception:
            continue
    return [], []


def fetch_futures_data():
    """
    获取 BTCUSDT 合约 OI 和资金费率（Binance Futures API）。
    带 60 秒缓存。
    返回 {'oi': float, 'funding_rate': float}，获取失败时值为 nan。
    """
    with _market_data_lock:
        cached = _market_data_cache['futures']
        if time.time() - cached['ts'] < _FUTURES_CACHE_TTL:
            return {'oi': cached['oi'], 'funding_rate': cached['funding_rate']}

    oi = float('nan')
    funding_rate = float('nan')

    # OI
    try:
        r = safe_get(
            f"https://fapi.binance.com/fapi/v1/openInterest?symbol={SYMBOL_FUTURES}",
            retries=1, timeout=8
        )
        if r is not None:
            oi = float(r.json().get("openInterest", 0.0))
    except Exception:
        pass

    # 资金费率（尝试两个端点）
    try:
        r = safe_get(
            f"https://fapi.binance.com/fapi/v1/premiumIndex?symbol={SYMBOL_FUTURES}",
            retries=1, timeout=8
        )
        if r is not None:
            val = r.json().get('lastFundingRate')
            if val is not None:
                funding_rate = float(val)
    except Exception:
        pass
    if math.isnan(funding_rate):
        try:
            r = safe_get(
                f"https://fapi.binance.com/fapi/v1/fundingRate?symbol={SYMBOL_FUTURES}&limit=1",
                retries=1, timeout=8
            )
            if r is not None:
                arr = r.json()
                if isinstance(arr, list) and arr:
                    funding_rate = float(arr[-1].get('fundingRate', 0.0))
        except Exception:
            pass

    with _market_data_lock:
        _market_data_cache['futures'] = {'oi': oi, 'funding_rate': funding_rate, 'ts': time.time()}
    return {'oi': oi, 'funding_rate': funding_rate}


def fetch_fear_greed():
    """
    获取加密货币恐惧贪婪指数（alternative.me 免费 API）。
    带 5 分钟缓存。返回 0-100 整数。
    """
    with _market_data_lock:
        cached = _market_data_cache['fear_greed']
        if time.time() - cached['ts'] < _FNG_CACHE_TTL:
            return cached['value']

    try:
        r = safe_get("https://api.alternative.me/fng/", retries=1, timeout=5)
        if r is not None:
            data = r.json()
            if 'data' in data and len(data['data']) > 0:
                val = int(data['data'][0]['value'])
                if 0 <= val <= 100:
                    with _market_data_lock:
                        _market_data_cache['fear_greed'] = {'value': val, 'ts': time.time()}
                    return val
    except Exception:
        pass
    return 50  # 默认中性

# ========== K线获取 ==========
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
    for w in [5, 10, 20, 50]:
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
    # 下划线别名（兼容 train_multi_timeframe.py 的特征命名）
    df['bbw_20'] = df['bb_width20']
    df['bb_position_20'] = df['bb_pos20']
    df['bb_squeeze_20'] = df['bb_squeeze20']
    df['bbw_50'] = df['bb_width50']
    df['bb_position_50'] = df['bb_pos50']
    df['bb_squeeze_50'] = df['bb_squeeze50']

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
    # 修复：cumsum绝对值在训练(1000根)和推理(300根)时起点不同，数值不可比
    # 改为：只保留相对化的变化量，去掉无意义的绝对值
    _obv_raw = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
    # OBV相对于20期均线的偏离（原有特征，保留）
    df['obv_ma_diff'] = _obv_raw - _obv_raw.rolling(20).mean()
    # OBV标准化变化量（替换无意义的绝对值obv）
    df['obv'] = _obv_raw.diff(5) / (_obv_raw.rolling(5).std() + 1e-9)

    # --- VWAP ---
    # 修复：cumsum从头累积，训练和推理窗口长度不同导致数值不可比（前视偏差）
    # 改为：滚动20期窗口VWAP，任何窗口长度下数值含义一致
    tp = (df['high'] + df['low'] + df['close']) / 3
    _vwap_window = 20
    df['vwap'] = (tp * df['volume']).rolling(_vwap_window).sum() / (df['volume'].rolling(_vwap_window).sum() + 1e-9)
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
    # 修复：close绝对价格含有伪相关（牛市高价≈上涨），改为相对收益率
    # volume绝对值同理，改为成交量变化率
    _ret1 = df['close'].pct_change(1)
    _vol_chg1 = df['volume'].pct_change(1)
    for lag in [1, 2, 3, 5]:
        df[f'close_lag{lag}']  = _ret1.shift(lag)    # 过去第lag根K线的涨跌幅（相对量）
        df[f'volume_lag{lag}'] = _vol_chg1.shift(lag) # 过去第lag根K线的成交量变化率

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

    # --- Hurst 指数 + 分形维度（R/S 方法） ---
    df['hurst'] = 0.5       # 默认随机游走值，最后一行会被实际计算覆盖
    df['fractal_dim'] = 2.0 # 默认值 D=2-Hurst

    def _hurst_exponent(ts, max_lag=20):
        """R/S 方法计算 Hurst 指数，ts 为 numpy array"""
        if len(ts) < max_lag * 2:
            return 0.5
        lags = list(range(2, min(max_lag, len(ts) // 2)))
        tau = []
        for lag in lags:
            chunks = [ts[i:i+lag] for i in range(0, len(ts)-lag, lag)]
            if len(chunks) < 2:
                continue
            rs_vals = []
            for chunk in chunks:
                if len(chunk) < 2:
                    continue
                mean_c = np.mean(chunk)
                deviation = np.cumsum(chunk - mean_c)
                R = np.max(deviation) - np.min(deviation)
                S = np.std(chunk, ddof=1)
                if S > 1e-12:
                    rs_vals.append(R / S)
            if rs_vals:
                tau.append(np.mean(rs_vals))
        if len(tau) < 2:
            return 0.5
        try:
            x = np.log(lags[:len(tau)])
            y = np.log(tau)
            slope, _ = np.polyfit(x, y, 1)
            return float(np.clip(slope, 0, 1))
        except Exception:
            return 0.5

    # 对最后一行做真实计算（训练和推理都只看最后一行有效）
    close_arr = df['close'].values
    if len(close_arr) >= 60:
        window = min(200, len(close_arr))
        h_val = _hurst_exponent(close_arr[-window:])
        df.at[df.index[-1], 'hurst'] = h_val
        df.at[df.index[-1], 'fractal_dim'] = round(2.0 - h_val, 4)

    # ================================================================
    # --- 时间周期特征 (P0 新增) ---
    # 使用 open_time 列（已是 pd.Timestamp / datetime64[ns]）
    # 所有特征全列赋值，训练和推理行为完全一致，无前视偏差
    # ================================================================
    try:
        _ts = pd.to_datetime(df['open_time'])            # 确保是 datetime 类型
        _utc8 = _ts + pd.Timedelta(hours=8)             # UTC → UTC+8（北京时间）

        # 1. 小时（0-23），对应一天内的交易时段
        df['hour_of_day'] = _utc8.dt.hour.astype(float)

        # 2. 星期几（0=周一 … 6=周日），捕捉周末/工作日节奏
        df['day_of_week'] = _utc8.dt.dayofweek.astype(float)

        # 3. sin/cos 编码——让模型理解"23点和0点很近"
        df['hour_sin']  = np.sin(2 * np.pi * df['hour_of_day'] / 24)
        df['hour_cos']  = np.cos(2 * np.pi * df['hour_of_day'] / 24)
        df['dow_sin']   = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos']   = np.cos(2 * np.pi * df['day_of_week'] / 7)

        # 4. 交易时段分类（BTC 24h 不停盘，但流动性有明显差异）
        #    亚洲盘 08-16 → 0；欧洲盘 16-22 → 1；美洲盘 22-08 → 2
        def _session(h):
            if 8 <= h < 16:
                return 0   # 亚洲盘
            elif 16 <= h < 22:
                return 1   # 欧洲盘
            else:
                return 2   # 美洲盘（含深夜）
        df['trading_session'] = df['hour_of_day'].apply(_session).astype(float)

        # 5. 是否周末（周六=5 / 周日=6），周末流动性通常较低
        df['is_weekend'] = (_utc8.dt.dayofweek >= 5).astype(float)

        # 6. 分钟数（0-59）——对 5/10/30 分钟 K 线尤其有用，
        #    捕捉整点/半点附近的行为差异
        df['minute_of_hour'] = _utc8.dt.minute.astype(float)
        df['min_sin']  = np.sin(2 * np.pi * df['minute_of_hour'] / 60)
        df['min_cos']  = np.cos(2 * np.pi * df['minute_of_hour'] / 60)

    except Exception as _te:
        # open_time 不可用时（如测试数据）降级为全零，不影响运行
        print(f"[TIME FEAT] open_time 解析失败，跳过时间特征: {_te}")
        for _col in ['hour_of_day','day_of_week','hour_sin','hour_cos',
                     'dow_sin','dow_cos','trading_session','is_weekend',
                     'minute_of_hour','min_sin','min_cos']:
            df[_col] = 0.0

    # --- 外部市场数据特征（订单簿/OI/资金费率/贪婪恐惧指数） ---
    # 默认值 0.0 / nan，最后一行会被真实数据覆盖
    df['orderbook_imbalance'] = 0.0
    df['orderbook_pressure']  = 0.0
    df['open_interest']       = 0.0
    df['funding_rate']        = 0.0
    # 新增特征（旧模型 pkl 里没有这些列，推理时会自动忽略）
    df['depth_skew']          = 0.0
    df['spread_ratio']        = 0.0
    df['funding_trend']       = 0.0
    df['fear_greed']          = 0.5
    df['fear_greed_change']   = 0.0

    # 对最后一行填充真实外部数据（训练和推理都只看最后一行有效）
    last_idx = df.index[-1]
    try:
        # --- 订单簿数据 ---
        bids, asks = fetch_orderbook()
        if bids and asks:
            top_bid_p, top_bid_q = bids[0]
            top_ask_p, top_ask_q = asks[0]
            bid_vol = sum(q for _, q in bids[:10])
            ask_vol = sum(q for _, q in asks[:10])
            # 不平衡度：>0 买方强，<0 卖方强
            imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9)
            # 压力：加权（价格×数量）
            bid_pressure = sum(p * q for p, q in bids[:5])
            ask_pressure = sum(p * q for p, q in asks[:5])
            pressure = (bid_pressure - ask_pressure) / (bid_pressure + ask_pressure + 1e-9)
            # 深度偏斜（同 imbalance 但用全部20档）
            bid_vol20 = sum(q for _, q in bids)
            ask_vol20 = sum(q for _, q in asks)
            skew = (bid_vol20 - ask_vol20) / (bid_vol20 + ask_vol20 + 1e-9)
            # 价差比
            spread = top_ask_p - top_bid_p
            spread_r = spread / top_bid_p

            df.at[last_idx, 'orderbook_imbalance'] = round(imbalance, 6)
            df.at[last_idx, 'orderbook_pressure']  = round(pressure, 6)
            df.at[last_idx, 'depth_skew']          = round(skew, 6)
            df.at[last_idx, 'spread_ratio']        = round(spread_r, 8)

        # --- 合约数据（OI + 资金费率） ---
        futures = fetch_futures_data()
        if not math.isnan(futures['oi']):
            df.at[last_idx, 'open_interest'] = futures['oi']
        if not math.isnan(futures['funding_rate']):
            df.at[last_idx, 'funding_rate'] = futures['funding_rate']

        # --- 贪婪恐惧指数 ---
        fng = fetch_fear_greed()
        df.at[last_idx, 'fear_greed'] = fng / 100.0  # 归一化到 0-1
        # fear_greed_change 用上次缓存值计算（简化处理，0-1范围）
        with _market_data_lock:
            prev_ts = _market_data_cache['fear_greed'].get('prev_value')
        if prev_ts is not None:
            df.at[last_idx, 'fear_greed_change'] = round(fng / 100.0 - prev_ts, 4)
        with _market_data_lock:
            _market_data_cache['fear_greed']['prev_value'] = fng / 100.0

    except Exception as e:
        print(f"[MARKET DATA] 外部数据获取异常（不影响预测）: {e}")

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

# ========== 8个技术指标规则模型 ==========
def run_technical_models(df):
    """
    基于8个经典技术分析规则的投票系统。
    每个规则模型独立判断方向，返回 {model_name: {'pred': 1/0/None, 'conf': float}}
      pred=1  → 看涨
      pred=0  → 看跌
      pred=None → 无明确信号，不参与投票
    """
    if df is None or len(df) < 30:
        return {}

    results = {}
    try:
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) >= 2 else latest

        # ── 1. 趋势跟随 (trend_following) ──
        # EMA8/21 金叉 + MACD方向
        ema8 = latest.get('ema8', 0)
        ema21 = latest.get('ema21', 0)
        macd = latest.get('macd', 0)
        macd_sig = latest.get('macd_sig', 0)
        if ema8 > 0 and ema21 > 0:
            golden_cross = ema8 > ema21
            macd_bull = macd > macd_sig
            if golden_cross and macd_bull:
                results['trend'] = {'pred': 1, 'conf': 0.72}
            elif (not golden_cross) and (not macd_bull):
                results['trend'] = {'pred': 0, 'conf': 0.72}

        # ── 2. 均值回归 (mean_reversion) ──
        # RSI 超卖/超买 + 布林带位置
        rsi = latest.get('rsi', 50)
        bb_pos = latest.get('bb_position', 0.5)
        if not np.isnan(rsi) and not np.isnan(bb_pos):
            if rsi < 30 and bb_pos < 0.15:
                results['mean_rev'] = {'pred': 1, 'conf': 0.68}  # 超卖反弹
            elif rsi > 70 and bb_pos > 0.85:
                results['mean_rev'] = {'pred': 0, 'conf': 0.68}  # 超买回落
            elif rsi < 35 and bb_pos < 0.25:
                results['mean_rev'] = {'pred': 1, 'conf': 0.55}
            elif rsi > 65 and bb_pos > 0.75:
                results['mean_rev'] = {'pred': 0, 'conf': 0.55}

        # ── 3. 动量 (momentum) ──
        # 短期收益方向 + 价格加速度
        ret5 = latest.get('ret5', 0)
        accel = latest.get('price_accel', 0)
        if not np.isnan(ret5) and not np.isnan(accel):
            if ret5 > 0.001 and accel > 0:
                results['momentum'] = {'pred': 1, 'conf': 0.65}
            elif ret5 < -0.001 and accel < 0:
                results['momentum'] = {'pred': 0, 'conf': 0.65}

        # ── 4. 成交量 (volume) ──
        # 量比放大 + 方向确认
        vol_ratio = latest.get('vol_ratio10', 1.0)
        price_chg = latest.get('ret1', 0)
        close_now = latest.get('close', 0)
        close_prev = prev.get('close', 0)
        if not np.isnan(vol_ratio) and close_prev > 0:
            vol_surge = vol_ratio > 1.3
            price_up = close_now > close_prev
            price_dn = close_now < close_prev
            if vol_surge and price_up:
                results['volume'] = {'pred': 1, 'conf': 0.63}
            elif vol_surge and price_dn:
                results['volume'] = {'pred': 0, 'conf': 0.63}

        # ── 5. 波动率 (volatility) ──
        # ATR 在合理区间 + 布林带收窄预示突破
        atr_pct = latest.get('atr_percent', 0)
        bb_squeeze = latest.get('bb_squeeze', 0)
        bb_width = latest.get('bb_width20', 0)
        if not np.isnan(atr_pct) and not np.isnan(bb_width):
            # 布林带收窄 + ATR 相对低 → 突破在即
            if bb_width < bb_width * 0.8 + 0.0001 and atr_pct < 0.003:
                # 突破方向由短期动量决定
                ret3 = latest.get('ret3', 0)
                if not np.isnan(ret3):
                    if ret3 > 0:
                        results['volatility'] = {'pred': 1, 'conf': 0.58}
                    elif ret3 < 0:
                        results['volatility'] = {'pred': 0, 'conf': 0.58}

        # ── 6. 突破 (breakout) ──
        # 价格突破20期高低点
        if len(df) >= 22:
            high20 = df['high'].iloc[-22:-2].max()
            low20 = df['low'].iloc[-22:-2].min()
            price = latest.get('close', 0)
            if not np.isnan(high20) and not np.isnan(low20) and price > 0:
                if price > high20:
                    results['breakout'] = {'pred': 1, 'conf': 0.70}
                elif price < low20:
                    results['breakout'] = {'pred': 0, 'conf': 0.70}

        # ── 7. K线形态 (pattern) ──
        # 锤头线 / 流星线 / 十字星
        body_ratio = latest.get('oc_range', 0)
        lower_shadow = latest.get('lower_shadow', 0)
        upper_shadow = latest.get('upper_shadow', 0)
        hl_range = latest.get('hl_range', 0)
        if not np.isnan(body_ratio) and not np.isnan(lower_shadow) and not np.isnan(upper_shadow) and hl_range > 0:
            # 锤头线：下影线长，上影线短，实体小 → 看涨
            if lower_shadow > 2 * abs(body_ratio) and upper_shadow < abs(body_ratio) * 0.5:
                results['pattern'] = {'pred': 1, 'conf': 0.60}
            # 流星线：上影线长，下影线短，实体小 → 看跌
            elif upper_shadow > 2 * abs(body_ratio) and lower_shadow < abs(body_ratio) * 0.5:
                results['pattern'] = {'pred': 0, 'conf': 0.60}
            # 十字星：实体极小 → 不确定方向，不投票

        # ── 8. 市场情绪 (sentiment) ──
        # ADX 强度 + 成交量趋势 + 趋势方向
        adx = latest.get('adx', 0)
        vol_over_ma = latest.get('vol_over_ma', 1.0)
        market_regime = latest.get('market_regime', 0)
        if not np.isnan(adx) and not np.isnan(vol_over_ma):
            if adx > 25 and vol_over_ma > 1.2:
                # 强趋势 + 放量
                if market_regime > 0:
                    results['sentiment'] = {'pred': 1, 'conf': 0.67}
                elif market_regime < 0:
                    results['sentiment'] = {'pred': 0, 'conf': 0.67}

    except Exception as e:
        print(f"[RULES] 技术指标规则模型计算失败: {e}")

    return results


# ========== 综合进场评分（100分制） ==========
def compute_entry_score(pred, conf, agreement_rate, trend_dir, against_trend,
                        ob_imbalance=0, fear_greed=0.5, near_integer=False):
    """
    将多个维度综合成 0-100 的进场评分。
    pred: 1(涨) or 0(跌)
    conf: 0-1 置信度
    agreement_rate: 0-1 模型一致性
    trend_dir: 1(涨), -1(跌), 0(震荡)
    against_trend: bool 是否逆势
    ob_imbalance: float 订单簿不平衡度（>0买方强）
    fear_greed: float 0-1（0=极度恐惧, 1=极度贪婪）
    near_integer: bool 是否接近整数位
    """
    score = 0.0
    details = {}

    # 1. 模型置信度（0-40分）
    s_conf = min(conf, 1.0) * 40
    score += s_conf
    details['confidence'] = round(s_conf, 1)

    # 2. 模型一致性（0-20分）
    s_agree = min(agreement_rate, 1.0) * 20
    score += s_agree
    details['agreement'] = round(s_agree, 1)

    # 3. 趋势一致性（0-15分）
    if against_trend:
        s_trend = 3  # 逆势给最低分
    elif trend_dir == 0:
        s_trend = 8  # 震荡给中间分
    else:
        s_trend = 15  # 顺趋势满分
    score += s_trend
    details['trend'] = s_trend

    # 4. 订单簿支持（0-10分）
    # 方向与订单簿一致：买方强+看涨 或 卖方强+看跌 → 加分
    ob_signal = 1 if ob_imbalance > 0.1 else (-1 if ob_imbalance < -0.1 else 0)
    pred_dir = 1 if pred == 1 else -1
    if ob_signal == pred_dir:
        s_ob = min(abs(ob_imbalance) * 20, 10)  # imbalance越大分越高，上限10
    elif ob_signal == 0:
        s_ob = 5  # 中性给一半分
    else:
        s_ob = 1  # 方向相反给最低分
    score += s_ob
    details['orderbook'] = round(s_ob, 1)

    # 5. 贪婪恐惧指数（0-10分）
    # 极端恐惧(0-0.25)时看涨加分，极端贪婪(0.75-1)时看跌加分
    # （逆向思维：恐惧时买入，贪婪时卖出）
    if pred == 1:  # 看涨
        if fear_greed < 0.25:
            s_fng = 10  # 极度恐惧时看涨，完美逆向
        elif fear_greed < 0.45:
            s_fng = 7
        elif fear_greed < 0.55:
            s_fng = 5  # 中性
        else:
            s_fng = 2  # 贪婪时看涨，不理想
    else:  # 看跌
        if fear_greed > 0.75:
            s_fng = 10  # 极度贪婪时看跌，完美逆向
        elif fear_greed > 0.55:
            s_fng = 7
        elif fear_greed > 0.45:
            s_fng = 5  # 中性
        else:
            s_fng = 2  # 恐惧时看跌，不理想
    score += s_fng
    details['sentiment'] = s_fng

    # 6. 整数位参考（0-5分）
    s_int = 5 if near_integer else 0
    score += s_int
    details['integer'] = s_int

    return {
        'score': int(min(max(score, 0), 100)),
        'details': details,
        'grade': _score_grade(int(min(max(score, 0), 100)))
    }

def _score_grade(score):
    if score >= 85: return 'S'
    if score >= 70: return 'A'
    if score >= 55: return 'B'
    if score >= 40: return 'C'
    return 'D'

# ========== 核心预测 ==========
def run_prediction(df, price, precomputed=False, period=None):
    """
    df: K线 DataFrame
    precomputed: True 时表示 df 已经过 add_period_label 处理，
                 直接在其上补充技术指标，不覆盖 label
    period: 预测周期（分钟），5/10/30。传入时优先使用对应周期的模型
    """
    with state_lock:
        # 根据周期选择模型集合
        if period and period in state.get('tf_models', {}):
            models = state['tf_models'][period]
            scalers = state['tf_scalers'].get(period, {})
            saved_features = state['tf_features'].get(period, [])
            per_model_features_map = state.get('tf_per_model_features', {}).get(period, {})
            print(f"[PRED] 使用 {period}min 时间框架模型 ({len(models)} 个)")
        else:
            models = state['models']
            scalers = state['scalers']
            saved_features = state['features']
            per_model_features_map = state.get('per_model_features', {})
            if period:
                print(f"[PRED] {period}min 无可用模型，回退到根目录模型 ({len(models)} 个)")
    
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

    # ── 合并8个技术指标规则模型投票 ──
    technical_results = run_technical_models(df_feat)
    rule_votes_up = 0.0
    rule_votes_down = 0.0
    rule_up_raw = 0
    rule_dn_raw = 0
    for rname, rdata in technical_results.items():
        rp = rdata.get('pred')
        rc = rdata.get('conf', 0.6)
        if rp == 1:
            rule_votes_up += rc
            rule_up_raw += 1
        elif rp == 0:
            rule_votes_down += rc
            rule_dn_raw += 1

    # ML 权重 60% + 规则 权重 40%
    ml_weight = 0.6
    rule_weight = 0.4
    combined_up = votes_up * ml_weight + rule_votes_up * rule_weight
    combined_down = votes_down * ml_weight + rule_votes_down * rule_weight

    combined_total = combined_up + combined_down
    if combined_total > 0:
        final_pred = 1 if combined_up >= combined_down else 0
        final_conf = combined_up / combined_total if final_pred == 1 else combined_down / combined_total
    else:
        final_pred = 1 if votes_up >= votes_down else 0
        final_conf = votes_up / total_votes if final_pred == 1 else votes_down / total_votes
    
    # 一致性指数：方向一致的模型比例（0~1，越高越可靠）
    total_up_raw = up_count_raw + rule_up_raw
    total_dn_raw = dn_count_raw + rule_dn_raw
    total_all_raw = total_up_raw + total_dn_raw
    consensus = total_up_raw / total_all_raw if final_pred == 1 and total_all_raw > 0 else (total_dn_raw / total_all_raw if total_all_raw > 0 else 0.5)
    # 将一致性融入最终置信度（低一致性时拉低置信度）
    # 一致性<50%不可能（因为final_pred就是多数方向），最低约50%
    # 一致性>80%才算强信号
    final_conf = final_conf * (0.5 + 0.5 * consensus)

    # 把规则模型结果也放进 model_results（供前端展示）
    model_results['__rule__'] = technical_results
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
def _train_single_model(model, X_train_s, y_train, X_val_s, y_val, timeout=60):
    """在子线程里训练单个模型，超时则放弃"""
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
    result = {'model': None, 'acc': 0.0}
    
    def do_fit():
        model.fit(X_train_s, y_train)
        acc = float(np.mean(model.predict(X_val_s) == y_val))
        result['model'] = model
        result['acc'] = acc
    
    with ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(do_fit)
        try:
            fut.result(timeout=timeout)
        except FuturesTimeout:
            print(f"    ⚠ 训练超时（>{timeout}s），跳过")
        except Exception as e:
            print(f"    ⚠ 训练异常: {e}")
    
    return result


def _compute_shap_importance(trained_models, X_val_scaled, feature_cols, max_samples=200):
    """
    对已训练的模型集合计算 SHAP 特征重要性。
    优先用 TreeExplainer（RF/ET/GB/LGB），其次 LinearExplainer（LR），最后 KernelExplainer（兜底）。
    返回 {feature_name: mean_abs_shap} 字典，按重要性降序排列。
    """
    try:
        import shap
    except ImportError:
        # shap 未安装时优雅降级：用模型自带 feature_importances_
        aggregated = {}
        for name, model in trained_models.items():
            fi = getattr(model, 'feature_importances_', None)
            if fi is not None and len(fi) == len(feature_cols):
                for f, v in zip(feature_cols, fi):
                    aggregated[f] = aggregated.get(f, 0) + float(v)
        if not aggregated:
            return {}
        n = len([m for m in trained_models.values() if getattr(m, 'feature_importances_', None) is not None])
        result = {f: round(v / max(1, n), 6) for f, v in aggregated.items()}
        return dict(sorted(result.items(), key=lambda x: -x[1]))

    # 限制样本数（SHAP 计算耗时）
    import numpy as np
    X_bg = X_val_scaled[:min(max_samples, len(X_val_scaled))]

    aggregated = {}
    count = 0

    tree_models = ['rf1','rf2','rf3','et','gb','lgb1','lgb2','xgb1','xgb2','cat1','cat2']
    linear_models = ['lr', 'lda']

    for name, model in trained_models.items():
        try:
            if name in tree_models:
                explainer = shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_bg)
                # 二分类 shap_values 可能返回 list[class0, class1]，取 class1
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    shap_vals = shap_vals[1]
            elif name in linear_models:
                explainer = shap.LinearExplainer(model, X_bg)
                shap_vals = explainer.shap_values(X_bg)
            else:
                # 其它模型（MLP/KNN等）：用100个背景样本的KernelExplainer
                bg = shap.sample(X_bg, min(50, len(X_bg)))
                explainer = shap.KernelExplainer(model.predict_proba, bg)
                shap_vals = explainer.shap_values(X_bg[:30], nsamples=50)  # 限速
                if isinstance(shap_vals, list) and len(shap_vals) == 2:
                    shap_vals = shap_vals[1]

            mean_abs = np.abs(shap_vals).mean(axis=0)
            for f, v in zip(feature_cols, mean_abs):
                aggregated[f] = aggregated.get(f, 0.0) + float(v)
            count += 1
        except Exception as e:
            print(f"[SHAP] {name} 跳过: {e}")

    if not aggregated or count == 0:
        return {}

    result = {f: round(v / count, 6) for f, v in aggregated.items()}
    return dict(sorted(result.items(), key=lambda x: -x[1]))


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

    update_progress(5, "获取K线数据（需VPN连接Binance）...")
    try:
        df = get_klines(DATA_LIMIT)
    except Exception as e:
        print(f"[TRAIN] get_klines 异常: {e}")
        df = None
    if df is None or len(df) < 200:
        msg = '数据获取失败，请检查网络（是否开启VPN）' if df is None else f'数据不足（仅{len(df)}条，需200+）'
        with state_lock:
            state['training'] = False
            state['train_progress'] = 0
            state['train_status'] = msg
        # 删除训练标志文件
        try: os.remove('.training')
        except Exception: pass
        print(f"[TRAIN] {msg}")
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
            res = _train_single_model(model, X_train_s, y_train, X_val_s, y_val, timeout=60)
            if res['model'] is not None:
                trained[name] = res['model']
                print(f"  {name}: val_acc={res['acc']:.3f}")
            else:
                print(f"  {name}: 跳过（超时或失败）")
        except Exception as e:
            print(f"  {name} 失败: {e}")

    update_progress(82, "保存模型文件...")
    # 按 period 决定保存目录：period=10 时存入 models/（根目录），5/30 存入 models/5min/ models/30min/
    if period == 10:
        _save_dir = MODEL_DIR
    else:
        _save_dir = os.path.join(MODEL_DIR, f'{period}min')
    os.makedirs(_save_dir, exist_ok=True)
    for name, model in trained.items():
        path = os.path.join(_save_dir, f'btc_model_{name}.pkl')
        try:
            with open(path, 'wb') as f:
                pickle.dump({'model': model, 'scaler': scaler, 'features': feature_cols}, f)
        except Exception as e:
            print(f"  保存 {name} 失败: {e}")

    # 同时保存 _meta.json（训练元数据）
    try:
        import json as _json
        _meta = {
            'period': period,
            'feature_count': len(feature_cols),
            'trained_models': list(trained.keys()),
            'timestamp': __import__('datetime').datetime.utcnow().isoformat()
        }
        with open(os.path.join(_save_dir, '_meta.json'), 'w') as f:
            _json.dump(_meta, f, indent=2)
    except Exception:
        pass

    update_progress(90, "重新加载模型到内存...")
    new_models = {}
    new_scalers = {}
    for name in trained:
        path = os.path.join(_save_dir, f'btc_model_{name}.pkl')
        if os.path.exists(path):
            with open(path, 'rb') as f:
                d = pickle.load(f)
            new_models[name] = d.get('model')
            new_scalers[name] = d.get('scaler')

    # ── SHAP 特征重要性分析（P0 新增）──
    update_progress(95, "SHAP 特征重要性分析...")
    shap_result = _compute_shap_importance(trained, X_val_s, feature_cols)
    if shap_result:
        with state_lock:
            state['shap_importance'] = shap_result
        print(f"[SHAP] 分析完成，Top特征: {list(shap_result.keys())[:5]}")

    with state_lock:
        state['models'] = new_models
        state['scalers'] = new_scalers
        state['features'] = feature_cols
        state['loaded'] = True
        state['training'] = False
        state['train_progress'] = 100
        state['train_status'] = f'训练完成，{len(new_models)} 个模型已加载'
        # 删除训练标志文件
        try: os.remove('.training')
        except Exception: pass

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

@app.route('/assets/<path:filename>', methods=['GET'])
def serve_assets(filename):
    """提供 assets/ 目录下的静态文件（logo 等）"""
    # 1. 本地 assets/ 目录（开发模式 / python运行）
    assets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'assets')
    file_path = os.path.join(assets_dir, filename)
    if os.path.exists(file_path):
        return send_file(file_path)
    # 2. exe 同级 assets/ 目录（用户可替换 logo 等资源）
    if getattr(sys, 'frozen', False):
        exe_dir = os.path.dirname(sys.executable)
        exe_assets = os.path.join(exe_dir, 'assets', filename)
        if os.path.exists(exe_assets):
            return send_file(exe_assets)
    # 3. PyInstaller _MEIPASS 内置资源（打包时的默认）
    meipass_path = os.path.join(_BASE_DIR, 'assets', filename)
    if os.path.exists(meipass_path):
        return send_file(meipass_path)
    return "文件未找到", 404

@app.route('/api/status', methods=['GET'])
def api_status():
    with state_lock:
        n = len([m for m in state['models'].values() if m is not None])
        tf_models = state.get('tf_models', {})
        tf_info = {}
        total_tf = 0
        for p in sorted(tf_models.keys()):
            cnt = len([m for m in tf_models[p].values() if m is not None])
            tf_info[str(p) + 'min'] = cnt
            total_tf += cnt
        return jsonify({
            'loaded': state['loaded'],
            'model_count': n,
            'tf_models': tf_info,         # 多时间框架模型数量
            'total_model_count': n + total_tf,  # 总模型数量
            'training': state['training'],
            'train_progress': state['train_progress'],
            'train_status': state['train_status'],
        })

@app.route('/api/market_data', methods=['GET'])
def api_market_data():
    """返回外部市场数据：订单簿、OI、资金费率、贪婪恐惧指数"""
    try:
        bids, asks = fetch_orderbook()
        futures = fetch_futures_data()
        fng = fetch_fear_greed()

        ob_summary = {}
        if bids and asks:
            bid_vol = sum(q for _, q in bids[:10])
            ask_vol = sum(q for _, q in asks[:10])
            ob_summary = {
                'bid_vol_10': round(bid_vol, 4),
                'ask_vol_10': round(ask_vol, 4),
                'top_bid': bids[0][0],
                'top_ask': asks[0][0],
                'spread': round(asks[0][0] - bids[0][0], 2),
                'imbalance': round((bid_vol - ask_vol) / (bid_vol + ask_vol + 1e-9), 4),
            }

        return jsonify({
            'orderbook': ob_summary,
            'open_interest': futures['oi'] if not math.isnan(futures['oi']) else None,
            'funding_rate': futures['funding_rate'] if not math.isnan(futures['funding_rate']) else None,
            'fear_greed': fng,
            'fear_greed_label': _fng_label(fng),
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def _fng_label(val):
    """贪婪恐惧指数中文标签"""
    if val <= 25: return "极度恐惧"
    if val <= 45: return "恐惧"
    if val <= 55: return "中性"
    if val <= 75: return "贪婪"
    return "极度贪婪"

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

    # 直接用原始 df 做特征提取预测
    # 多时间框架：根据 period 使用对应周期的模型
    pred, conf, model_results = run_prediction(df, price, period=period)
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

    # 各模型票数统计（排除规则模型特殊键）
    ml_results = {k: v for k, v in model_results.items() if k != '__rule__'}
    up_count = sum(1 for r in ml_results.values() if r['pred'] == 1)
    dn_count = sum(1 for r in ml_results.values() if r['pred'] == 0)
    total_models = len(ml_results)
    # 一致性：方向一致的模型占比（含规则模型）
    rule_results = model_results.get('__rule__', {})
    rule_up = sum(1 for v in rule_results.values() if v.get('pred') == 1)
    rule_dn = sum(1 for v in rule_results.values() if v.get('pred') == 0)
    all_up = up_count + rule_up
    all_dn = dn_count + rule_dn
    all_total = all_up + all_dn
    agreement_rate = all_up / all_total if pred == 1 and all_total > 0 else (all_dn / all_total if all_total > 0 else 0.5)

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
        # 规则模型数据
        'technical_up': rule_up,
        'technical_down': rule_dn,
        'technical_details': {k: v['pred'] for k, v in rule_results.items()},
    }

    # 计算综合进场评分
    try:
        with _market_data_lock:
            ob_imb = _market_data_cache['orderbook'].get('bids', []) and \
                     _market_data_cache['orderbook'].get('asks', [])
            if ob_imb:
                bids = _market_data_cache['orderbook']['bids']
                asks = _market_data_cache['orderbook']['asks']
                bv = sum(q for _, q in bids[:10])
                av = sum(q for _, q in asks[:10])
                ob_val = (bv - av) / (bv + av + 1e-9)
            else:
                ob_val = 0
            fng_val = _market_data_cache['fear_greed'].get('value', 50) / 100.0
        entry_score = compute_entry_score(
            pred=int(pred), conf=conf_adjusted,
            agreement_rate=agreement_rate, trend_dir=trend_dir,
            against_trend=against_trend,
            ob_imbalance=ob_val, fear_greed=fng_val,
            near_integer=near
        )
        result['entry_score'] = entry_score
    except Exception as e:
        print(f"[SCORE] 计算进场评分异常: {e}")
        result['entry_score'] = {'score': 0, 'details': {}, 'grade': 'D'}

    with state_lock:
        state['last_prediction'] = result
        state['last_price'] = price
        state['last_signal'] = 'up' if pred == 1 else 'down'  # 供复利模块使用
        state['pending_model_preds'] = model_results  # 供在线学习权重更新使用

    # ── 持久化预测日志 ──
    try:
        pred_db.add_prediction(
            price=price,
            direction=result['direction'],
            confidence=result['confidence'],
            confidence_raw=result.get('confidence_raw'),
            agreement_rate=result.get('agreement_rate'),
            trend_dir=result.get('trend_dir'),
            trend_strength=result.get('trend_strength'),
            against_trend=result.get('against_trend', False),
            entry_score=result.get('entry_score', {}).get('score') if isinstance(result.get('entry_score'), dict) else None,
            period=period,
            model_detail={k: v for k, v in model_results.items() if k != '__rule__'}
        )
    except Exception as e:
        print(f"[DB] 记录预测日志失败: {e}")

    return jsonify(result)

# ========== 动态置信度阈值 API ==========
@app.route('/api/settle', methods=['POST'])
def api_settle():
    """赢/输结算，更新动态置信度阈值 + 记录到预测日志"""
    data = request.get_json(silent=True) or {}
    result = data.get('result', '')  # 'win' / 'loss'
    if result not in ('win', 'loss'):
        return jsonify({'error': '参数错误，需要 result=win/loss'}), 400

    # ── 结算最近一条未结算的预测 ──
    settle_info = None
    try:
        pending = pred_db.get_pending_predictions()
        if pending:
            latest = pending[-1]  # 最近一条到期的
            current_price = get_price()
            if current_price:
                settle_info = pred_db.settle_prediction(latest['id'], current_price, 'manual')
    except Exception as e:
        print(f"[DB] 结算预测日志失败: {e}")

    with state_lock:
        dc = state['dynamic_conf']
        prev_threshold = dc['current_threshold']

        if result == 'win':
            dc['win_streak'] += 1
            dc['loss_streak'] = 0
            # 连胜3次以上：阈值放宽
            if dc['win_streak'] >= 3:
                dc['current_threshold'] = max(0.58, dc['base_threshold'] - 0.07)
            elif dc['win_streak'] >= 2:
                dc['current_threshold'] = max(0.60, dc['base_threshold'] - 0.03)
            else:
                dc['current_threshold'] = dc['base_threshold']
        else:
            dc['win_streak'] = 0
            dc['loss_streak'] += 1
            # 连败：阈值收紧
            if dc['loss_streak'] >= 5:
                dc['current_threshold'] = 0.99  # 暂停信号
            else:
                dc['current_threshold'] = min(0.85, dc['base_threshold'] + dc['loss_streak'] * 0.03)

        changed = abs(dc['current_threshold'] - prev_threshold) > 0.001
        if changed:
            action = '放宽' if dc['current_threshold'] < prev_threshold else '收紧'
            print(f"[DYN-CONF] {result}→ 阈值{action}至 {(dc['current_threshold']*100).toFixed(0)}% (连胜{dc['win_streak']}/连败{dc['loss_streak']})")

    return jsonify({
        'threshold': round(state['dynamic_conf']['current_threshold'], 4),
        'win_streak': state['dynamic_conf']['win_streak'],
        'loss_streak': state['dynamic_conf']['loss_streak'],
        'paused': state['dynamic_conf']['current_threshold'] >= 0.95,
        'settle_info': settle_info,  # 预测日志结算详情
    })

@app.route('/api/dynamic-conf', methods=['GET'])
def api_dynamic_conf():
    """获取当前动态置信度状态"""
    with state_lock:
        dc = state['dynamic_conf']
    return jsonify({
        'threshold': round(dc['current_threshold'], 4),
        'base': round(dc['base_threshold'], 4),
        'win_streak': dc['win_streak'],
        'loss_streak': dc['loss_streak'],
        'paused': dc['current_threshold'] >= 0.95,
    })

@app.route('/api/train', methods=['POST'])
def api_train():
    with state_lock:
        if state['training']:
            return jsonify({'status': 'already_training', 'message': '训练进行中'}), 409
        state['training'] = True
        state['train_progress'] = 0
        state['train_status'] = '准备开始训练...'
        # 写训练标志文件，通知启动器不要开浏览器
        try:
            with open('.training', 'w') as f:
                f.write(str(int(time.time())))
        except Exception:
            pass

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
# ========== 自动补全 & 自动重训后台线程 ==========

# ========== 智能重训策略（P1 新增）==========
# 不再用固定100次计数，改为三重条件触发：
#   1. 胜率滑坡：近30次结算胜率 < 45%（且有统计意义）
#   2. 方向偏差：UP/DOWN 任一方向占比 > 70%（严重单边）
#   3. 数据漂移：近30次置信度均值 < 0.58（模型不确定性上升）
# 同时保留兜底：超过 RETRAIN_HARD_LIMIT 次结算无论如何都重训一次

RETRAIN_HARD_LIMIT   = 200   # 兜底：200次结算强制重训
RETRAIN_WIN_WINDOW   = 30    # 滑动窗口大小
RETRAIN_WIN_THRESH   = 0.45  # 胜率低于此值触发
RETRAIN_BIAS_THRESH  = 0.70  # 方向占比超过此值触发
RETRAIN_CONF_THRESH  = 0.58  # 置信度均值低于此值触发

_retrain_hard_counter = 0    # 结算计数（兜底用）

def _auto_retrain_check():
    """
    智能重训触发检测（每次自动结算后调用）。
    从数据库读取近 RETRAIN_WIN_WINDOW 条已结算记录，
    按三个条件判断是否需要重训。
    """
    global _retrain_hard_counter
    _retrain_hard_counter += 1

    # ── 从数据库查最近 N 条已结算记录 ──
    reason = None
    try:
        conn = pred_db._get_conn()
        rows = conn.execute(
            '''SELECT correct, direction, confidence FROM predictions
               WHERE settled = 1 ORDER BY id DESC LIMIT ?''',
            (RETRAIN_WIN_WINDOW,)
        ).fetchall()
        conn.close()

        if len(rows) >= 15:   # 至少15条才有统计意义
            corrects    = [r['correct'] for r in rows if r['correct'] is not None]
            directions  = [r['direction'] for r in rows]
            confidences = [r['confidence'] for r in rows if r['confidence'] is not None]

            win_rate  = sum(corrects) / len(corrects) if corrects else 1.0
            up_ratio  = directions.count('UP') / len(directions)
            down_ratio = 1 - up_ratio
            avg_conf  = sum(confidences) / len(confidences) if confidences else 1.0

            if win_rate < RETRAIN_WIN_THRESH:
                reason = f'胜率滑坡 {win_rate:.1%} < {RETRAIN_WIN_THRESH:.0%}'
            elif max(up_ratio, down_ratio) > RETRAIN_BIAS_THRESH:
                dom_dir = 'UP' if up_ratio > down_ratio else 'DOWN'
                reason = f'方向偏差 {dom_dir} {max(up_ratio,down_ratio):.1%} > {RETRAIN_BIAS_THRESH:.0%}'
            elif avg_conf < RETRAIN_CONF_THRESH:
                reason = f'置信度下降 avg={avg_conf:.3f} < {RETRAIN_CONF_THRESH}'
    except Exception as e:
        print(f"[SMART-RETRAIN] 检测异常: {e}")

    # 兜底强制
    if reason is None and _retrain_hard_counter >= RETRAIN_HARD_LIMIT:
        reason = f'兜底触发（已结算 {_retrain_hard_counter} 次）'

    if reason:
        _retrain_hard_counter = 0
        t = threading.Thread(target=_auto_retrain_worker, kwargs={'reason': reason}, daemon=True)
        t.start()
        print(f"[SMART-RETRAIN] 触发重训，原因: {reason}")


def auto_settle_loop():
    """
    后台线程：每60秒检查一次是否有到期的未结算预测，
    自动拉取当前价格并结算。这样即使不手动点结算，
    预测日志也会自动补全结果。
    """
    while True:
        try:
            time.sleep(60)  # 每60秒检查一次
            pending = pred_db.get_pending_predictions()
            if not pending:
                continue
            current_price = get_price()
            if not current_price:
                continue
            settled_count = 0
            for p in pending:
                try:
                    pred_db.settle_prediction(p['id'], current_price, 'auto')
                    settled_count += 1
                except Exception:
                    pass
            if settled_count > 0:
                print(f"[AUTO-SETTLE] 自动结算了 {settled_count} 条到期预测 (price={current_price})")
                # 同时触发自适应权重更新
                _auto_retrain_check()
        except Exception as e:
            print(f"[AUTO-SETTLE] 异常: {e}")

def _auto_retrain_worker(reason='auto'):
    """
    自动重训逻辑：
    1. 拉最新1000根K线训练新模型
    2. 对比新模型与当前模型的准确率
    3. 只有新模型更好才替换
    4. 记录重训日志到数据库
    """
    with state_lock:
        if state['training']:
            print("[AUTO-RETRAIN] 已有训练在进行中，跳过")
            return
        state['training'] = True
        state['train_progress'] = 0
        state['train_status'] = '自动重训中...'

    start_time = time.time()
    import datetime
    period = 10  # 默认重训周期

    try:
        # 获取旧模型准确率基准
        old_stats = pred_db.get_statistics(24)
        old_acc = old_stats['win_rate'] / 100.0  # 转为 0~1

        # ── 训练新模型（复用现有训练逻辑）──
        from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import RobustScaler
        from sklearn.svm import SVC
        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
        from sklearn.naive_bayes import GaussianNB
        from sklearn.calibration import CalibratedClassifierCV

        with state_lock:
            state['train_status'] = '自动重训：获取K线数据...'

        df = get_klines(DATA_LIMIT)
        if df is None or len(df) < 200:
            with state_lock:
                state['training'] = False
                state['train_status'] = '自动重训失败：数据获取失败'
            return

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
        try:
            from lightgbm import LGBMClassifier
            model_defs.append(('lgb1', LGBMClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbose=-1)))
            model_defs.append(('lgb2', LGBMClassifier(n_estimators=100, num_leaves=31, random_state=43, n_jobs=-1, verbose=-1)))
        except Exception: pass
        try:
            import xgboost as xgb
            model_defs.append(('xgb1', xgb.XGBClassifier(n_estimators=200, random_state=42, n_jobs=-1, verbosity=0, eval_metric='logloss')))
            model_defs.append(('xgb2', xgb.XGBClassifier(n_estimators=100, max_depth=4, random_state=43, n_jobs=-1, verbosity=0, eval_metric='logloss')))
        except Exception: pass
        try:
            from catboost import CatBoostClassifier
            model_defs.append(('cat1', CatBoostClassifier(iterations=200, random_state=42, verbose=False)))
            model_defs.append(('cat2', CatBoostClassifier(iterations=100, depth=4, random_state=43, verbose=False)))
        except Exception: pass

        trained = {}
        total = len(model_defs)
        for i, (name, model) in enumerate(model_defs):
            pct = 10 + int(70 * i / total)
            with state_lock:
                state['train_progress'] = pct
                state['train_status'] = f'自动重训：训练 {name.upper()} ({i+1}/{total})...'
            try:
                res = _train_single_model(model, X_train_s, y_train, X_val_s, y_val, timeout=60)
                if res['model'] is not None:
                    trained[name] = res['model']
            except Exception:
                pass

        if not trained:
            with state_lock:
                state['training'] = False
                state['train_status'] = '自动重训失败：无模型训练成功'
            return

        # ── 评估新模型验证集准确率 ──
        new_acc = 0
        for name, model in trained.items():
            try:
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(X_val_s)
                    preds = np.argmax(proba, axis=1)
                else:
                    preds = model.predict(X_val_s)
                acc = (preds == y_val).mean()
                new_acc += acc
            except Exception:
                pass
        new_acc = new_acc / max(1, len(trained))

        # ── 决定是否替换 ──
        # 策略：新模型验证集准确率 > 旧模型近24h实际胜率 × 0.8 才替换
        # （因为验证集准确率和实际胜率有差距，给一定容差）
        threshold = max(0.45, old_acc * 0.8) if old_acc > 0.3 else 0.45
        should_replace = new_acc >= threshold

        duration = round(time.time() - start_time, 1)

        with state_lock:
            if should_replace:
                # 替换模型
                new_models = {}
                new_scalers = {}
                for name in trained:
                    path = os.path.join(MODEL_DIR, f'btc_model_{name}.pkl')
                    try:
                        with open(path, 'wb') as f:
                            pickle.dump({'model': trained[name], 'scaler': scaler, 'features': feature_cols}, f)
                        with open(path, 'rb') as f:
                            d = pickle.load(f)
                        new_models[name] = d.get('model')
                        new_scalers[name] = d.get('scaler')
                    except Exception:
                        pass
                if new_models:
                    state['models'] = new_models
                    state['scalers'] = new_scalers
                    state['features'] = feature_cols
                msg = f'自动重训完成：{len(trained)}个模型训练，{len(new_models)}个替换 (新acc={new_acc:.1%} > 阈值{threshold:.1%})'
                models_kept = len(new_models)
            else:
                msg = f'自动重训完成但未替换：新acc={new_acc:.1%} < 阈值{threshold:.1%}（保留旧模型）'
                models_kept = 0
            state['training'] = False
            state['train_progress'] = 100
            state['train_status'] = msg
            try: os.remove('.training')
            except Exception: pass

        print(f"[AUTO-RETRAIN] {msg}")

        # 记录重训日志
        pred_db.add_retrain_log(
            period=period, old_acc=round(old_acc, 4), new_acc=round(new_acc, 4),
            models_trained=len(trained), models_kept=models_kept,
            duration_sec=duration, reason=reason
        )

    except Exception as e:
        with state_lock:
            state['training'] = False
            state['train_progress'] = 0
            state['train_status'] = f'自动重训异常: {e}'
            try: os.remove('.training')
            except Exception: pass
        print(f"[AUTO-RETRAIN] 异常: {e}")


def main():
    print("=" * 50)
    print("复利引擎后端服务 v1.0")
    print(f"模型目录: {MODEL_DIR}")
    print(f"预测日志: {DB_PATH}")
    print("=" * 50)

    # 后台加载模型
    t = threading.Thread(target=load_all_models, daemon=True)
    t.start()

    # 启动自动补全后台线程（每60秒检查到期预测并自动结算）
    t_settle = threading.Thread(target=auto_settle_loop, daemon=True)
    t_settle.start()
    print("[INFO] 自动补全线程已启动（每60秒结算到期预测）")

    port = int(os.environ.get('PORT', 7788))

    # 启动前检查端口是否被占用
    # 注意：端口检测由 launcher.py 负责，这里只是最后一道保险
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(('0.0.0.0', port))
        s.close()
    except OSError:
        s.close()
        print(f"[ERROR] 端口 {port} 已被占用。复利引擎已在运行中，请勿重复启动。")
        return

    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)

# ========== 回测系统 API ==========
@app.route('/api/backtest', methods=['GET'])
def api_backtest():
    """
    简单滑动窗口回测。
    用已加载的模型在历史K线上做模拟预测，统计胜率。
    参数：
      period: 预测周期（分钟），默认5
      samples: 回测样本数，默认100
    """
    with state_lock:
        models = state['models']
        scalers = state['scalers']
        per_model_features_map = state.get('per_model_features', {})
    if not models:
        return jsonify({'error': '模型未加载'}), 503

    try:
        period = int(request.args.get('period', 5))
        samples = min(int(request.args.get('samples', 100)), 200)
    except Exception:
        period, samples = 5, 100

    # 获取足够的历史K线
    need = 300  # 滑动窗口用200，前面留100做特征预热
    df = get_klines(need)
    if df is None or len(df) < need:
        return jsonify({'error': 'K线数据不足'}), 503

    results = []
    lookback = 200  # 用于计算特征的K线数量

    for i in range(lookback, len(df) - period):
        if len(results) >= samples:
            break

        window = df.iloc[i - lookback:i + 1].copy()
        try:
            feat = calculate_features(window)
            feat_cols = get_feature_cols(feat)

            votes_up = 0.0
            votes_down = 0.0
            n_voted = 0

            for name, model in models.items():
                if model is None:
                    continue
                try:
                    scaler = scalers.get(name)
                    pmf = per_model_features_map.get(name)
                    if pmf:
                        fc = [c for c in pmf if c in feat.columns]
                    else:
                        fc = [c for c in feat_cols if c in feat.columns]
                    if not fc:
                        continue
                    row = feat[fc].dropna().iloc[-1:]
                    if row.empty:
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
                    if pred == 1:
                        votes_up += conf
                    else:
                        votes_down += conf
                    n_voted += 1
                except Exception:
                    continue

            if n_voted < 5:
                continue

            total = votes_up + votes_down
            if total == 0:
                continue

            pred_dir = 1 if votes_up > votes_down else 0
            pred_conf = max(votes_up, votes_down) / total

            # 真实方向：period 根后的涨跌
            actual_close = df.iloc[min(i + period, len(df) - 1)]['close']
            current_close = df.iloc[i]['close']
            actual_dir = 1 if actual_close > current_close else 0
            win = (pred_dir == actual_dir)
            pct_change = (actual_close - current_close) / current_close * 100

            results.append({
                'index': i,
                'time': str(df.iloc[i]['open_time']),
                'price': round(current_close, 2),
                'pred': pred_dir,
                'actual': actual_dir,
                'conf': round(pred_conf, 3),
                'win': win,
                'pct': round(pct_change, 4),
            })
        except Exception:
            continue

    if not results:
        return jsonify({'error': '回测无有效样本'}), 500

    # 统计
    wins = [r for r in results if r['win']]
    losses = [r for r in results if not r['win']]
    win_rate = len(wins) / len(results) * 100

    # 连胜/连输
    max_streak_w = max_streak_l = cur_w = cur_l = 0
    for r in results:
        if r['win']:
            cur_w += 1; cur_l = 0
        else:
            cur_l += 1; cur_w = 0
        max_streak_w = max(max_streak_w, cur_w)
        max_streak_l = max(max_streak_l, cur_l)

    # 累计盈亏（模拟0.85赔率，每次1U）
    cumulative = []
    balance = 0
    for r in results:
        if r['win']:
            balance += 0.85
        else:
            balance -= 1.0
        cumulative.append(round(balance, 2))

    # 按置信度分桶统计胜率
    conf_buckets = {'60-65': [0,0], '65-70': [0,0], '70-75': [0,0], '75-80': [0,0], '80+': [0,0]}
    for r in results:
        c = r['conf'] * 100
        if c < 65: key = '60-65'
        elif c < 70: key = '65-70'
        elif c < 75: key = '70-75'
        elif c < 80: key = '75-80'
        else: key = '80+'
        conf_buckets[key][0] += 1
        if r['win']: conf_buckets[key][1] += 1
    conf_stats = {}
    for k, (total, w) in conf_buckets.items():
        conf_stats[k] = {'total': total, 'wins': w, 'rate': round(w/total*100, 1) if total > 0 else 0}

    return jsonify({
        'total': len(results),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': round(win_rate, 1),
        'max_streak_win': max_streak_w,
        'max_streak_loss': max_streak_l,
        'cumulative': cumulative,
        'conf_stats': conf_stats,
        'period': period,
        'samples': len(results),
        'details': results[-20:],  # 最近20条详细数据
    })

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
        perf_windows = {k: list(v) for k, v in state['model_perf_window'].items()}
    # 持久化权重到 SQLite（关机不丢）
    try:
        pred_db.save_adaptive_weights(weights, perf_windows)
    except Exception as e:
        print(f"[DB] 保存自适应权重失败: {e}")
    return jsonify({"ok": True, "weights": weights})


@app.route('/api/adaptive/weights', methods=['GET'])
def api_adaptive_weights():
    """查询当前各模型自适应权重"""
    with state_lock:
        weights = dict(state['adaptive_weights'])
        perf = {k: {"count": len(v), "winrate": round(sum(v)/len(v)*100, 1) if v else 0}
                for k, v in state['model_perf_window'].items()}
    return jsonify({"ok": True, "weights": weights, "performance": perf})


# ========== 预测日志 & 统计 API ==========

@app.route('/api/stats', methods=['GET'])
def api_stats():
    """获取预测统计信息（胜率、置信度分桶等）"""
    try:
        hours = int(request.args.get('hours', 24))
        hours = max(1, min(720, hours))  # 限制 1h~30d
    except Exception:
        hours = 24
    stats = pred_db.get_statistics(hours)
    stats['unsettled_count'] = pred_db.get_unsettled_count()
    return jsonify({"ok": True, **stats})


@app.route('/api/predictions', methods=['GET'])
def api_predictions():
    """获取最近的预测记录"""
    try:
        limit = int(request.args.get('limit', 50))
        limit = max(1, min(200, limit))
    except Exception:
        limit = 50
    settled_only = request.args.get('settled') == '1'
    rows = pred_db.get_recent_predictions(limit, settled_only)
    return jsonify({"ok": True, "predictions": rows, "count": len(rows)})


@app.route('/api/predictions/settle_pending', methods=['POST'])
def api_settle_pending():
    """
    手动结算所有到期的未结算预测
    后台线程会自动做，但这个接口允许用户手动触发
    """
    pending = pred_db.get_pending_predictions()
    if not pending:
        return jsonify({"ok": True, "settled": 0, "message": "没有到期的未结算预测"})
    current_price = get_price()
    if not current_price:
        return jsonify({"error": "无法获取当前价格"}), 503
    settled = 0
    correct = 0
    for p in pending:
        try:
            info = pred_db.settle_prediction(p['id'], current_price, 'manual')
            if info:
                settled += 1
                if info['correct']:
                    correct += 1
        except Exception:
            pass
    return jsonify({
        "ok": True, "settled": settled, "correct": correct,
        "message": f"结算了 {settled} 条预测，其中 {correct} 条正确"
    })


@app.route('/api/retrain/history', methods=['GET'])
def api_retrain_history():
    """获取自动重训历史"""
    rows = pred_db.get_retrain_history()
    return jsonify({"ok": True, "history": rows})


# ========== SHAP 特征重要性 API（P0 新增）==========
@app.route('/api/shap', methods=['GET'])
def api_shap():
    """
    返回最近一次训练的 SHAP 特征重要性。
    参数：
      top_n: 返回前N个特征，默认30，最大138
    """
    try:
        top_n = min(int(request.args.get('top_n', 30)), 200)
    except Exception:
        top_n = 30

    with state_lock:
        importance = dict(state.get('shap_importance', {}))

    if not importance:
        return jsonify({"ok": False, "msg": "SHAP 数据尚未生成，请先完成一次训练", "data": []})

    # 取 top_n
    items = sorted(importance.items(), key=lambda x: -x[1])[:top_n]
    total = sum(v for _, v in items)
    result = [
        {"feature": f, "importance": round(v, 6), "pct": round(v / total * 100, 2) if total > 0 else 0}
        for f, v in items
    ]
    return jsonify({"ok": True, "data": result, "total_features": len(importance)})


# ========== 模型诊断 API（P1 新增）==========
@app.route('/api/diagnostics', methods=['GET'])
def api_diagnostics():
    """
    模型诊断面板：
    - 各模型自适应权重 + 近期胜率
    - 分时段胜率（亚盘/欧盘/美盘）
    - 分周期胜率（5/10/30min）
    - 方向偏差统计（UP/DOWN 比例）
    - 智能重训状态（触发条件当前值）
    """
    try:
        hours = int(request.args.get('hours', 48))
        hours = max(1, min(720, hours))
    except Exception:
        hours = 48

    import datetime

    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).isoformat()

    try:
        conn = pred_db._get_conn()

        # ── 1. 分周期胜率 ──
        period_stats = conn.execute('''
            SELECT period,
                   COUNT(*) as total,
                   SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins,
                   AVG(confidence) as avg_conf,
                   SUM(CASE WHEN direction='UP' THEN 1 ELSE 0 END) as up_cnt,
                   SUM(CASE WHEN direction='DOWN' THEN 1 ELSE 0 END) as down_cnt
            FROM predictions
            WHERE settled=1 AND timestamp >= ?
            GROUP BY period ORDER BY period
        ''', (cutoff,)).fetchall()

        by_period = []
        for r in period_stats:
            total = r['total'] or 1
            by_period.append({
                "period": r['period'],
                "total": total,
                "wins": r['wins'] or 0,
                "win_rate": round((r['wins'] or 0) / total * 100, 1),
                "avg_conf": round(r['avg_conf'] or 0, 3),
                "up_ratio": round((r['up_cnt'] or 0) / total * 100, 1),
                "down_ratio": round((r['down_cnt'] or 0) / total * 100, 1),
            })

        # ── 2. 分时段胜率（UTC+8 小时段：亚盘 8-16, 欧盘 16-22, 美盘 22-8）──
        # SQLite 内用 strftime 取小时（UTC），再换算 UTC+8
        session_rows = conn.execute('''
            SELECT
                CASE
                    WHEN ((CAST(strftime('%H', timestamp) AS INTEGER) + 8) % 24) >= 8
                     AND ((CAST(strftime('%H', timestamp) AS INTEGER) + 8) % 24) < 16
                    THEN '亚洲盘(08-16)'
                    WHEN ((CAST(strftime('%H', timestamp) AS INTEGER) + 8) % 24) >= 16
                     AND ((CAST(strftime('%H', timestamp) AS INTEGER) + 8) % 24) < 22
                    THEN '欧洲盘(16-22)'
                    ELSE '美洲盘(22-08)'
                END as session,
                COUNT(*) as total,
                SUM(CASE WHEN correct=1 THEN 1 ELSE 0 END) as wins
            FROM predictions
            WHERE settled=1 AND timestamp >= ?
            GROUP BY session ORDER BY session
        ''', (cutoff,)).fetchall()

        by_session = [
            {
                "session": r['session'],
                "total": r['total'] or 0,
                "wins": r['wins'] or 0,
                "win_rate": round((r['wins'] or 0) / max(1, r['total'] or 1) * 100, 1),
            }
            for r in session_rows
        ]

        # ── 3. 近30条滚动统计（用于智能重训阈值对比）──
        recent_rows = conn.execute('''
            SELECT correct, direction, confidence FROM predictions
            WHERE settled=1 ORDER BY id DESC LIMIT 30
        ''').fetchall()

        recent_total = len(recent_rows)
        recent_wins  = sum(1 for r in recent_rows if r['correct'] == 1)
        recent_wr    = round(recent_wins / max(1, recent_total) * 100, 1)
        recent_up    = sum(1 for r in recent_rows if r['direction'] == 'UP')
        recent_dn    = recent_total - recent_up
        recent_conf  = round(sum(r['confidence'] or 0 for r in recent_rows) / max(1, recent_total), 3)

        conn.close()

        # ── 4. 各模型权重 & 性能 ──
        with state_lock:
            weights  = dict(state.get('adaptive_weights', {}))
            perf_win = state.get('model_perf_window', {})
            perf = {}
            for k, v in perf_win.items():
                lst = list(v)
                if lst:
                    perf[k] = {
                        "count": len(lst),
                        "win_rate": round(sum(lst) / len(lst) * 100, 1),
                        "weight": round(weights.get(k, 1.0), 3),
                    }

        # ── 5. 智能重训触发条件当前状态 ──
        retrain_status = {
            "win_rate_30":     recent_wr,
            "win_thresh":      RETRAIN_WIN_THRESH * 100,
            "win_triggered":   recent_wr < RETRAIN_WIN_THRESH * 100 and recent_total >= 15,
            "direction_bias":  round(max(recent_up, recent_dn) / max(1, recent_total) * 100, 1),
            "bias_thresh":     RETRAIN_BIAS_THRESH * 100,
            "bias_triggered":  max(recent_up, recent_dn) / max(1, recent_total) > RETRAIN_BIAS_THRESH and recent_total >= 15,
            "avg_conf_30":     recent_conf,
            "conf_thresh":     RETRAIN_CONF_THRESH,
            "conf_triggered":  recent_conf < RETRAIN_CONF_THRESH and recent_total >= 15,
            "hard_counter":    _retrain_hard_counter,
            "hard_limit":      RETRAIN_HARD_LIMIT,
        }
        retrain_status["any_triggered"] = any([
            retrain_status["win_triggered"],
            retrain_status["bias_triggered"],
            retrain_status["conf_triggered"],
        ])

        return jsonify({
            "ok": True,
            "hours": hours,
            "by_period": by_period,
            "by_session": by_session,
            "model_performance": perf,
            "recent_30": {
                "total": recent_total, "wins": recent_wins,
                "win_rate": recent_wr, "up": recent_up, "down": recent_dn,
                "avg_conf": recent_conf,
            },
            "retrain_status": retrain_status,
        })

    except Exception as e:
        return jsonify({"ok": False, "error": str(e)}), 500


@app.route('/api/data/export', methods=['GET'])
def api_data_export():
    """
    导出预测日志数据（供数据回传或分析用）
    返回 CSV 格式
    """
    try:
        hours = int(request.args.get('hours', 168))  # 默认导出最近7天
        hours = max(1, min(8760, hours))
    except Exception:
        hours = 168
    import datetime
    import io as _io
    cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).isoformat()
    conn = pred_db._get_conn()
    try:
        rows = conn.execute('''
            SELECT timestamp, price, direction, confidence, confidence_raw,
                   agreement_rate, trend_dir, trend_strength, entry_score, period,
                   actual_price, actual_change, actual_direction, correct, settled
            FROM predictions WHERE timestamp >= ? ORDER BY id
        ''', (cutoff,)).fetchall()
        if not rows:
            return jsonify({"ok": False, "message": "无数据可导出"})
        buf = _io.StringIO()
        cols = rows[0].keys()
        buf.write(','.join(cols) + '\n')
        for r in rows:
            vals = [str(r[c] if r[c] is not None else '') for c in cols]
            buf.write(','.join(vals) + '\n')
        from flask import make_response
        resp = make_response(buf.getvalue(), 200)
        resp.headers['Content-Type'] = 'text/csv; charset=utf-8'
        resp.headers['Content-Disposition'] = f'attachment; filename=prediction_log_{hours}h.csv'
        return resp
    finally:
        conn.close()


# ========== 数据回传配置 ==========
# 仓库地址（改成你自己的）
CONTRIBUTE_REPO = "https://github.com/OxMagic/compound-engine-data"
# GitHub Personal Access Token（免费，不需要充值）
# 生成教程：https://github.com/settings/tokens → Generate new token (classic)
#   勾选 public_repo 权限即可，其他不用勾
# 生成后粘贴到这里 ↓
CONTRIBUTE_TOKEN = ""


@app.route('/api/data/contribute', methods=['POST'])
def api_data_contribute():
    """
    数据回传接口（匿名贡献预测数据）
    通过 GitHub Issues API 提交已结算的预测结果
    仅上传：时间、周期、方向、置信度、实际涨跌（不传任何用户身份信息）
    """
    try:
        hours = int(request.args.get('hours', 24))
        hours = max(1, min(168, hours))  # 最多回传7天
    except Exception:
        hours = 24

    if not CONTRIBUTE_TOKEN:
        return jsonify({"ok": False, "message": "未配置 CONTRIBUTE_TOKEN，数据回传暂不可用"})

    import datetime
    conn = pred_db._get_conn()
    try:
        cutoff = (datetime.datetime.utcnow() - datetime.timedelta(hours=hours)).isoformat()
        rows = conn.execute('''
            SELECT timestamp, price, direction, confidence, period,
                   actual_change, actual_direction, correct
            FROM predictions WHERE settled = 1 AND timestamp >= ?
            ORDER BY id
        ''', (cutoff,)).fetchall()

        if not rows:
            return jsonify({"ok": True, "summary": {"total": 0}, "ready": False, "message": "无已结算数据"})

        # 构建数据摘要
        correct = sum(1 for r in rows if r['correct'] == 1)
        wrong = sum(1 for r in rows if r['correct'] == 0)
        summary = {
            'total': len(rows),
            'correct': correct,
            'wrong': wrong,
            'win_rate': round(correct / len(rows) * 100, 1),
            'period_hours': hours,
        }

        # 提交到 GitHub Issues
        try:
            data_lines = []
            for r in rows:
                data_lines.append(
                    f"{r['timestamp']}|{r['period'] or '-'}|{r['direction']}|"
                    f"{float(r['confidence']) if r['confidence'] else '-'}|"
                    f"{float(r['actual_change']) if r['actual_change'] else '-'}|"
                    f"{r['actual_direction'] or '-'}|{r['correct']}"
                )

            issue_title = f"数据回传 {datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M')} | {summary['total']}条 胜率{summary['win_rate']}%"
            issue_body = (
                f"## 复利引擎匿名预测数据\n\n"
                f"- **条数**: {summary['total']}\n"
                f"- **胜率**: {summary['win_rate']}%\n"
                f"- **周期范围**: {hours}h\n\n"
                f"| 时间 | 周期 | 方向 | 置信度 | 实际涨跌 | 实际方向 | 对错 |\n"
                f"|------|------|------|--------|----------|----------|------|\n"
                + "\n".join(f"| {line.replace('|', ' | ')} |" for line in data_lines)
                + "\n\n---\n*由复利引擎自动提交*"
            )

            api_url = CONTRIBUTE_REPO.replace("https://github.com/", "https://api.github.com/repos/") + "/issues"
            headers = {
                "Authorization": f"token {CONTRIBUTE_TOKEN}",
                "Accept": "application/vnd.github.v3+json",
            }
            resp = requests.post(api_url, headers=headers, json={
                "title": issue_title,
                "body": issue_body,
                "labels": ["data-contrib"]
            }, timeout=15)

            if resp.status_code == 201:
                issue_url = resp.json().get('html_url', '')
                print(f"[CONTRIBUTE] 成功提交 {summary['total']} 条数据 → {issue_url}")
                return jsonify({"ok": True, "summary": summary, "ready": True, "github": "ok", "url": issue_url})
            elif resp.status_code == 401:
                print("[CONTRIBUTE] Token 无效或过期")
                return jsonify({"ok": False, "summary": summary, "message": "Token无效或过期，请重新生成"})
            elif resp.status_code == 404:
                print(f"[CONTRIBUTE] 仓库不存在: {CONTRIBUTE_REPO}")
                return jsonify({"ok": False, "summary": summary, "message": "仓库不存在，请先创建"})
            else:
                print(f"[CONTRIBUTE] GitHub 异常: HTTP {resp.status_code}")
                return jsonify({"ok": False, "summary": summary, "message": f"GitHub HTTP {resp.status_code}"})
        except requests.exceptions.ConnectionError:
            print("[CONTRIBUTE] 网络错误，可能需要VPN")
            return jsonify({"ok": False, "summary": summary, "message": "网络错误，请检查VPN连接"})
        except Exception as e:
            print(f"[CONTRIBUTE] 提交失败: {e}")
            return jsonify({"ok": False, "summary": summary, "message": str(e)})
    finally:
        conn.close()


@app.route('/api/data/contribute/enable', methods=['POST'])
def api_contribute_toggle():
    """开关数据回传（仅记录用户偏好，前端存储即可）"""
    data = request.json or {}
    enabled = data.get('enabled', False)
    print(f"[DATA-CONTRIBUTE] 数据回传: {'开启' if enabled else '关闭'}")
    return jsonify({"ok": True, "enabled": enabled})




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
