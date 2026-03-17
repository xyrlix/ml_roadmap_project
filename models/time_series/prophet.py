# Prophet 时间序列预测模型
# 由 Meta 开发的加法模型：趋势 + 周期 + 节假日效应
# 使用 statsmodels 模拟 Prophet 分解思路（避免额外安装依赖）

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from statsmodels.tsa.seasonal import seasonal_decompose
from utils import get_results_path, save_and_close


# ──────────────────────────────────────────────────────────────────────
# Prophet 核心思想的轻量级实现（不依赖 fbprophet）
#   y(t) = trend(t) + seasonal(t) + holiday(t) + ε
# ──────────────────────────────────────────────────────────────────────

class SimpleProphet:
    """
    加法时间序列分解模型（Prophet-style）：
      - 趋势: 分段线性回归
      - 周期: 傅里叶级数近似
      - 节假日: 哑变量
    """
    def __init__(self, seasonality_period=52, n_fourier=5,
                 n_changepoints=10, holiday_dates=None):
        self.period     = seasonality_period
        self.K          = n_fourier          # 傅里叶阶数
        self.n_cp       = n_changepoints     # 变化点数量
        self.holidays   = holiday_dates or []
        self.ridge      = Ridge(alpha=0.1)
        self.scaler     = StandardScaler()

    def _fourier_features(self, t):
        """生成傅里叶基函数特征 (2*K 列)"""
        cols = []
        for k in range(1, self.K + 1):
            cols.append(np.sin(2 * np.pi * k * t / self.period))
            cols.append(np.cos(2 * np.pi * k * t / self.period))
        return np.column_stack(cols)

    def _changepoint_features(self, t, T):
        """分段线性趋势变化点特征"""
        cp_locs = np.linspace(0, T, self.n_cp + 2)[1:-1]
        cols = []
        for cp in cp_locs:
            cols.append(np.maximum(0, t - cp))
        return np.column_stack(cols) if cols else np.zeros((len(t), 1))

    def _holiday_features(self, t):
        """节假日哑变量"""
        h_feat = np.zeros(len(t))
        for hd in self.holidays:
            h_feat[np.abs(t - hd) < 1] = 1.0
        return h_feat.reshape(-1, 1)

    def _build_features(self, t, T):
        t = np.asarray(t, dtype=float)
        T_feat  = t.reshape(-1, 1) / T           # 线性趋势
        CP_feat = self._changepoint_features(t, T)
        F_feat  = self._fourier_features(t)
        H_feat  = self._holiday_features(t)
        return np.hstack([T_feat, CP_feat, F_feat, H_feat])

    def fit(self, y):
        self.T = len(y)
        t = np.arange(self.T, dtype=float)
        Phi = self._build_features(t, self.T)
        Phi_s = self.scaler.fit_transform(Phi)
        self.ridge.fit(Phi_s, y)
        return self

    def predict(self, t_future=None, horizon=0):
        if t_future is None:
            t_future = np.arange(self.T, self.T + horizon, dtype=float)
        else:
            t_future = np.asarray(t_future, dtype=float)
        Phi = self._build_features(t_future, self.T)
        Phi_s = self.scaler.transform(Phi)
        return self.ridge.predict(Phi_s)

    def fitted(self):
        t = np.arange(self.T, dtype=float)
        Phi = self._build_features(t, self.T)
        return self.ridge.predict(self.scaler.transform(Phi))


def generate_prophet_data(n_weeks=156, seed=42):
    """生成含趋势 + 年季节 + 周季节 + 节假日脉冲的周数据"""
    np.random.seed(seed)
    t = np.arange(n_weeks)
    trend    = 50 + 0.3 * t
    yearly   = 10 * np.sin(2 * np.pi * t / 52)
    weekly   = 3  * np.sin(2 * np.pi * t / 4)
    noise    = np.random.normal(0, 2, n_weeks)
    # 模拟节假日脉冲
    holidays = [26, 78, 130]
    h_effect = np.zeros(n_weeks)
    for hd in holidays:
        if hd < n_weeks:
            h_effect[hd] = 15
    return trend + yearly + weekly + h_effect + noise, holidays


def prophet():
    """Prophet-style 加法分解时间序列预测实现"""
    print("Prophet 时间序列预测模型运行中...\n")

    # 1. 数据准备
    print("1. 生成含趋势+季节+节假日的时间序列...")
    N_WEEKS = 156
    series, holiday_dates = generate_prophet_data(n_weeks=N_WEEKS)
    t = np.arange(N_WEEKS)
    print(f"   序列长度: {N_WEEKS} 周  节假日位置: {holiday_dates}")

    # 2. 训练/测试划分
    train_size = 130
    train_series = series[:train_size]
    test_series  = series[train_size:]
    print(f"   训练集: {train_size} 周  测试集: {N_WEEKS - train_size} 周")

    # 3. 拟合 SimpleProphet
    print("2. 拟合 Prophet-style 加法模型...")
    model = SimpleProphet(
        seasonality_period=52,
        n_fourier=5,
        n_changepoints=12,
        holiday_dates=holiday_dates
    )
    model.fit(train_series)

    # 在训练集上拟合值
    fitted_train = model.fitted()

    # 预测测试集
    t_test = np.arange(train_size, N_WEEKS, dtype=float)
    preds  = model.predict(t_future=t_test)

    rmse = np.sqrt(mean_squared_error(test_series, preds))
    mae  = mean_absolute_error(test_series, preds)
    print(f"   测试集 RMSE={rmse:.4f}  MAE={mae:.4f}")

    # 4. 时序分解（statsmodels）
    print("3. 时序分解（趋势 + 季节 + 残差）...")
    decomp = seasonal_decompose(series, model='additive', period=52, extrapolate_trend='freq')

    # 5. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── 子图1：原始序列 + Prophet 拟合 ──
    ax = axes[0, 0]
    ax.plot(t, series, color='steelblue', linewidth=1.0, alpha=0.7, label='真实值')
    ax.plot(np.arange(train_size), fitted_train, color='darkorange',
            linewidth=1.5, linestyle='--', label='拟合值（训练集）')
    ax.plot(t_test, preds, color='tomato', linewidth=2.0,
            label=f'预测值（测试集, RMSE={rmse:.2f}）')
    ax.axvline(x=train_size, color='gray', linestyle=':', linewidth=1.5)
    for hd in holiday_dates:
        ax.axvline(x=hd, color='green', linestyle=':', alpha=0.5)
    ax.set_title('Prophet-style 预测')
    ax.set_xlabel('Week')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图2：时序分解趋势 ──
    ax = axes[0, 1]
    ax.plot(t, decomp.trend, color='steelblue', linewidth=1.8, label='趋势')
    ax.set_title('趋势分量（Trend）')
    ax.set_xlabel('Week')
    ax.set_ylabel('Trend')
    ax.grid(True, alpha=0.3)

    # ── 子图3：季节性分量 ──
    ax = axes[1, 0]
    ax.plot(t[:52*2], decomp.seasonal[:52*2],
            color='darkorange', linewidth=1.5, label='季节性')
    ax.set_title('季节性分量（前 2 年）')
    ax.set_xlabel('Week')
    ax.set_ylabel('Seasonal')
    ax.grid(True, alpha=0.3)

    # ── 子图4：残差 ──
    ax = axes[1, 1]
    resid = decomp.resid
    ax.plot(t, resid, color='gray', linewidth=0.8, alpha=0.8)
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.fill_between(t, resid, 0, alpha=0.2, color='gray')
    ax.set_title('残差（Residual）')
    ax.set_xlabel('Week')
    ax.set_ylabel('Residual')
    ax.grid(True, alpha=0.3)

    plt.suptitle('Prophet-style 时间序列分解与预测', fontsize=14, y=1.01)
    save_path = get_results_path('prophet_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    prophet()
