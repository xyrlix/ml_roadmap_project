# ARIMA 时间序列模型
# 自回归积分滑动平均模型，适合平稳/可差分为平稳的时序数据

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error, mean_absolute_error
from utils import get_results_path, save_and_close


def generate_arima_data(n=200, seed=42):
    """生成 ARIMA(1,1,1) 过程的时间序列"""
    np.random.seed(seed)
    errors = np.random.normal(0, 1, n + 100)
    # AR(1) + MA(1) in integrated form
    series = np.zeros(n + 100)
    for t in range(2, n + 100):
        series[t] = (0.8 * series[t-1]          # AR(1)
                     + errors[t]                 # 白噪声
                     - 0.4 * errors[t-1])        # MA(1)
    # 积分（一阶差分逆运算）
    series = np.cumsum(series[100:]) * 0.1
    return series


def arima():
    """ARIMA 时间序列预测实现"""
    print("ARIMA 模型运行中...\n")

    # 1. 生成数据
    print("1. 生成/准备时间序列数据...")
    series = generate_arima_data(n=200)
    print(f"   序列长度: {len(series)}")

    # 2. 平稳性检验（ADF）
    print("2. ADF 平稳性检验...")
    adf_raw = adfuller(series)
    diff1   = np.diff(series)
    adf_d1  = adfuller(diff1)
    print(f"   原始序列  ADF p值: {adf_raw[1]:.4f}  "
          f"{'平稳' if adf_raw[1] < 0.05 else '非平稳'}")
    print(f"   一阶差分  ADF p值: {adf_d1[1]:.4f}  "
          f"{'平稳' if adf_d1[1] < 0.05 else '非平稳'}")

    # 3. 训练/测试划分
    train_size = int(len(series) * 0.8)
    train, test = series[:train_size], series[train_size:]
    print(f"3. 训练集: {len(train)}  测试集: {len(test)}")

    # 4. 拟合 ARIMA(1,1,1)
    print("4. 拟合 ARIMA(1,1,1) 模型...")
    model = ARIMA(train, order=(1, 1, 1))
    result = model.fit()
    print(f"   AIC={result.aic:.2f}  BIC={result.bic:.2f}")
    print(result.summary().tables[1])

    # 5. 逐步预测（滚动窗口）
    print("5. 滚动预测测试集...")
    history  = list(train)
    preds    = []
    for t in range(len(test)):
        m = ARIMA(history, order=(1, 1, 1))
        r = m.fit()
        fc = r.forecast(steps=1)[0]
        preds.append(fc)
        history.append(test[t])

    preds = np.array(preds)
    mse  = mean_squared_error(test, preds)
    rmse = np.sqrt(mse)
    mae  = mean_absolute_error(test, preds)
    print(f"   RMSE={rmse:.4f}  MAE={mae:.4f}")

    # 6. 对比不同 ARIMA 阶数
    print("6. 比较不同 ARIMA(p,d,q) 阶数...")
    orders = [(1,1,0), (1,1,1), (2,1,1), (1,1,2), (2,1,2)]
    order_results = {}
    for order in orders:
        try:
            m = ARIMA(train, order=order).fit()
            fc_list = []
            hist = list(train)
            for obs in test:
                r = ARIMA(hist, order=order).fit()
                fc_list.append(r.forecast(1)[0])
                hist.append(obs)
            rmse_o = np.sqrt(mean_squared_error(test, fc_list))
            order_results[str(order)] = rmse_o
            print(f"   ARIMA{order}  RMSE={rmse_o:.4f}")
        except Exception as e:
            print(f"   ARIMA{order}  跳过: {e}")

    # 7. 可视化
    print("7. 可视化结果...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    # ── 原始序列 + 一阶差分 ──
    ax = axes[0, 0]
    ax.plot(series, color='steelblue', linewidth=1.2, label='原始序列')
    ax.axvline(x=train_size, color='red', linestyle='--', label='Train/Test split')
    ax.set_title('ARIMA 原始时间序列')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── ACF & PACF（一阶差分后）──
    ax = axes[0, 1]
    lags_show = min(30, len(diff1) // 4)
    acf_vals  = acf(diff1, nlags=lags_show)
    pacf_vals = pacf(diff1, nlags=lags_show)
    ax.stem(range(len(acf_vals)), acf_vals, linefmt='steelblue',
            markerfmt='D', basefmt='k-', label='ACF')
    ax.stem(range(len(pacf_vals)), pacf_vals, linefmt='darkorange',
            markerfmt='s', basefmt='k-', label='PACF')
    ax.axhline(y=1.96/np.sqrt(len(diff1)), color='gray', linestyle='--', alpha=0.7)
    ax.axhline(y=-1.96/np.sqrt(len(diff1)), color='gray', linestyle='--', alpha=0.7)
    ax.set_title('一阶差分序列 ACF & PACF')
    ax.set_xlabel('Lag')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 预测 vs 真实值 ──
    ax = axes[1, 0]
    test_idx = np.arange(train_size, len(series))
    ax.plot(np.arange(train_size), train, color='steelblue',
            linewidth=1.0, label='Train')
    ax.plot(test_idx, test, color='darkgreen',
            linewidth=1.5, label='真实值')
    ax.plot(test_idx, preds, color='tomato', linewidth=1.5,
            linestyle='--', label=f'预测值 (RMSE={rmse:.3f})')
    ax.fill_between(test_idx,
                    preds - 1.96 * rmse, preds + 1.96 * rmse,
                    alpha=0.15, color='tomato', label='95% CI')
    ax.axvline(x=train_size, color='gray', linestyle=':', linewidth=1)
    ax.set_title('ARIMA(1,1,1) 滚动预测结果')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 不同阶数 RMSE 对比 ──
    ax = axes[1, 1]
    names_ord = list(order_results.keys())
    rmse_vals = list(order_results.values())
    colors_bar = ['tomato' if n == '(1, 1, 1)' else 'steelblue'
                  for n in names_ord]
    bars = ax.bar(names_ord, rmse_vals, color=colors_bar, edgecolor='white')
    for bar, v in zip(bars, rmse_vals):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.001,
                f'{v:.4f}', ha='center', va='bottom', fontsize=9)
    ax.set_title('不同 ARIMA 阶数 RMSE 对比')
    ax.set_xlabel('ARIMA Order')
    ax.set_ylabel('RMSE')
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')

    save_path = get_results_path('arima_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    arima()
