# LSTM 时间序列预测模型
# 利用长短期记忆网络捕捉序列长期依赖关系

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from utils import get_results_path, save_and_close


def generate_ts_data(n=500, seed=42):
    """生成含趋势 + 周期 + 噪声的合成时间序列"""
    np.random.seed(seed)
    t = np.arange(n)
    trend    = 0.02 * t
    seasonal = 2.0 * np.sin(2 * np.pi * t / 50)
    noise    = np.random.normal(0, 0.4, n)
    return trend + seasonal + noise


def create_sequences(data, look_back=30):
    """将一维序列切分成 (样本, look_back, 1) 形式的监督数据"""
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


def lstm():
    """LSTM 时间序列预测实现"""
    print("LSTM 时间序列预测模型运行中...\n")

    # 1. 数据准备
    print("1. 生成时间序列数据...")
    LOOK_BACK = 30
    series = generate_ts_data(n=600)
    print(f"   序列长度: {len(series)}")

    # 归一化
    scaler = MinMaxScaler(feature_range=(0, 1))
    series_scaled = scaler.fit_transform(series.reshape(-1, 1)).ravel()

    X, y = create_sequences(series_scaled, look_back=LOOK_BACK)
    X = X[:, :, np.newaxis]   # (samples, timesteps, 1)

    # 训练/测试划分
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    print(f"   训练集: {X_train.shape}  测试集: {X_test.shape}")

    # 2. 对比三种 LSTM 架构
    print("2. 训练并对比三种 LSTM 架构...")

    def make_vanilla_lstm():
        m = Sequential([
            LSTM(64, input_shape=(LOOK_BACK, 1), return_sequences=False),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    def make_stacked_lstm():
        m = Sequential([
            LSTM(64, input_shape=(LOOK_BACK, 1), return_sequences=True),
            Dropout(0.2),
            LSTM(32, return_sequences=False),
            Dropout(0.2),
            Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    def make_bilstm():
        m = Sequential([
            Bidirectional(LSTM(64, return_sequences=False),
                          input_shape=(LOOK_BACK, 1)),
            Dropout(0.2),
            Dense(32, activation='relu'),
            Dense(1)
        ])
        m.compile(optimizer='adam', loss='mse')
        return m

    architectures = {
        'Vanilla LSTM':  make_vanilla_lstm(),
        'Stacked LSTM':  make_stacked_lstm(),
        'Bi-LSTM':       make_bilstm(),
    }

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]

    histories, preds_dict = {}, {}
    for name, model in architectures.items():
        hist = model.fit(X_train, y_train,
                         epochs=80, batch_size=32,
                         validation_split=0.15,
                         callbacks=callbacks, verbose=0)
        pred_scaled = model.predict(X_test, verbose=0).ravel()
        pred = scaler.inverse_transform(pred_scaled.reshape(-1, 1)).ravel()
        true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()
        rmse = np.sqrt(mean_squared_error(true, pred))
        mae  = mean_absolute_error(true, pred)
        print(f"   {name:15s}  RMSE={rmse:.4f}  MAE={mae:.4f}  "
              f"epochs={len(hist.history['loss'])}")
        histories[name] = hist
        preds_dict[name] = (pred, rmse, mae)

    true_vals = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

    # 3. 多步预测（Vanilla LSTM，以最佳模型递归预测）
    print("3. 多步预测（递归自回归）...")
    best_name = min(preds_dict, key=lambda k: preds_dict[k][1])
    # 重建最佳模型并训练
    model_best = make_vanilla_lstm()
    model_best.fit(X_train, y_train, epochs=80, batch_size=32,
                   validation_split=0.15, callbacks=callbacks, verbose=0)

    n_forecast = 30
    seed_seq = series_scaled[-(LOOK_BACK + n_forecast):-n_forecast]
    buf = list(seed_seq)
    multistep_preds_sc = []
    for _ in range(n_forecast):
        inp = np.array(buf[-LOOK_BACK:], dtype=np.float32)[np.newaxis, :, np.newaxis]
        out = model_best.predict(inp, verbose=0)[0, 0]
        multistep_preds_sc.append(out)
        buf.append(out)
    multistep_preds = scaler.inverse_transform(
        np.array(multistep_preds_sc).reshape(-1, 1)).ravel()
    multistep_true  = series[-(n_forecast):]

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    colors = {'Vanilla LSTM': 'tomato', 'Stacked LSTM': 'darkorange', 'Bi-LSTM': 'seagreen'}

    # ── 子图1：预测 vs 真实（Vanilla）──
    ax = axes[0]
    ax.plot(true_vals, color='steelblue', linewidth=1.2, label='True')
    for name, (pred, rmse, mae) in preds_dict.items():
        ax.plot(pred, color=colors[name], linewidth=1.2,
                linestyle='--', label=f'{name} (RMSE={rmse:.3f})', alpha=0.85)
    ax.set_title('LSTM 单步预测对比')
    ax.set_xlabel('Time Step (Test)')
    ax.set_ylabel('Value')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── 子图2：训练损失曲线 ──
    ax = axes[1]
    for name, hist in histories.items():
        ax.plot(hist.history['val_loss'], color=colors[name],
                label=name, linewidth=1.5)
    ax.set_title('验证集损失曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss')
    ax.legend()
    ax.set_yscale('log')
    ax.grid(True, alpha=0.3)

    # ── 子图3：多步预测 ──
    ax = axes[2]
    n_context = 50
    context = series[-(n_context + n_forecast):-n_forecast]
    ax.plot(np.arange(n_context), context, color='steelblue',
            linewidth=1.5, label='历史序列')
    ax.plot(np.arange(n_context, n_context + n_forecast),
            multistep_true, color='darkgreen', linewidth=1.5, label='真实值')
    ax.plot(np.arange(n_context, n_context + n_forecast),
            multistep_preds, color='tomato', linewidth=1.5,
            linestyle='--', label=f'多步预测 ({n_forecast} 步)')
    ax.axvline(x=n_context, color='gray', linestyle=':', linewidth=1)
    ax.set_title(f'LSTM {n_forecast}步多步预测')
    ax.set_xlabel('Time')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('lstm_timeseries_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    lstm()
