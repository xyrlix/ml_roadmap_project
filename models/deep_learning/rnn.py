# RNN 循环神经网络模型
# 使用 TensorFlow/Keras 实现 SimpleRNN 与 LSTM 对比，展示序列学习能力

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (SimpleRNN, LSTM, GRU, Dense,
                                      Dropout, Embedding, GlobalAveragePooling1D)
from tensorflow.keras.callbacks import EarlyStopping
from utils import get_results_path, save_and_close


# ── 生成正弦波时序分类数据 ──────────────────────────────────────────────
def generate_sequence_data(n_samples=1000, seq_len=50, n_classes=3, random_state=42):
    """
    生成三类正弦序列数据:
      Class 0: 低频正弦  (f=0.1)
      Class 1: 高频正弦  (f=0.5)
      Class 2: 叠加正弦  (f=0.1 + 0.4)
    """
    np.random.seed(random_state)
    X, y = [], []
    per_class = n_samples // n_classes
    t = np.linspace(0, 1, seq_len)

    for cls in range(n_classes):
        for _ in range(per_class):
            noise = np.random.normal(0, 0.15, seq_len)
            if cls == 0:
                seq = np.sin(2 * np.pi * 1.5 * t) + noise
            elif cls == 1:
                seq = np.sin(2 * np.pi * 5.0 * t) + noise
            else:
                seq = np.sin(2 * np.pi * 1.5 * t) + np.sin(2 * np.pi * 4.0 * t) * 0.5 + noise
            X.append(seq)
            y.append(cls)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def rnn():
    """RNN / LSTM / GRU 序列分类对比实现"""
    print("RNN 循环神经网络模型运行中...\n")

    # 1. 数据准备
    print("1. 准备正弦序列数据...")
    SEQ_LEN   = 50
    N_CLASSES = 3
    X, y = generate_sequence_data(n_samples=1500, seq_len=SEQ_LEN,
                                   n_classes=N_CLASSES)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # reshape -> (samples, timesteps, features)
    X_train_3d = X_train[:, :, np.newaxis]
    X_test_3d  = X_test[:, :, np.newaxis]
    y_train_oh = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  N_CLASSES)
    print(f"   训练集: {X_train_3d.shape}  测试集: {X_test_3d.shape}")

    # 2. 构建并训练三种模型
    print("2. 训练 SimpleRNN / LSTM / GRU 模型...")
    architectures = {
        'SimpleRNN': Sequential([
            SimpleRNN(64, input_shape=(SEQ_LEN, 1), return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(N_CLASSES, activation='softmax')
        ]),
        'LSTM': Sequential([
            LSTM(64, input_shape=(SEQ_LEN, 1), return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(N_CLASSES, activation='softmax')
        ]),
        'GRU': Sequential([
            GRU(64, input_shape=(SEQ_LEN, 1), return_sequences=False),
            Dropout(0.3),
            Dense(32, activation='relu'),
            Dense(N_CLASSES, activation='softmax')
        ]),
    }

    histories, results = {}, {}
    for name, model in architectures.items():
        model.compile(optimizer='adam',
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])
        cb = EarlyStopping(monitor='val_loss', patience=10,
                           restore_best_weights=True, verbose=0)
        hist = model.fit(X_train_3d, y_train_oh,
                         epochs=60, batch_size=64,
                         validation_split=0.15,
                         callbacks=[cb], verbose=0)
        y_pred = np.argmax(model.predict(X_test_3d, verbose=0), axis=1)
        acc = accuracy_score(y_test, y_pred)
        print(f"   {name:10s} 准确率: {acc:.4f}  "
              f"(训练轮次={len(hist.history['loss'])})")
        histories[name] = hist
        results[name] = {'acc': acc, 'pred': y_pred}

    # 3. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    colors = {'SimpleRNN': 'steelblue', 'LSTM': 'darkorange', 'GRU': 'seagreen'}

    # ── 子图1：训练准确率曲线对比 ──
    ax = axes[0]
    for name, hist in histories.items():
        ax.plot(hist.history['val_accuracy'], color=colors[name],
                label=f'{name} val', linewidth=1.8)
    ax.set_title('验证准确率曲线对比')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # ── 子图2：测试准确率柱状图 ──
    ax = axes[1]
    names = list(results.keys())
    accs  = [results[n]['acc'] for n in names]
    bars = ax.bar(names, accs,
                  color=[colors[n] for n in names],
                  edgecolor='white', width=0.5)
    for bar, acc in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{acc:.3f}', ha='center', va='bottom', fontsize=11)
    ax.set_title('三种架构测试准确率')
    ax.set_ylabel('Accuracy')
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图3：LSTM 混淆矩阵 ──
    ax = axes[2]
    cm = confusion_matrix(y_test, results['LSTM']['pred'])
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_title('LSTM 混淆矩阵')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    labels = ['低频', '高频', '叠加']
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=11)

    save_path = get_results_path('rnn_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")

    # ── 额外图：样本序列可视化 ──
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    t_axis = np.linspace(0, 1, SEQ_LEN)
    class_names = ['低频正弦 (Class 0)', '高频正弦 (Class 1)', '叠加正弦 (Class 2)']
    for cls in range(N_CLASSES):
        sample = X[y == cls][0]
        axes2[cls].plot(t_axis, sample, color=list(colors.values())[cls], linewidth=1.8)
        axes2[cls].set_title(class_names[cls])
        axes2[cls].set_xlabel('Time')
        axes2[cls].set_ylabel('Amplitude')
        axes2[cls].grid(True, alpha=0.3)
    plt.suptitle('输入序列样本（三类）', fontsize=13)
    save_path2 = get_results_path('rnn_sequences.png')
    save_and_close(save_path2)
    print(f"   序列样本图已保存: {save_path2}")


if __name__ == "__main__":
    rnn()
