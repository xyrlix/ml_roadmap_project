# CNN 卷积神经网络模型（深度学习版）
# 使用 TensorFlow/Keras 在合成图像数据上演示卷积层特征提取

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Flatten,
                                      Dense, Dropout, BatchNormalization,
                                      GlobalAveragePooling2D, Input)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from utils import get_results_path, save_and_close


def generate_shape_images(n_per_class=300, img_size=28, random_state=42):
    """
    生成三类几何形状图像（灰度图）：
      Class 0: 圆形
      Class 1: 矩形
      Class 2: 三角形（填充线段近似）
    """
    rng = np.random.default_rng(random_state)
    X, y = [], []
    for cls in range(3):
        for _ in range(n_per_class):
            img = np.zeros((img_size, img_size), dtype=np.float32)
            if cls == 0:  # 圆形
                cx, cy = rng.integers(8, 20, size=2)
                r = rng.integers(4, 9)
                for row in range(img_size):
                    for col in range(img_size):
                        if (row - cy)**2 + (col - cx)**2 <= r**2:
                            img[row, col] = 1.0
            elif cls == 1:  # 矩形
                r0, r1 = sorted(rng.integers(4, 24, size=2))
                c0, c1 = sorted(rng.integers(4, 24, size=2))
                img[r0:r1+1, c0:c1+1] = 1.0
            else:  # 三角形（填充）
                top = rng.integers(3, 10)
                bot = rng.integers(18, 26)
                cx_t = rng.integers(8, 20)
                for row in range(top, bot):
                    half_w = int((row - top) / (bot - top) * 8) + 1
                    c0 = max(0, cx_t - half_w)
                    c1 = min(img_size - 1, cx_t + half_w)
                    img[row, c0:c1+1] = 1.0
            # 添加噪声
            noise = rng.normal(0, 0.1, img.shape)
            img = np.clip(img + noise, 0, 1)
            X.append(img)
            y.append(cls)

    X = np.array(X, dtype=np.float32)[..., np.newaxis]   # (N, 28, 28, 1)
    y = np.array(y, dtype=np.int32)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_cnn(input_shape=(28, 28, 1), n_classes=3):
    """构建 LeNet 风格的 CNN"""
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same',
               input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        Dropout(0.4),
        Dense(n_classes, activation='softmax')
    ])
    return model


def cnn():
    """CNN 图像分类演示（三类几何形状）"""
    print("CNN 卷积神经网络模型运行中...\n")

    # 1. 数据准备
    print("1. 生成几何形状图像数据集...")
    N_CLASSES = 3
    X, y = generate_shape_images(n_per_class=400)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    y_train_oh = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  N_CLASSES)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}")
    print(f"   图像尺寸: {X.shape[1:3]}  通道: {X.shape[3]}")

    # 2. 构建模型
    print("2. 构建 CNN 模型...")
    model = build_cnn(input_shape=X.shape[1:], n_classes=N_CLASSES)
    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 3. 训练
    print("3. 训练模型...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]
    history = model.fit(
        X_train, y_train_oh,
        epochs=80, batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )
    print(f"   实际训练轮数: {len(history.history['loss'])}")

    # 4. 评估
    print("4. 模型评估...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    print(f"   测试准确率: {acc:.4f} ({acc*100:.2f}%)")
    print(f"   混淆矩阵:\n{cm}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    class_names = ['圆形', '矩形', '三角形']

    # ── 子图1：准确率 & 损失曲线 ──
    ax = axes[0]
    ax_r = ax.twinx()
    ax.plot(history.history['accuracy'], label='Train Acc', color='steelblue')
    ax.plot(history.history['val_accuracy'], label='Val Acc',
            color='steelblue', linestyle='--')
    ax_r.plot(history.history['loss'], label='Train Loss', color='darkorange')
    ax_r.plot(history.history['val_loss'], label='Val Loss',
              color='darkorange', linestyle='--')
    ax.set_title('训练曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy', color='steelblue')
    ax_r.set_ylabel('Loss', color='darkorange')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc='center right', fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图2：混淆矩阵 ──
    ax = axes[1]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=12)
    ax.set_title('混淆矩阵')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # ── 子图3：样本图像展示 ──
    ax = axes[2]
    ax.axis('off')
    n_show = 9
    inner_rows, inner_cols = 3, 3
    indices = []
    for cls in range(N_CLASSES):
        idxs = np.where(y_test == cls)[0][:3]
        indices.extend(idxs.tolist())

    grid = np.ones((3 * 28 + 2 * 2, 3 * 28 + 2 * 2), dtype=np.float32)
    for k, idx in enumerate(indices[:9]):
        r, c = divmod(k, 3)
        rr = r * (28 + 2)
        cc = c * (28 + 2)
        grid[rr:rr+28, cc:cc+28] = X_test[idx, :, :, 0]

    ax.imshow(grid, cmap='gray', vmin=0, vmax=1)
    ax.set_title(f'测试样本示例\n(准确率={acc:.3f})')
    ax.axis('off')

    save_path = get_results_path('deep_cnn_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    cnn()
