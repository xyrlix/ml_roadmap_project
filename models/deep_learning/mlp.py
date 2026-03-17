# MLP 多层感知机模型
# 全连接神经网络，使用 TensorFlow/Keras 实现

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, plot_training_history, save_and_close)


def mlp():
    """多层感知机（MLP）实现"""
    print("MLP 多层感知机模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=1000, n_features=20,
                                         n_informative=10, n_redundant=5,
                                         n_classes=3, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 特征标准化（MLP 对尺度敏感）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    n_classes = len(np.unique(y))
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}")
    print(f"   特征数: {X.shape[1]}  类别数: {n_classes}")

    # one-hot 编码
    y_train_oh = tf.keras.utils.to_categorical(y_train, n_classes)
    y_test_oh  = tf.keras.utils.to_categorical(y_test, n_classes)

    # 2. 构建 MLP 模型
    print("2. 构建 MLP 模型...")
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    # 3. 训练
    print("3. 训练模型（含早停 & 学习率衰减）...")
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=15,
                      restore_best_weights=True, verbose=0),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                          patience=7, min_lr=1e-6, verbose=0)
    ]
    history = model.fit(
        X_train, y_train_oh,
        epochs=100, batch_size=32,
        validation_split=0.15,
        callbacks=callbacks,
        verbose=0
    )
    print(f"   实际训练轮数: {len(history.history['loss'])}")

    # 4. 评估
    print("4. 模型评估...")
    test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    accuracy, cm = print_classification_report(y_test, y_pred, "MLP")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 准确率曲线 ──
    ax = axes[0]
    ax.plot(history.history['accuracy'], label='Train Acc', color='steelblue')
    ax.plot(history.history['val_accuracy'], label='Val Acc', color='darkorange')
    ax.set_title('MLP 准确率曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # ── 损失曲线 ──
    ax = axes[1]
    ax.plot(history.history['loss'], label='Train Loss', color='steelblue')
    ax.plot(history.history['val_loss'], label='Val Loss', color='darkorange')
    ax.set_title('MLP 损失曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # ── 混淆矩阵热力图 ──
    ax = axes[2]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    ax.set_title('混淆矩阵')
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    tick_marks = np.arange(n_classes)
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max() / 2 else 'black',
                    fontsize=11)

    save_path = get_results_path('mlp_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    mlp()
