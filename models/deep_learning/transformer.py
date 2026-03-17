# Transformer 模型（从零实现核心组件）
# 多头自注意力机制 + 位置编码，用于序列分类任务

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Layer, Dense, Dropout, LayerNormalization,
                                      MultiHeadAttention, Embedding,
                                      GlobalAveragePooling1D, Input)
from tensorflow.keras.callbacks import EarlyStopping
from utils import get_results_path, save_and_close


# ── 位置编码 ───────────────────────────────────────────────────────────
class PositionalEncoding(Layer):
    """正弦/余弦位置编码层"""
    def __init__(self, max_len=100, d_model=64, **kwargs):
        super().__init__(**kwargs)
        # 预计算位置编码矩阵
        pos = np.arange(max_len)[:, np.newaxis]        # (max_len, 1)
        dim = np.arange(d_model)[np.newaxis, :]         # (1, d_model)
        angle = pos / np.power(10000, (2 * (dim // 2)) / d_model)
        pe = np.zeros_like(angle)
        pe[:, 0::2] = np.sin(angle[:, 0::2])
        pe[:, 1::2] = np.cos(angle[:, 1::2])
        self.pe = tf.cast(pe[np.newaxis, :, :], dtype=tf.float32)  # (1, L, d)

    def call(self, x):
        seq_len = tf.shape(x)[1]
        return x + self.pe[:, :seq_len, :]


# ── Transformer 编码器块 ───────────────────────────────────────────────
class TransformerBlock(Layer):
    """单层 Transformer 编码器：多头注意力 + FFN + 残差 + LayerNorm"""
    def __init__(self, d_model, n_heads, ffn_dim, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.attn = MultiHeadAttention(num_heads=n_heads, key_dim=d_model // n_heads)
        self.ffn1 = Dense(ffn_dim, activation='relu')
        self.ffn2 = Dense(d_model)
        self.norm1 = LayerNormalization(epsilon=1e-6)
        self.norm2 = LayerNormalization(epsilon=1e-6)
        self.drop1 = Dropout(dropout_rate)
        self.drop2 = Dropout(dropout_rate)

    def call(self, x, training=False):
        # 多头自注意力 + 残差
        attn_out = self.attn(x, x, training=training)
        attn_out = self.drop1(attn_out, training=training)
        out1 = self.norm1(x + attn_out)
        # 前馈网络 + 残差
        ffn_out = self.ffn2(self.ffn1(out1))
        ffn_out = self.drop2(ffn_out, training=training)
        out2 = self.norm2(out1 + ffn_out)
        return out2


def build_transformer(seq_len, d_model=64, n_heads=4,
                       ffn_dim=128, n_layers=2, n_classes=3,
                       dropout=0.1):
    """构建 Transformer 分类模型"""
    inputs = Input(shape=(seq_len, 1))

    # 投影到 d_model 维
    x = Dense(d_model)(inputs)
    x = PositionalEncoding(max_len=seq_len, d_model=d_model)(x)
    x = Dropout(dropout)(x)

    # 堆叠 Transformer 块
    for _ in range(n_layers):
        x = TransformerBlock(d_model, n_heads, ffn_dim, dropout)(x)

    # 全局池化 + 分类头
    x = GlobalAveragePooling1D()(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(n_classes, activation='softmax')(x)

    return Model(inputs, outputs, name='Transformer_Classifier')


def generate_sequence_data(n_samples=1200, seq_len=60, n_classes=3, seed=42):
    """生成三类频率不同的正弦波序列"""
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, seq_len)
    freqs = [1.0, 4.0, 10.0]
    X, y = [], []
    per = n_samples // n_classes
    for cls, f in enumerate(freqs):
        for _ in range(per):
            phase = rng.uniform(0, 2 * np.pi)
            amp   = rng.uniform(0.8, 1.2)
            noise = rng.normal(0, 0.2, seq_len)
            seq   = amp * np.sin(2 * np.pi * f * t + phase) + noise
            X.append(seq.astype(np.float32))
            y.append(cls)
    X, y = np.array(X), np.array(y)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def transformer():
    """Transformer 自注意力序列分类实现"""
    print("Transformer 模型运行中...\n")

    # 1. 数据准备
    print("1. 准备序列数据...")
    SEQ_LEN   = 60
    N_CLASSES = 3
    X, y = generate_sequence_data(n_samples=1200, seq_len=SEQ_LEN,
                                   n_classes=N_CLASSES)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split, :, np.newaxis], X[split:, :, np.newaxis]
    y_train, y_test = y[:split], y[split:]
    y_train_oh = tf.keras.utils.to_categorical(y_train, N_CLASSES)
    y_test_oh  = tf.keras.utils.to_categorical(y_test,  N_CLASSES)
    print(f"   训练集: {X_train.shape}  测试集: {X_test.shape}")

    # 2. 构建 Transformer
    print("2. 构建 Transformer 模型（2层, 4头注意力）...")
    model = build_transformer(SEQ_LEN, d_model=64, n_heads=4,
                               ffn_dim=128, n_layers=2, n_classes=N_CLASSES)
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    model.summary()

    # 3. 训练
    print("3. 训练模型...")
    cb = EarlyStopping(monitor='val_loss', patience=15,
                       restore_best_weights=True, verbose=0)
    history = model.fit(
        X_train, y_train_oh,
        epochs=80, batch_size=32,
        validation_split=0.15,
        callbacks=[cb], verbose=0
    )
    print(f"   实际训练轮数: {len(history.history['loss'])}")

    # 4. 评估
    print("4. 模型评估...")
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    acc = accuracy_score(y_test, y_pred)
    cm  = confusion_matrix(y_test, y_pred)
    print(f"   测试准确率: {acc:.4f}")

    # 5. 可视化注意力权重（取第一层第一个头）
    print("5. 提取并可视化注意力权重...")

    # 构建注意力提取子模型
    transformer_block = [l for l in model.layers
                          if isinstance(l, TransformerBlock)][0]
    attn_layer = transformer_block.attn

    # 取前 5 个测试样本求注意力
    sample_input = X_test[:5]
    # 手动前向传播到第一个 TransformerBlock 输入
    inp = model.input
    x = model.get_layer(index=1)(inp)     # Dense 投影
    x = model.get_layer(index=2)(x)       # PositionalEncoding
    x = model.get_layer(index=3)(x)       # Dropout

    sub_model = Model(inputs=inp, outputs=x)
    x_before_attn = sub_model.predict(sample_input, verbose=0)

    # 计算注意力矩阵（不训练）
    _, attn_weights = attn_layer(
        tf.constant(x_before_attn),
        tf.constant(x_before_attn),
        return_attention_scores=True
    )
    # attn_weights: (batch, heads, seq, seq) -> 取第一个样本第一个头
    attn_map = attn_weights[0, 0].numpy()   # (seq, seq)

    # ── 可视化 ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：训练曲线 ──
    ax = axes[0]
    ax.plot(history.history['accuracy'], label='Train Acc', color='steelblue')
    ax.plot(history.history['val_accuracy'], label='Val Acc',
            color='darkorange', linestyle='--')
    ax.set_title('Transformer 训练准确率曲线')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # ── 子图2：混淆矩阵 ──
    ax = axes[1]
    im = ax.imshow(cm, interpolation='nearest', cmap='Blues')
    fig.colorbar(im, ax=ax)
    class_names = ['低频', '中频', '高频']
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                    color='white' if cm[i, j] > cm.max()/2 else 'black',
                    fontsize=12)
    ax.set_title(f'混淆矩阵 (Acc={acc:.3f})')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')

    # ── 子图3：自注意力热力图 ──
    ax = axes[2]
    im2 = ax.imshow(attn_map, cmap='hot', aspect='auto')
    fig.colorbar(im2, ax=ax)
    ax.set_title('自注意力权重图\n(Layer-1, Head-1, Sample-0)')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')

    save_path = get_results_path('transformer_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")

    # ── 位置编码可视化 ──
    d_model = 64
    max_len = SEQ_LEN
    pos = np.arange(max_len)[:, np.newaxis]
    dim = np.arange(d_model)[np.newaxis, :]
    angle = pos / np.power(10000, (2 * (dim // 2)) / d_model)
    pe = np.zeros_like(angle)
    pe[:, 0::2] = np.sin(angle[:, 0::2])
    pe[:, 1::2] = np.cos(angle[:, 1::2])

    fig2, ax2 = plt.subplots(figsize=(12, 4))
    im3 = ax2.imshow(pe.T, aspect='auto', cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(im3, ax=ax2)
    ax2.set_title('Transformer 位置编码矩阵（正弦/余弦）')
    ax2.set_xlabel('Position')
    ax2.set_ylabel('Encoding Dimension')
    save_path2 = get_results_path('transformer_positional_encoding.png')
    save_and_close(save_path2)
    print(f"   位置编码图已保存: {save_path2}")


if __name__ == "__main__":
    transformer()
