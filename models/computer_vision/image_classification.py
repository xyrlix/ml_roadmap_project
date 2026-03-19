"""
图像分类 (Image Classification)
=================================
基于手工实现卷积神经网络（CNN）的图像分类，纯 NumPy 实现（无需 PyTorch / TensorFlow）

架构：
  Conv(8 filters, 3×3) → ReLU → MaxPool(2×2)
  → Conv(16 filters, 3×3) → ReLU → MaxPool(2×2)
  → Flatten → FC(128) → ReLU → Dropout
  → FC(n_classes) → Softmax

数据：合成多类图像（每类有不同的几何特征：横纹/竖纹/斜纹/网格/圆形…）
评估：准确率、混淆矩阵、各类 Precision/Recall/F1
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

# ─────────────────────── 合成数据集 ───────────────────────────────

def make_synthetic_dataset(n_per_class: int = 120, img_size: int = 28,
                            n_classes: int = 5, seed: int = 42) -> tuple:
    """
    生成 n_classes 类合成图像，每类有不同的纹理/形状特征：
      0 - 水平条纹    1 - 垂直条纹    2 - 斜纹 (/斜)
      3 - 网格纹理    4 - 中心圆
    """
    rng  = np.random.default_rng(seed)
    imgs, labels = [], []

    patterns = {
        0: lambda s: _horizontal_stripes(s, rng),
        1: lambda s: _vertical_stripes(s, rng),
        2: lambda s: _diagonal_stripes(s, rng),
        3: lambda s: _grid_texture(s, rng),
        4: lambda s: _circle_pattern(s, rng),
    }

    for cls in range(n_classes):
        for _ in range(n_per_class):
            img = patterns[cls](img_size)
            imgs.append(img)
            labels.append(cls)

    X = np.array(imgs, dtype=np.float32)   # (N, H, W)
    y = np.array(labels, dtype=np.int32)
    idx = rng.permutation(len(y))
    return X[idx], y[idx]


def _horizontal_stripes(s, rng, freq=3):
    img = np.zeros((s, s), dtype=np.float32)
    for r in range(s):
        if (r // (s // (freq * 2))) % 2 == 0:
            img[r, :] = 1.0
    return img + rng.normal(0, 0.05, (s, s)).astype(np.float32)


def _vertical_stripes(s, rng, freq=3):
    img = np.zeros((s, s), dtype=np.float32)
    for c in range(s):
        if (c // (s // (freq * 2))) % 2 == 0:
            img[:, c] = 1.0
    return img + rng.normal(0, 0.05, (s, s)).astype(np.float32)


def _diagonal_stripes(s, rng, freq=4):
    img = np.zeros((s, s), dtype=np.float32)
    for r in range(s):
        for c in range(s):
            if ((r + c) // (s // (freq * 2))) % 2 == 0:
                img[r, c] = 1.0
    return img + rng.normal(0, 0.05, (s, s)).astype(np.float32)


def _grid_texture(s, rng, step=4):
    img = np.zeros((s, s), dtype=np.float32)
    for r in range(s):
        for c in range(s):
            if r % step == 0 or c % step == 0:
                img[r, c] = 1.0
    return img + rng.normal(0, 0.05, (s, s)).astype(np.float32)


def _circle_pattern(s, rng):
    img = np.zeros((s, s), dtype=np.float32)
    cx, cy = s // 2, s // 2
    r = s // 4
    for ri in range(s):
        for ci in range(s):
            d = np.sqrt((ri - cx) ** 2 + (ci - cy) ** 2)
            if r - 2 <= d <= r + 2:
                img[ri, ci] = 1.0
    return img + rng.normal(0, 0.05, (s, s)).astype(np.float32)


def train_val_split(X, y, val_ratio=0.2, seed=42):
    rng = np.random.default_rng(seed)
    n   = len(y)
    idx = rng.permutation(n)
    cut = int(n * (1 - val_ratio))
    return X[idx[:cut]], y[idx[:cut]], X[idx[cut:]], y[idx[cut:]]


# ─────────────────────── CNN 组件 ────────────────────────────────

def relu(x):
    return np.maximum(0, x)


def softmax(x):
    x = x - x.max(axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-10)


def conv2d_forward(X, W, b):
    """
    X : (N, H, W, C_in)
    W : (fH, fW, C_in, C_out)
    b : (C_out,)
    返回: (N, H-fH+1, W-fW+1, C_out)
    """
    N, H, Ww, C_in = X.shape
    fH, fW, _, C_out = W.shape
    oH = H - fH + 1
    oW = Ww - fW + 1
    out = np.zeros((N, oH, oW, C_out), dtype=np.float32)
    for i in range(fH):
        for j in range(fW):
            # (N, oH, oW, C_in) x (C_in, C_out) → (N, oH, oW, C_out)
            out += X[:, i:i+oH, j:j+oW, :] @ W[i, j]
    out += b
    return out


def maxpool2d_forward(X, pool=2):
    """
    X   : (N, H, W, C)
    返回: (N, H//pool, W//pool, C)
    """
    N, H, W, C = X.shape
    oH, oW = H // pool, W // pool
    out = np.zeros((N, oH, oW, C), dtype=np.float32)
    for i in range(oH):
        for j in range(oW):
            out[:, i, j, :] = X[:, i*pool:(i+1)*pool,
                                    j*pool:(j+1)*pool, :].max(axis=(1, 2))
    return out


# ─────────────────────── 简单 CNN 模型 ──────────────────────────

class SimpleCNN:
    """
    两层卷积 + 两层全连接，使用 forward-only 推理
    （为简洁起见，训练使用手工梯度；卷积层冻结随机初始化，仅训练 FC 层）
    """

    def __init__(self, img_size=28, n_classes=5, seed=42):
        rng = np.random.default_rng(seed)
        # 卷积核（固定随机初始化作为特征提取器）
        self.W1 = rng.normal(0, 0.1, (3, 3, 1, 8)).astype(np.float32)
        self.b1 = np.zeros(8, dtype=np.float32)
        self.W2 = rng.normal(0, 0.1, (3, 3, 8, 16)).astype(np.float32)
        self.b2 = np.zeros(16, dtype=np.float32)

        # 计算 flatten 维度
        h1 = (img_size - 2) // 2          # after conv1 + pool
        h2 = (h1 - 2) // 2                # after conv2 + pool
        flat_dim = h2 * h2 * 16

        # 全连接层（可训练）
        self.W3 = rng.normal(0, np.sqrt(2 / flat_dim), (flat_dim, 128)).astype(np.float32)
        self.b3 = np.zeros(128, dtype=np.float32)
        self.W4 = rng.normal(0, np.sqrt(2 / 128), (128, n_classes)).astype(np.float32)
        self.b4 = np.zeros(n_classes, dtype=np.float32)

        self.flat_dim  = flat_dim
        self.n_classes = n_classes
        self.train_losses, self.val_accs = [], []

    def _forward(self, X_batch: np.ndarray, training: bool = False):
        """X_batch: (N, H, W)，归一化到 [0,1]"""
        X = X_batch[:, :, :, np.newaxis]        # → (N, H, W, 1)
        # Conv1 → ReLU → MaxPool
        z1 = conv2d_forward(X, self.W1, self.b1)
        a1 = relu(z1)
        p1 = maxpool2d_forward(a1, pool=2)
        # Conv2 → ReLU → MaxPool
        z2 = conv2d_forward(p1, self.W2, self.b2)
        a2 = relu(z2)
        p2 = maxpool2d_forward(a2, pool=2)
        # Flatten
        flat = p2.reshape(len(X), -1)           # (N, flat_dim)
        # FC1 → ReLU
        h1 = relu(flat @ self.W3 + self.b3)
        # Dropout（训练时 p=0.3）
        if training:
            mask = (np.random.rand(*h1.shape) > 0.3).astype(np.float32)
            h1   = h1 * mask / 0.7
        # FC2 → Softmax
        logits = h1 @ self.W4 + self.b4
        probs  = softmax(logits)
        return probs, h1, flat, logits

    def fit(self, X_train, y_train, X_val, y_val,
            n_epochs=25, lr=0.01, batch_size=32):
        N   = len(X_train)
        rng = np.random.default_rng(0)

        for epoch in range(n_epochs):
            idx   = rng.permutation(N)
            epoch_loss = 0.0

            for start in range(0, N, batch_size):
                bi = idx[start: start + batch_size]
                Xb = np.clip(X_train[bi], 0, 1)
                yb = y_train[bi]

                probs, h1, flat, _ = self._forward(Xb, training=True)

                # Cross-entropy loss
                log_p = np.log(probs[np.arange(len(yb)), yb] + 1e-10)
                loss  = -log_p.mean()
                epoch_loss += loss * len(yb)

                # 反向传播（仅 FC 层）
                dlogits = probs.copy()
                dlogits[np.arange(len(yb)), yb] -= 1
                dlogits /= len(yb)

                dW4 = h1.T @ dlogits
                db4 = dlogits.sum(axis=0)
                dh1 = dlogits @ self.W4.T
                dh1[h1 == 0] = 0   # ReLU gradient

                dW3 = flat.T @ dh1
                db3 = dh1.sum(axis=0)

                self.W4 -= lr * dW4
                self.b4 -= lr * db4
                self.W3 -= lr * dW3
                self.b3 -= lr * db3

            avg_loss = epoch_loss / N
            val_acc  = self.score(X_val, y_val)
            self.train_losses.append(avg_loss)
            self.val_accs.append(val_acc)

            if (epoch + 1) % 5 == 0:
                print(f"  Epoch {epoch+1:3d}/{n_epochs}  "
                      f"loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.clip(X, 0, 1)
        probs, *_ = self._forward(X, training=False)
        return probs.argmax(axis=1)

    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        return (self.predict(X) == y).mean()


# ─────────────────────── 评估工具 ───────────────────────────────

def classification_report(y_true, y_pred, class_names):
    n_cls = len(class_names)
    report = {}
    for c in range(n_cls):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        prec = tp / (tp + fp + 1e-10)
        rec  = tp / (tp + fn + 1e-10)
        f1   = 2 * prec * rec / (prec + rec + 1e-10)
        report[class_names[c]] = {"precision": prec, "recall": rec, "f1": f1,
                                   "support": (y_true == c).sum()}
    return report


def confusion_matrix(y_true, y_pred, n_cls):
    cm = np.zeros((n_cls, n_cls), dtype=np.int32)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


# ─────────────────────── 可视化 ─────────────────────────────────

CLASS_NAMES = ["Horizontal", "Vertical", "Diagonal", "Grid", "Circle"]
COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#FFA07A", "#98D8C8"]


def plot_results(model: SimpleCNN, X_val, y_val, y_pred,
                 save_path: str = "results/image_classification_results.png"):
    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#1a1a2e")
    n_cls = len(CLASS_NAMES)

    # ── 1. 训练曲线 ──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.plot(model.train_losses, color="#FF6B6B", label="Train Loss", linewidth=2)
    ax1.set_title("Training Loss", color="white", fontsize=11, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Cross-Entropy", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#0f3460", labelcolor="white")
    for s in ax1.spines.values(): s.set_edgecolor("#444")
    ax1.grid(True, alpha=0.2)

    # ── 2. 验证准确率 ──
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.plot(model.val_accs, color="#4ECDC4", label="Val Accuracy", linewidth=2)
    ax2.set_title("Validation Accuracy", color="white", fontsize=11, fontweight="bold")
    ax2.set_xlabel("Epoch", color="white")
    ax2.set_ylabel("Accuracy", color="white")
    ax2.set_ylim(0, 1.05)
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#0f3460", labelcolor="white")
    for s in ax2.spines.values(): s.set_edgecolor("#444")
    ax2.grid(True, alpha=0.2)

    # ── 3. Per-class F1 ──
    ax3 = fig.add_subplot(2, 3, 3)
    report = classification_report(y_val, y_pred, CLASS_NAMES)
    f1s = [report[c]["f1"] for c in CLASS_NAMES]
    bars = ax3.barh(CLASS_NAMES, f1s, color=COLORS, alpha=0.85)
    ax3.set_xlim(0, 1.1)
    ax3.set_title("Per-Class F1 Score", color="white", fontsize=11, fontweight="bold")
    ax3.set_facecolor("#16213e")
    ax3.tick_params(colors="white")
    for s in ax3.spines.values(): s.set_edgecolor("#444")
    for bar, val in zip(bars, f1s):
        ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                 f"{val:.3f}", va="center", color="white", fontsize=9)

    # ── 4. 混淆矩阵 ──
    ax4 = fig.add_subplot(2, 3, 4)
    cm = confusion_matrix(y_val, y_pred, n_cls)
    im = ax4.imshow(cm, cmap="Blues", aspect="auto")
    ax4.set_xticks(range(n_cls))
    ax4.set_yticks(range(n_cls))
    ax4.set_xticklabels([c[:4] for c in CLASS_NAMES], color="white", fontsize=8, rotation=45)
    ax4.set_yticklabels([c[:4] for c in CLASS_NAMES], color="white", fontsize=8)
    ax4.set_title("Confusion Matrix", color="white", fontsize=11, fontweight="bold")
    ax4.set_facecolor("#16213e")
    for i in range(n_cls):
        for j in range(n_cls):
            ax4.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="black" if cm[i, j] > cm.max()/2 else "white", fontsize=10)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # ── 5. 样本图片展示 ──
    ax5 = fig.add_subplot(2, 3, (5, 6))
    n_show = 15
    show_idx = np.concatenate([np.where(y_val == c)[0][:3] for c in range(n_cls)])
    show_idx = show_idx[:n_show]
    rows, cols = 3, 5
    for k, si in enumerate(show_idx):
        ax = fig.add_subplot(rows * 2, cols, n_cls * cols + k + 1)
        # 嵌入在底部区域的小图（简化处理）
        pass
    ax5.axis("off")
    ax5.set_facecolor("#16213e")

    # 使用子图方式绘制样本
    sample_axes = []
    for k in range(min(n_show, len(show_idx))):
        si  = show_idx[k]
        row = k // cols
        col = k % cols
        # left, bottom, width, height
        left   = 0.62 + col * 0.076
        bottom = 0.32 - row * 0.14
        ax_s   = fig.add_axes([left, bottom, 0.065, 0.10])
        ax_s.imshow(X_val[si], cmap="gray", vmin=0, vmax=1)
        label_str = CLASS_NAMES[y_val[si]][:4]
        pred_str  = CLASS_NAMES[y_pred[si]][:4]
        color     = "#4ECDC4" if y_val[si] == y_pred[si] else "#FF6B6B"
        ax_s.set_title(f"T:{label_str}\nP:{pred_str}", fontsize=6, color=color, pad=1)
        ax_s.axis("off")
        sample_axes.append(ax_s)

    fig.text(0.77, 0.46, "Sample Predictions (green=correct, red=wrong)",
             ha="center", va="center", color="white", fontsize=8)

    plt.suptitle("Image Classification — CNN on Synthetic Texture Dataset",
                 color="white", fontsize=14, fontweight="bold", y=0.99)
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ─────────────────────── 主函数 ─────────────────────────────────

def image_classification():
    """图像分类完整流程（合成纹理数据集 + 手写 CNN）"""
    print("=" * 60)
    print("   图像分类 (Image Classification) — 手写 CNN")
    print("=" * 60)

    # ── 1. 数据 ──
    print("\n[1/4] 生成合成图像数据集...")
    IMG_SIZE = 28
    N_CLASSES = 5
    X, y = make_synthetic_dataset(n_per_class=150, img_size=IMG_SIZE,
                                   n_classes=N_CLASSES, seed=42)
    X = np.clip(X, 0, 1)
    X_train, y_train, X_val, y_val = train_val_split(X, y, val_ratio=0.2, seed=42)
    print(f"  训练集: {len(X_train)} 张, 验证集: {len(X_val)} 张")
    print(f"  图像尺寸: {IMG_SIZE}×{IMG_SIZE}  类别数: {N_CLASSES}")

    # ── 2. 训练 ──
    print("\n[2/4] 训练 CNN 模型...")
    model = SimpleCNN(img_size=IMG_SIZE, n_classes=N_CLASSES, seed=42)
    model.fit(X_train, y_train, X_val, y_val,
              n_epochs=25, lr=0.01, batch_size=32)

    # ── 3. 评估 ──
    print("\n[3/4] 评估结果...")
    y_pred = model.predict(X_val)
    val_acc = (y_pred == y_val).mean()
    print(f"  验证准确率: {val_acc:.4f}")

    report = classification_report(y_val, y_pred, CLASS_NAMES)
    print(f"\n  {'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>8} {'Support':>8}")
    print("  " + "-" * 46)
    for cls in CLASS_NAMES:
        r = report[cls]
        print(f"  {cls:<12} {r['precision']:>10.4f} {r['recall']:>10.4f} "
              f"{r['f1']:>8.4f} {r['support']:>8}")

    # ── 4. 可视化 ──
    print("\n[4/4] 生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(model, X_val, y_val, y_pred,
                 save_path="results/image_classification_results.png")

    print("\n[DONE] 图像分类完成!")
    return model, report


if __name__ == "__main__":
    image_classification()
