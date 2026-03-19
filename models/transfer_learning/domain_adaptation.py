"""
迁移学习：域适应 (Domain Adaptation)
======================================
实现域适应方法，将源域知识迁移到目标域

核心思想：
  - 源域：有标签数据 (X_s, y_s)
  - 目标域：无标签/少标签数据 (X_t, 无 y_t)
  - 目标：让特征提取器学习域不变特征，减少域差异

实现方法：
  1. CORAL (CORrelation Alignment)：对齐源域和目标域的协方差
  2. DANN (Domain Adversarial Neural Network)：域对抗训练
  3. 简化演示：协方差对齐 + 二分类域判别器
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── CORAL 对齐 ─────────────────────────────

def coral(source, target):
    """
    CORAL (Correlation Alignment)

    目标：变换源域特征，使其与目标域的协方差对齐

    步骤：
      1. 计算源域和目标域协方差 C_s, C_t
      2. 线性变换 A 使得 A * C_s * A^T = C_t
      3. 简化：A = C_t^{-1/2} * C_s^{1/2}
    """
    # 中心化
    source_c = source - source.mean(axis=0)
    target_c = target - target.mean(axis=0)

    # 协方差
    C_s = (source_c.T @ source_c) / (len(source) - 1)
    C_t = (target_c.T @ target_c) / (len(target) - 1)

    # 协方差平方根（使用 SVD）
    def sqrtm(C):
        U, S, Vt = np.linalg.svd(C)
        return U @ np.diag(np.sqrt(S)) @ Vt

    C_s_sqrt = sqrtm(C_s)
    C_t_sqrt = sqrtm(C_t)
    C_t_inv_sqrt = sqrtm(np.linalg.pinv(C_t))

    # 变换矩阵
    A = C_t_sqrt @ np.linalg.pinv(C_s_sqrt)

    # 应用变换
    source_aligned = source @ A.T
    return source_aligned, A


# ─────────────────────── 简化 DANN ─────────────────────────────

class DomainClassifier:
    """
    简化域判别器（判断样本来自源域还是目标域）

    输入：特征 x
    输出：域标签（0=源域，1=目标域）
    """
    def __init__(self, input_dim, hidden_dim=32):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)

    def forward(self, x):
        h = np.maximum(0, x @ self.W1 + self.b1)
        out = 1 / (1 + np.exp(-(h @ self.W2 + self.b2)))
        return out

    def train(self, X_s, X_t, epochs=30, lr=0.01):
        """训练域判别器"""
        losses = []
        X = np.vstack([X_s, X_t])
        y = np.hstack([np.zeros(len(X_s)), np.ones(len(X_t))])

        for epoch in range(epochs):
            # 简化 SGD
            loss = 0
            correct = 0
            for i in range(len(X)):
                pred = self.forward(X[i:i+1])[0,0]
                loss += -(y[i] * np.log(pred + 1e-8) + (1 - y[i]) * np.log(1 - pred + 1e-8))
                if (pred >= 0.5) == y[i]:
                    correct += 1
                # 简化梯度更新
                grad = (pred - y[i])
                h = np.maximum(0, X[i] @ self.W1 + self.b1)
                d_out = grad
                d_h = d_out * self.W2.T * (h > 0)
                self.W2 -= lr * (h.T @ np.array([[d_out]]))
                self.b2 -= lr * d_out
                self.W1 -= lr * X[i].T @ d_h
                self.b1 -= lr * d_h.sum(axis=0)

            avg_loss = loss / len(X)
            acc = correct / len(X)
            losses.append(avg_loss)
            if epoch % 5 == 0:
                print(f"  Epoch {epoch}: Loss={avg_loss:.4f}, Acc={acc:.4f}")

        return losses

    def get_domain_features(self, X, threshold=0.7):
        """获取域混淆特征（输出概率接近 0.5 的样本）"""
        preds = self.forward(X)[:, 0]
        # 域不变特征：判别器难以区分的样本（pred ≈ 0.5）
        domain_confusion = 1 - np.abs(preds - 0.5) * 2
        return X * domain_confusion[:, np.newaxis]


# ─────────────────────── 生成域偏移数据 ─────────────────────────

def generate_domain_shift(n_source=400, n_target=400, seed=42):
    """
    生成域偏移数据：源域和目标域有类别分布偏移

    源域：2 类高斯簇
    目标域：平移 + 旋转后的 2 类高斯簇
    """
    np.random.seed(seed)
    # 源域
    X_s, y_s = make_blobs(n_samples=n_source, centers=2, cluster_std=1.5, random_state=42)
    # 目标域：旋转 + 平移
    X_t, y_t = make_blobs(n_samples=n_target, centers=2, cluster_std=1.8, random_state=123)
    theta = np.pi / 6
    rot = np.array([[np.cos(theta), -np.sin(theta)],
                    [np.sin(theta),  np.cos(theta)]])
    X_t = X_t @ rot.T + np.array([1.5, 1.5])
    return X_s, y_s, X_t, y_t


def domain_adaptation():
    print("迁移学习：域适应运行中...\n")

    # ── 生成域偏移数据 ────────────────────────────────────────────
    print("1. 生成源域和目标域数据（平移+旋转）...")
    X_source, y_source, X_target, y_target = generate_domain_shift(n_source=400, n_target=400)
    print(f"   源域: {X_source.shape[0]} 样本, 目标域: {X_target.shape[0]} 样本")

    # ── 基线：直接用源域模型预测目标域 ─────────────────────────────
    print("\n2. 基线：源域模型在目标域的性能...")
    from sklearn.linear_model import LogisticRegression
    clf_source = LogisticRegression(max_iter=500, random_state=42)
    clf_source.fit(X_source, y_source)
    y_pred_baseline = clf_source.predict(X_target)
    acc_baseline = accuracy_score(y_target, y_pred_baseline)
    print(f"   目标准确率: {acc_baseline:.4f}")

    # ── 方法1：CORAL 协方差对齐 ───────────────────────────────────
    print("\n3. CORAL 对齐...")
    X_source_aligned, A_coral = coral(X_source, X_target)
    clf_coral = LogisticRegression(max_iter=500, random_state=42)
    clf_coral.fit(X_source_aligned, y_source)
    X_target_aligned = X_target @ A_coral.T
    y_pred_coral = clf_coral.predict(X_target_aligned)
    acc_coral = accuracy_score(y_target, y_pred_coral)
    print(f"   目标准确率: {acc_coral:.4f}")

    # ── 方法2：DANN 域对抗训练（简化版） ────────────────────────────────
    print("\n4. DANN 域对抗训练（简化）...")
    domain_clf = DomainClassifier(input_dim=2, hidden_dim=32)
    domain_losses = domain_clf.train(X_source, X_target, epochs=30, lr=0.01)

    # 使用域不变特征训练分类器
    X_source_inv = domain_clf.get_domain_features(X_source)
    X_target_inv = domain_clf.get_domain_features(X_target)
    clf_dann = LogisticRegression(max_iter=500, random_state=42)
    clf_dann.fit(X_source_inv, y_source)
    y_pred_dann = clf_dann.predict(X_target_inv)
    acc_dann = accuracy_score(y_target, y_pred_dann)
    print(f"   目标准确率: {acc_dann:.4f}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n5. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：源域分布 ────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(X_source[:, 0], X_source[:, 1], c=y_source, cmap="coolwarm",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.3)
    ax.set_title("源域分布 (有标签)", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图2：目标域分布 ────────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(X_target[:, 0], X_target[:, 1], c=y_target, cmap="coolwarm",
               s=50, alpha=0.8, edgecolors="white", linewidths=0.3)
    ax.set_title("目标域分布 (无标签/少标签)", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图3：CORAL 对齐后分布 ─────────────────────────────────────
    ax = axes[0, 2]
    ax.scatter(X_source_aligned[:, 0], X_source_aligned[:, 1], c=y_source, cmap="coolwarm",
               s=50, alpha=0.6, label="源域")
    ax.scatter(X_target_aligned[:, 0], X_target_aligned[:, 1], c=y_target, cmap="coolwarm",
               marker="s", s=30, alpha=0.6, label="目标域")
    ax.set_title("CORAL 对齐后", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：域判别器决策边界 ─────────────────────────────────
    ax = axes[1, 0]
    grid_x = np.linspace(-6, 8, 100)
    grid_y = np.linspace(-6, 8, 100)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    Z_grid = np.zeros_like(X_grid)
    for i in range(100):
        for j in range(100):
            Z_grid[j, i] = domain_clf.forward(np.array([[X_grid[j, i], Y_grid[j, i]]]))[0,0]
    ax.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap="RdBu_r", alpha=0.7)
    ax.scatter(X_source[:, 0], X_source[:, 1], c="#3498db", s=30, alpha=0.6, label="源域")
    ax.scatter(X_target[:, 0], X_target[:, 1], c="#e74c3c", s=30, alpha=0.6, label="目标域")
    ax.set_title("域判别器决策边界", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图5：方法准确率对比 ─────────────────────────────────────
    ax = axes[1, 1]
    methods = ["Baseline", "CORAL", "DANN"]
    accs = [acc_baseline, acc_coral, acc_dann]
    pal = ["#f39c12", "#3498db", "#2ecc71"]
    bars = ax.bar(methods, accs, color=pal, alpha=0.85)
    ax.set_title("域适应方法准确率对比", color="white", pad=8)
    ax.set_ylabel("Target Accuracy", color="#aaa")
    ax.set_ylim(0, 1)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, accs):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=9)

    # ── 子图6：域适应方法总结表 ─────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["方法", "类型", "需目标域标签", "优势"],
        ["Baseline", "直接迁移", "否", "实现简单"],
        ["CORAL", "统计对齐", "否", "协方差对齐"],
        ["DANN", "对抗训练", "否", "域不变特征"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   colWidths=[0.2, 0.2, 0.25, 0.35])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white"); cell.set_edgecolor("#334")
    ax.set_title("域适应方法总结", color="white", pad=10)

    plt.suptitle("迁移学习：域适应 (CORAL & DANN)", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("domain_adaptation.png"))

    print("\n[DONE] 域适应完成!")


if __name__ == "__main__":
    domain_adaptation()
