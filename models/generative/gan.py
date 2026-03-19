"""
生成对抗网络 (Generative Adversarial Network, GAN)
================================================
实现 GAN：生成器与判别器博弈训练

核心思想：
  - 生成器 G：噪声 z → 假样本 x_hat
  - 判别器 D：输入 x → 真假概率 p
  - 博弈：G 想骗过 D，D 想区分真假
  - 目标：纳什均衡（G 生成真实样本，D 无法区分）

损失函数（原始 GAN）：
  - D: 最大化 log D(x) + log(1 - D(G(z)))
  - G: 最小化 log(1 - D(G(z))) 或等价最大化 log D(G(z))

实现内容：
  1. 纯 NumPy 小型 GAN (MLP G/D)
  2. 合成 2D 数据（双月牙形）训练
  3. 可视化：生成样本分布、判别边界、损失曲线
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import expit
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── GAN 实现 ────────────────────────────────

class MLP:
    """简单多层感知机"""
    def __init__(self, layer_dims, activation="relu", output_act="sigmoid"):
        self.layer_dims = layer_dims
        self.activation = activation
        self.output_act = output_act
        self.params = {}
        self.cache = {}

        # 初始化权重
        for i in range(1, len(layer_dims)):
            self.params[f"W{i}"] = np.random.randn(layer_dims[i-1], layer_dims[i]) * 0.01
            self.params[f"b{i}"] = np.zeros(layer_dims[i])

    def relu(self, x):  return np.maximum(0, x)
    def sigmoid(self, x): return expit(x)

    def forward(self, x):
        a = x
        for i in range(1, len(self.layer_dims)-1):
            z = a @ self.params[f"W{i}"] + self.params[f"b{i}"]
            a = self.relu(z)
            self.cache[f"a{i}"] = a
        # 输出层
        z_last = a @ self.params[f"W{len(self.layer_dims)-1}"] + self.params[f"b{len(self.layer_dims)-1}"]
        if self.output_act == "sigmoid":
            a_last = self.sigmoid(z_last)
        elif self.output_act == "tanh":
            a_last = np.tanh(z_last)
        else:
            a_last = z_last
        self.cache["a_last"] = a_last
        return a_last

    def backward(self, x, dout):
        grads = {}
        m = x.shape[0]
        # 反向传播
        da = dout
        for i in range(len(self.layer_dims)-1, 0, -1):
            W = self.params[f"W{i}"]
            a_prev = self.cache[f"a{i-1}"] if i > 1 else x
            if self.output_act == "sigmoid" and i == len(self.layer_dims)-1:
                # sigmoid 导数
                a_last = self.cache["a_last"]
                dz = da * a_last * (1 - a_last)
            elif self.output_act == "tanh" and i == len(self.layer_dims)-1:
                a_last = self.cache["a_last"]
                dz = da * (1 - a_last**2)
            else:
                dz = da
            a = self.cache[f"a{i}"] if i > 1 else x
            da_prev = dz @ W.T
            grads[f"W{i}"] = a.T @ dz / m
            grads[f"b{i}"] = dz.mean(axis=0)
            da = da_prev
        return grads

    def update(self, grads, lr=0.01):
        for i in range(1, len(self.layer_dims)):
            self.params[f"W{i}"] -= lr * grads[f"W{i}"]
            self.params[f"b{i}"] -= lr * grads[f"b{i}"]


class GAN:
    """
    小型 GAN 实现（MLP 生成器 + 判别器）

    参数：
      z_dim:      噪声维度
      data_dim:   数据维度
      hidden_dim: 隐藏层维度
      lr:         学习率
    """
    def __init__(self, z_dim=2, data_dim=2, hidden_dim=128, lr=0.0002):
        self.z_dim = z_dim
        self.data_dim = data_dim
        self.lr = lr

        # 生成器：z -> hidden -> data
        self.G = MLP([z_dim, hidden_dim, hidden_dim, data_dim],
                     activation="relu", output_act="tanh")

        # 判别器：data -> hidden -> [0,1]
        self.D = MLP([data_dim, hidden_dim, hidden_dim, 1],
                     activation="relu", output_act="sigmoid")

        self.loss_D_history = []
        self.loss_G_history = []

    def generator_loss(self, fake_output):
        """生成器损失：最小化 log(1 - D(G(z))) ≈ 最大化 log(D(G(z)))"""
        eps = 1e-8
        return -np.log(fake_output + eps).mean()

    def discriminator_loss(self, real_output, fake_output):
        """判别器损失：-log D(x) - log(1 - D(G(z)))"""
        eps = 1e-8
        loss_real = -np.log(real_output + eps).mean()
        loss_fake = -np.log(1 - fake_output + eps).mean()
        return loss_real + loss_fake

    def train_step(self, X_real, batch_size=128, n_critic=1):
        """训练一步：更新 D n_critic 次，更新 G 1 次"""
        m = X_real.shape[0]
        rng = np.random.default_rng()

        for _ in range(n_critic):
            # 真实样本
            idx = rng.choice(m, batch_size)
            x_real = X_real[idx]
            # 假样本
            z = rng.standard_normal((batch_size, self.z_dim))
            x_fake = self.G.forward(z)

            # D 的前向
            d_real = self.D.forward(x_real)
            d_fake = self.D.forward(x_fake)

            # D 的损失和梯度
            loss_D = self.discriminator_loss(d_real, d_fake)
            # D 的梯度（数值近似，简化演示）
            # D 对真实输出梯度
            d_loss_dreal = -1 / (d_real + 1e-8) / batch_size
            d_loss_dfake = 1 / (1 - d_fake + 1e-8) / batch_size
            # 简单更新：直接用数值梯度
            grad_D = self._numerical_grad_D(self.D, x_real, x_fake, batch_size)

            # 更新 D
            for k in grad_D:
                self.D.params[k] -= self.lr * grad_D[k]

        # 更新 G
        z = rng.standard_normal((batch_size, self.z_dim))
        x_fake = self.G.forward(z)
        d_fake = self.D.forward(x_fake)
        loss_G = self.generator_loss(d_fake)

        grad_G = self._numerical_grad_G(self.G, self.D, z, batch_size)
        for k in grad_G:
            self.G.params[k] -= self.lr * grad_G[k]

        self.loss_D_history.append(loss_D)
        self.loss_G_history.append(loss_G)
        return loss_D, loss_G

    def _numerical_grad_D(self, D, x_real, x_fake, batch_size, eps=1e-5):
        """判别器数值梯度（简化版）"""
        grad = {}
        x_concat = np.vstack([x_real, x_fake])
        y = np.vstack([np.ones((batch_size, 1)), np.zeros((batch_size, 1))])

        def loss_func(params):
            old_params = D.params.copy()
            D.params.update(params)
            d_real = D.forward(x_real)
            d_fake = D.forward(x_fake)
            loss = self.discriminator_loss(d_real, d_fake)
            D.params = old_params
            return loss

        for k, v in D.params.items():
            grad[k] = np.zeros_like(v)
            for i in np.ndindex(v.shape):
                v_eps = v.copy()
                v_eps[i] += eps
                loss_plus = loss_func({k: v_eps})
                v_eps[i] -= 2 * eps
                loss_minus = loss_func({k: v_eps})
                grad[k][i] = (loss_plus - loss_minus) / (2 * eps)
                v_eps[i] = v[i]
        return grad

    def _numerical_grad_G(self, G, D, z, batch_size, eps=1e-5):
        """生成器数值梯度（简化版）"""
        grad = {}

        def loss_func(params):
            old_params = G.params.copy()
            G.params.update(params)
            x_fake = G.forward(z)
            d_fake = D.forward(x_fake)
            loss = self.generator_loss(d_fake)
            G.params = old_params
            return loss

        for k, v in G.params.items():
            grad[k] = np.zeros_like(v)
            for i in np.ndindex(v.shape):
                v_eps = v.copy()
                v_eps[i] += eps
                loss_plus = loss_func({k: v_eps})
                v_eps[i] -= 2 * eps
                loss_minus = loss_func({k: v_eps})
                grad[k][i] = (loss_plus - loss_minus) / (2 * eps)
                v_eps[i] = v[i]
        return grad

    def sample(self, n_samples=100):
        """生成新样本"""
        z = np.random.randn(n_samples, self.z_dim)
        return self.G.forward(z)


# ─────────────────────── 生成合成数据 ─────────────────────────

def generate_synthetic_2d(n_samples=1000, seed=42):
    """生成双月牙形合成数据"""
    np.random.seed(seed)
    n = n_samples // 2
    # 月牙1
    r1 = np.random.randn(n) * 0.1
    a1 = np.random.uniform(0, np.pi, n)
    x1 = np.cos(a1) * (1 + r1) - 0.5
    y1 = np.sin(a1) * (1 + r1)
    # 月牙2
    r2 = np.random.randn(n) * 0.1
    a2 = np.random.uniform(np.pi, 2*np.pi, n)
    x2 = np.cos(a2) * (1 + r2) + 0.5
    y2 = np.sin(a2) * (1 + r2) + 0.5

    X = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
    # 归一化到 [-1, 1]
    X = (X - X.min()) / (X.ptp() + 1e-8) * 2 - 1
    return X


# ─────────────────────── 主程序 ──────────────────────────────────

def gan():
    print("GAN (生成对抗网络) 运行中...\n")

    # ── 生成合成 2D 数据 ────────────────────────────────────────────
    print("1. 生成双月牙形合成数据...")
    X_train = generate_synthetic_2d(n_samples=1000)
    print(f"   训练集: {X_train.shape[0]} 样本, {X_train.shape[1]} 维")

    # ── 训练 GAN ──────────────────────────────────────────────────
    print("\n2. 训练 GAN (z_dim=2, epochs=200)...")
    gan = GAN(z_dim=2, data_dim=2, hidden_dim=64, lr=0.002)

    epochs = 200
    for epoch in range(epochs):
        loss_D, loss_G = gan.train_step(X_train, batch_size=128, n_critic=1)
        if epoch % 20 == 0:
            print(f"  Epoch {epoch}: Loss_D = {loss_D:.4f}, Loss_G = {loss_G:.4f}")

    # ── 生成新样本 ────────────────────────────────────────────────
    print("\n3. 生成新样本...")
    X_generated = gan.sample(n_samples=1000)

    # ── 可视化 ────────────────────────────────────────────────────
    print("4. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：真实数据分布 ────────────────────────────────────────
    ax = axes[0, 0]
    ax.scatter(X_train[:, 0], X_train[:, 1], c="#3498db", alpha=0.6, s=30)
    ax.set_title("真实数据分布", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图2：生成数据分布 ────────────────────────────────────────
    ax = axes[0, 1]
    ax.scatter(X_generated[:, 0], X_generated[:, 1],
               c="#e74c3c", alpha=0.6, s=30)
    ax.set_title("GAN 生成样本分布", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图3：真实 + 生成对比 ─────────────────────────────────────
    ax = axes[0, 2]
    ax.scatter(X_train[:, 0], X_train[:, 1],
               c="#3498db", alpha=0.5, s=25, label="真实")
    ax.scatter(X_generated[:, 0], X_generated[:, 1],
               c="#e74c3c", alpha=0.5, s=25, label="生成")
    ax.set_title("真实 vs 生成对比", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2)
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：损失曲线 ────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(gan.loss_D_history, color="#e74c3c", linewidth=2, label="Loss_D")
    ax.plot(gan.loss_G_history, color="#3498db", linewidth=2, label="Loss_G")
    ax.set_title("训练损失曲线", color="white", pad=8)
    ax.set_xlabel("Step", color="#aaa"); ax.set_ylabel("Loss", color="#aaa")
    ax.legend(fontsize=9, facecolor="#0d0d1a", labelcolor="white")
    ax.grid(alpha=0.2, color="#555")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图5：判别器决策边界（网格） ─────────────────────────────
    ax = axes[1, 1]
    grid_x = np.linspace(-1.2, 1.2, 100)
    grid_y = np.linspace(-1.2, 1.2, 100)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    Z_grid = np.zeros_like(X_grid)
    for i in range(100):
        for j in range(100):
            Z_grid[j, i] = gan.D.forward(np.array([[X_grid[j, i], Y_grid[j, i]]]))
    ax.contourf(X_grid, Y_grid, Z_grid, levels=20, cmap="RdBu_r", alpha=0.7)
    ax.scatter(X_train[:, 0], X_train[:, 1],
               c="#3498db", alpha=0.6, s=20, edgecolors="white", linewidths=0.5)
    ax.set_title("判别器决策边界", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图6：GAN 架构图示 ─────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    # 简单的架构图
    ax.text(0.5, 0.9, "GAN 架构", ha="center", color="white",
            fontsize=12, fontweight="bold")
    # 生成器
    ax.text(0.2, 0.7, "生成器 G", ha="center", color="#e74c3c", fontsize=10)
    ax.text(0.2, 0.6, "噪声 z", ha="center", color="#3498db", fontsize=8)
    ax.arrow(0.2, 0.58, 0, -0.04, color="#aaa", width=0.01)
    ax.text(0.2, 0.45, "MLP", ha="center", color="white")
    ax.arrow(0.2, 0.42, 0, -0.04, color="#aaa", width=0.01)
    ax.text(0.2, 0.30, "假样本 x_hat", ha="center", color="#e74c3c", fontsize=9)

    # 判别器
    ax.text(0.8, 0.7, "判别器 D", ha="center", color="#3498db", fontsize=10)
    ax.text(0.8, 0.6, "真实 x", ha="center", color="#3498db", fontsize=8)
    ax.text(0.8, 0.5, "或", ha="center", color="white", fontsize=8)
    ax.text(0.8, 0.4, "假 x_hat", ha="center", color="#e74c3c", fontsize=8)
    ax.arrow(0.8, 0.38, 0, -0.04, color="#aaa", width=0.01)
    ax.text(0.8, 0.25, "MLP", ha="center", color="white")
    ax.arrow(0.8, 0.22, 0, -0.04, color="#aaa", width=0.01)
    ax.text(0.8, 0.10, "真/假概率", ha="center", color="#3498db", fontsize=9)

    ax.arrow(0.3, 0.30, 0.4, 0, color="#aaa", width=0.01)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)

    plt.suptitle("GAN (生成对抗网络)", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("gan.png"))

    print("\n[DONE] GAN 完成!")


if __name__ == "__main__":
    gan()
