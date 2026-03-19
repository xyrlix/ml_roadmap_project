"""
扩散模型 (Diffusion Models)
===============================
实现 DDPM (Denoising Diffusion Probabilistic Models)

核心思想：
  1. 前向过程：逐步添加高斯噪声，最终变成纯噪声
     q(x_t | x_{t-1}) = N(x_t; sqrt(1-β_t)*x_{t-1}, β_t*I)
  2. 反向过程：训练神经网络从噪声恢复图像
     p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), σ_θ(x_t,t))
  3. 训练目标：预测噪声 ε_θ(x_t,t)
  4. 采样：从噪声逐步去噪生成样本

简化实现（1D 数据）：
  - 演示扩散/去噪过程
  - 训练小型 MLP 预测噪声
  - 可视化：扩散轨迹、预测误差、生成新样本
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 扩散过程 ─────────────────────────────

class DiffusionProcess:
    """
    简化版扩散过程（1D 数据）

    参数：
      n_steps: 扩散步数 T
      beta_schedule: β_t 调度（linear/cosine）
    """
    def __init__(self, n_steps=100, beta_schedule="linear"):
        self.n_steps = n_steps
        if beta_schedule == "linear":
            self.betas = np.linspace(0.0001, 0.02, n_steps)
        elif beta_schedule == "cosine":
            s = 0.008
            t = np.linspace(0, n_steps, n_steps + 1)
            f = np.cos((t / n_steps) * (np.pi / 2) * (1 - s) + s)**2
            alphas_cum = f / f[0]
            self.betas = 1 - (alphas_cum[1:] / alphas_cum[:-1])
        else:
            self.betas = np.linspace(0.0001, 0.02, n_steps)

        self.alphas = 1 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas)

    def q_sample(self, x_start, t, noise=None):
        """
        前向扩散：从 x_0 采样 x_t

        x_t = sqrt(ᾱ_t) * x_0 + sqrt(1-ᾱ_t) * ε
        """
        if noise is None:
            noise = np.random.randn(*x_start.shape)
        sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod[t])
        sqrt_one_minus_alphas_cumprod = np.sqrt(1 - self.alphas_cumprod[t])
        return sqrt_alphas_cumprod * x_start + sqrt_one_minus_alphas_cumprod * noise

    def get_posterior_mean_variance(self, x_t, t, x_start_pred):
        """
        后验 q(x_{t-1} | x_t, x_0) 的均值和方差
        """
        sqrt_recip_alphas_cumprod = np.sqrt(1 / self.alphas_cumprod[t])
        sqrt_recipm1_alphas_cumprod = np.sqrt(1 / self.alphas_cumprod[t] - 1)
        posterior_mean = (
            sqrt_recip_alphas_cumprod * x_t
            - sqrt_recipm1_alphas_cumprod * x_start_pred
        ) / self.alphas[t]
        posterior_variance = self.betas[t] * (1 - self.alphas_cumprod[t-1]) / (1 - self.alphas_cumprod[t])
        return posterior_mean, posterior_variance


# ─────────────────────── 噪声预测网络 ─────────────────────────

class NoisePredictor:
    """
    噪声预测网络 ε_θ(x_t, t) (简化 MLP）

    输入：x_t (n_samples, n_dims) + 时间嵌入 t
    输出：预测的噪声 ε_pred (n_samples, n_dims)
    """
    def __init__(self, n_dims, hidden_dim=64):
        self.n_dims = n_dims
        self.hidden_dim = hidden_dim

        # 权重：x_t -> h -> ε
        self.W1 = np.random.randn(n_dims + 1, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.W2 = np.random.randn(hidden_dim, hidden_dim) * 0.01
        self.b2 = np.zeros(hidden_dim)
        self.W3 = np.random.randn(hidden_dim, n_dims) * 0.01
        self.b3 = np.zeros(n_dims)

    def forward(self, x_t, t):
        """前向传播：预测噪声"""
        # 时间嵌入：t / T (归一化)
        t_embed = np.array([[t / self.n_steps]] * x_t.shape[0])
        # 拼接 x_t 和 t
        h = np.maximum(0, np.hstack([x_t, t_embed]) @ self.W1 + self.b1)
        h = np.maximum(0, h @ self.W2 + self.b2)
        eps_pred = h @ self.W3 + self.b3
        return eps_pred

    def train_step(self, x_0, t, lr=0.001):
        """训练一步：预测噪声 ε_θ，更新权重"""
        # 前向扩散：生成 x_t
        noise = np.random.randn(*x_0.shape)
        diffusion = DiffusionProcess(n_steps=100)
        x_t = diffusion.q_sample(x_0, t, noise)

        # 预测噪声
        eps_pred = self.forward(x_t, t)

        # 损失：MSE(ε, ε_θ)
        loss = ((noise - eps_pred)**2).mean()

        # 反向传播（简化梯度）
        # 使用数值近似梯度
        grad = self._numerical_grad(x_t, t, noise)
        for k in grad:
            if hasattr(self, k):
                setattr(self, k, getattr(self, k) - lr * grad[k])

        return loss

    def _numerical_grad(self, x_t, t, noise, eps=1e-5):
        """数值近似梯度（简化演示）"""
        grad = {}

        def loss_func(params):
            old_params = {k: getattr(self, k) for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
            for k, v in params.items():
                setattr(self, k, v)
            eps_pred = self.forward(x_t, t)
            loss = ((noise - eps_pred)**2).mean()
            # 恢复
            for k, v in old_params.items():
                setattr(self, k, v)
            return loss

        for k in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            param = getattr(self, k)
            grad_k = np.zeros_like(param)
            for i in np.ndindex(param.shape):
                param_eps = param.copy()
                param_eps[i] += eps
                loss_plus = loss_func({k: param_eps})
                param_eps[i] -= 2 * eps
                loss_minus = loss_func({k: param_eps})
                grad_k[i] = (loss_plus - loss_minus) / (2 * eps)
                param_eps[i] = param[i]
            grad[k] = grad_k

        return grad


# ─────────────────────── 采样（去噪） ─────────────────────────

def p_sample(model, diffusion, x_t, t):
    """
    反向采样：从 x_t 采样 x_{t-1}

    μ_θ(x_t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t,t))
    """
    eps_pred = model.forward(x_t, t)
    sqrt_recip_alphas = np.sqrt(1 / diffusion.alphas[t])
    sqrt_one_minus_alphas_cumprod = np.sqrt(1 - diffusion.alphas_cumprod[t])
    mu = sqrt_recip_alphas * (x_t - diffusion.betas[t] / sqrt_one_minus_alphas_cumprod * eps_pred)
    posterior_variance = diffusion.betas[t] * (1 - diffusion.alphas_cumprod[t-1]) / (1 - diffusion.alphas_cumprod[t])
    noise = np.random.randn(*x_t.shape)
    return mu + np.sqrt(posterior_variance) * noise


# ─────────────────────── 生成 1D 合成数据 ─────────────────────

def generate_1d_data(n_samples=1000, seed=42):
    """生成混合高斯分布的 1D 数据"""
    np.random.seed(seed)
    # 两个高斯混合
    data1 = np.random.randn(n_samples // 2) * 0.2 - 1.0
    data2 = np.random.randn(n_samples // 2) * 0.2 + 1.0
    X = np.hstack([data1, data2])
    return X


# ─────────────────────── 主程序 ──────────────────────────────────

def diffusion():
    print("扩散模型 (DDPM) 运行中...\n")

    # ── 生成 1D 合成数据 ────────────────────────────────────────────
    print("1. 生成 1D 混合高斯数据...")
    X_train = generate_1d_data(n_samples=800, seed=42)
    n_dims = 1
    n_steps = 50
    print(f"   训练集: {X_train.shape[0]} 样本, {n_dims} 维")

    # ── 初始化扩散过程和噪声预测网络 ─────────────────────────────
    diffusion = DiffusionProcess(n_steps=n_steps, beta_schedule="linear")
    model = NoisePredictor(n_dims=n_dims, hidden_dim=64)

    # ── 训练 ─────────────────────────────────────────────────────
    print("\n2. 训练噪声预测网络 (epochs=100)...")
    losses = []
    rng = np.random.default_rng(42)

    for epoch in range(100):
        epoch_loss = 0
        for _ in range(10):  # 每个 epoch 训练 10 批
            # 随机采样 x_0
            idx = rng.choice(len(X_train), 128)
            x_0 = X_train[idx].reshape(-1, 1)
            # 随机采样时间 t
            t = rng.integers(0, n_steps, size=(len(x_0), 1)).squeeze()
            loss = model.train_step(x_0, t, lr=0.005)
            epoch_loss += loss

        avg_loss = epoch_loss / 10
        losses.append(avg_loss)
        if epoch % 10 == 0:
            print(f"  Epoch {epoch}: Loss = {avg_loss:.4f}")

    # ── 演示扩散过程（单个样本） ───────────────────────────────
    print("\n3. 演示前向扩散过程...")
    x_0_demo = X_train[0].reshape(-1, 1)
    diffusion_trajectory = [x_0_demo.copy()]
    for t in range(1, n_steps):
        x_t = diffusion.q_sample(x_0_demo, t)
        diffusion_trajectory.append(x_t.copy())

    # ── 采样（反向去噪） ─────────────────────────────────────────
    print("\n4. 采样生成新样本...")
    x_T = np.random.randn(100, 1)  # 纯噪声
    x_generated = [x_T.copy()]
    for t in reversed(range(1, n_steps)):
        x_t_minus_1 = p_sample(model, diffusion, x_generated[-1], t-1)
        x_generated.append(x_t_minus_1.copy())

    # ── 可视化 ────────────────────────────────────────────────────
    print("5. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：原始数据分布 ────────────────────────────────────────
    ax = axes[0, 0]
    ax.hist(X_train, bins=50, color="#3498db", alpha=0.8, edgecolor="#333")
    ax.set_title("原始数据分布", color="white", pad=8)
    ax.set_xlabel("x", color="#aaa"); ax.set_ylabel("Count", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图2：前向扩散过程（不同时间步） ────────────────────────
    ax = axes[0, 1]
    time_steps_demo = [0, 10, 25, 40, 49]
    for t in time_steps_demo:
        x_t = diffusion.q_sample(x_0_demo, t)
        ax.hist(x_t, bins=30, alpha=0.4, label=f"t={t}")
    ax.set_title("前向扩散过程（单个样本）", color="white", pad=8)
    ax.set_xlabel("x", color="#aaa"); ax.set_ylabel("Count", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图3：β_t 调度 ────────────────────────────────────────
    ax = axes[0, 2]
    ax.plot(diffusion.betas, color="#e74c3c", linewidth=2, label="β_t")
    ax.plot(diffusion.alphas, color="#3498db", linewidth=2, label="α_t")
    ax.set_title("扩散调度 α_t & β_t", color="white", pad=8)
    ax.set_xlabel("Time Step t", color="#aaa"); ax.set_ylabel("Value", color="#aaa")
    ax.legend(fontsize=9, facecolor="#0d0d1a", labelcolor="white")
    ax.grid(alpha=0.2, color="#555")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：训练损失曲线 ────────────────────────────────────────
    ax = axes[1, 0]
    ax.plot(losses, color="#e74c3c", linewidth=2)
    ax.set_title("训练损失曲线", color="white", pad=8)
    ax.set_xlabel("Epoch", color="#aaa"); ax.set_ylabel("Loss", color="#aaa")
    ax.grid(alpha=0.2, color="#555")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图5：反向去噪采样过程（不同时间步） ───────────────────────
    ax = axes[1, 1]
    sample_steps = [0, 10, 25, 40, 49]
    for t in sample_steps:
        x_sample = x_generated[-1 - t]  # 反向
        ax.hist(x_sample, bins=30, alpha=0.4, label=f"t={t}")
    ax.hist(X_train, bins=30, alpha=0.3, color="#aaa", label="真实")
    ax.set_title("反向去噪采样过程", color="white", pad=8)
    ax.set_xlabel("x", color="#aaa"); ax.set_ylabel("Count", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图6：生成样本 vs 真实对比 ───────────────────────────────
    ax = axes[1, 2]
    generated_final = x_generated[-1]
    ax.hist(generated_final, bins=40, alpha=0.6, color="#e74c3c", label="生成")
    ax.hist(X_train, bins=40, alpha=0.4, color="#3498db", label="真实")
    ax.set_title("生成 vs 真实分布", color="white", pad=8)
    ax.set_xlabel("x", color="#aaa"); ax.set_ylabel("Count", color="#aaa")
    ax.legend(fontsize=9, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    plt.suptitle("扩散模型 (DDPM)", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("diffusion.png"))

    print("\n[DONE] 扩散模型完成!")


if __name__ == "__main__":
    diffusion()
