"""
Adam 变体优化算法 (Adam Variants)
===================================
从零实现 Adam 家族优化器，展示各变体的改进思路

实现算法：
  1. Adam   — Adaptive Moment Estimation (Kingma & Ba 2015)
  2. AdaMax — 基于无穷范数的 Adam 变体
  3. Nadam  — Adam + Nesterov Momentum
  4. AdamW  — Adam + Weight Decay 解耦（修复 L2 正则化 bug）
  5. RAdam  — Rectified Adam（修复早期训练方差过大问题）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 测试函数 ────────────────────────────────

def rosenbrock_grad(xy, a=1, b=100):
    x, y = xy
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return np.array([dx, dy])

def rosenbrock(xy, a=1, b=100):
    x, y = xy
    return (a - x)**2 + b*(y - x**2)**2


# ─────────────────────── Adam 家族实现 ───────────────────────────

class Adam:
    """
    Adam: Adaptive Moment Estimation
      m = beta1*m + (1-beta1)*g          # 一阶矩（梯度均值）
      v = beta2*v + (1-beta2)*g^2        # 二阶矩（梯度方差）
      m_hat = m / (1-beta1^t)            # 偏差修正
      v_hat = v / (1-beta2^t)
      w = w - lr * m_hat / (sqrt(v_hat) + eps)
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        return params - self.lr * m_hat / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m = self.v = None; self.t = 0


class AdaMax:
    """
    AdaMax: 用无穷范数（max）替代 Adam 的 L2 范数
      u = max(beta2 * u, |g|)   # 无穷范数估计
    """
    def __init__(self, lr=0.002, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = self.u = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.u = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.u = np.maximum(self.b2 * self.u, np.abs(grads))
        m_hat = self.m / (1 - self.b1**self.t)
        return params - (self.lr / (self.u + self.eps)) * m_hat

    def reset(self):
        self.m = self.u = None; self.t = 0


class Nadam:
    """
    Nadam: Adam + Nesterov（先看前方一步再更新）
      使用 m_{t+1} 的前看估计替代 m_t
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        # Nesterov look-ahead: use next step momentum estimate
        m_nag = self.b1 * m_hat + (1 - self.b1) * grads / (1 - self.b1**self.t)
        return params - self.lr * m_nag / (np.sqrt(v_hat) + self.eps)

    def reset(self):
        self.m = self.v = None; self.t = 0


class AdamW:
    """
    AdamW: 解耦 Weight Decay
      Adam 的 L2 正则化其实不等价于 weight decay（因为被 v_hat 缩放了）
      AdamW 直接在参数上加 weight decay，与梯度更新解耦
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.wd = weight_decay
        self.m = self.v = None
        self.t = 0

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads**2
        m_hat = self.m / (1 - self.b1**self.t)
        v_hat = self.v / (1 - self.b2**self.t)
        # 解耦 weight decay
        return (params * (1 - self.lr * self.wd)
                - self.lr * m_hat / (np.sqrt(v_hat) + self.eps))

    def reset(self):
        self.m = self.v = None; self.t = 0


class RAdam:
    """
    RAdam: Rectified Adam
      动态计算自适应学习率的方差，当方差过大时退化为 SGD + Momentum
      修复 Adam 早期训练方差不稳定的问题
    """
    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.b1, self.b2, self.eps = beta1, beta2, eps
        self.m = self.v = None
        self.t = 0
        self.rho_inf = 2 / (1 - beta2) - 1

    def step(self, params, grads):
        if self.m is None:
            self.m = np.zeros_like(params)
            self.v = np.zeros_like(params)
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grads
        self.v = self.b2 * self.v + (1 - self.b2) * grads**2
        m_hat = self.m / (1 - self.b1**self.t)
        # 计算 SMA 长度 rho_t
        rho_t = self.rho_inf - 2 * self.t * self.b2**self.t / (1 - self.b2**self.t)
        if rho_t > 4:  # 方差可处理，使用自适应更新
            v_hat = np.sqrt(self.v / (1 - self.b2**self.t))
            rect = np.sqrt(
                (rho_t - 4) * (rho_t - 2) * self.rho_inf
                / ((self.rho_inf - 4) * (self.rho_inf - 2) * rho_t)
            )
            return params - self.lr * rect * m_hat / (v_hat + self.eps)
        else:  # 方差不稳定，退化为 SGD
            return params - self.lr * m_hat

    def reset(self):
        self.m = self.v = None; self.t = 0


# ─────────────────────── 主实验 ──────────────────────────────────

def adam_variants():
    print("Adam 变体优化算法运行中...\n")

    print("1. Rosenbrock 函数优化对比 (2000步)...")
    x0 = np.array([-1.5, 0.8])
    n_steps = 2000

    optimizers = {
        "Adam":   Adam(lr=0.01),
        "AdaMax": AdaMax(lr=0.02),
        "Nadam":  Nadam(lr=0.01),
        "AdamW":  AdamW(lr=0.01, weight_decay=0.001),
        "RAdam":  RAdam(lr=0.01),
    }

    COLORS = {
        "Adam": "#e74c3c", "AdaMax": "#3498db",
        "Nadam": "#2ecc71", "AdamW": "#f39c12", "RAdam": "#9b59b6"
    }

    histories = {}
    losses = {}
    for name, opt in optimizers.items():
        opt.reset()
        x = x0.copy().astype(float)
        traj = [x.copy()]
        loss_hist = [rosenbrock(x)]
        for _ in range(n_steps):
            g = rosenbrock_grad(x)
            x = opt.step(x, g)
            traj.append(x.copy())
            loss_hist.append(rosenbrock(x))
        histories[name] = np.array(traj)
        losses[name]    = loss_hist

    # ── 可视化 ────────────────────────────────────────────────────
    print("2. 生成可视化...")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # 等高线背景
    xs = np.linspace(-2, 1.5, 300)
    ys = np.linspace(-0.5, 1.5, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = (1 - X)**2 + 100*(Y - X**2)**2

    # ── 子图1：所有算法轨迹 ──────────────────────────────────────
    ax = axes[0, 0]
    ax.contourf(X, Y, np.log1p(Z), levels=40, cmap="plasma", alpha=0.7)
    for name, traj in histories.items():
        n = min(800, len(traj))
        ax.plot(traj[:n, 0], traj[:n, 1], color=COLORS[name], alpha=0.85,
                linewidth=1.3, label=name)
    ax.plot(1, 1, "w*", markersize=14)
    ax.plot(x0[0], x0[1], "wo", markersize=7)
    ax.set_title("Rosenbrock 优化轨迹（前800步）", color="white", pad=10)
    ax.set_xlabel("x", color="#aaa"); ax.set_ylabel("y", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图2：收敛曲线 (log) ────────────────────────────────────
    ax = axes[0, 1]
    for name, ls in losses.items():
        ax.semilogy(ls, color=COLORS[name], label=name, alpha=0.9, linewidth=1.5)
    ax.set_title("收敛曲线 (log scale)", color="white", pad=10)
    ax.set_xlabel("迭代步数", color="#aaa"); ax.set_ylabel("Loss", color="#aaa")
    ax.tick_params(colors="gray"); ax.grid(alpha=0.2, color="#555")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图3：最终损失 Top-K 标记 ──────────────────────────────
    ax = axes[0, 2]
    sorted_items = sorted(losses.items(), key=lambda x: x[1][-1])
    names_s = [n for n, _ in sorted_items]
    vals_s   = [v[-1] for _, v in sorted_items]
    bars = ax.barh(names_s, vals_s, color=[COLORS[n] for n in names_s], alpha=0.85)
    ax.set_title("最终 Loss（越小越好）", color="white", pad=10)
    ax.set_xlabel("Final Loss", color="#aaa")
    ax.set_xscale("log")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, vals_s):
        ax.text(v * 1.1, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", color="white", fontsize=8)

    # ── 子图4：学习率有效值随步数变化 (Adam 内部) ───────────────
    ax = axes[1, 0]
    # 模拟 Adam 有效学习率 = lr * sqrt(1-b2^t) / (1-b1^t)
    ts = np.arange(1, 501)
    for lr_val, color, label in [
        (0.01, "#e74c3c", "lr=0.01"),
        (0.001, "#3498db", "lr=0.001"),
    ]:
        eff_lr = lr_val * np.sqrt(1 - 0.999**ts) / (1 - 0.9**ts)
        ax.plot(ts, eff_lr, color=color, label=label, linewidth=1.5, alpha=0.9)
    ax.axhline(0.01, color="#f39c12", linestyle="--", alpha=0.6, label="asymptote lr=0.01")
    ax.set_title("Adam 有效学习率预热曲线", color="white", pad=10)
    ax.set_xlabel("步数 t", color="#aaa"); ax.set_ylabel("有效学习率", color="#aaa")
    ax.tick_params(colors="gray"); ax.grid(alpha=0.2, color="#555")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图5：RAdam vs Adam 早期步骤对比 ───────────────────────
    ax = axes[1, 1]
    early_n = 100
    for name, ls in losses.items():
        ax.plot(ls[:early_n], color=COLORS[name], label=name, alpha=0.9, linewidth=1.5)
    ax.set_title("前100步收敛（早期稳定性）", color="white", pad=10)
    ax.set_xlabel("迭代步数", color="#aaa"); ax.set_ylabel("Loss", color="#aaa")
    ax.tick_params(colors="gray"); ax.grid(alpha=0.2, color="#555")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图6：算法对比表 ────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["算法", "核心改进", "适用场景"],
        ["Adam", "一阶+二阶矩估计", "通用首选"],
        ["AdaMax", "无穷范数稳定", "嵌入层/稀疏"],
        ["Nadam", "Nesterov前看", "RNN序列任务"],
        ["AdamW", "解耦weight decay", "预训练模型"],
        ["RAdam", "方差整流预热", "对LR敏感任务"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   colWidths=[0.2, 0.45, 0.35])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white"); cell.set_edgecolor("#334")
    ax.set_title("Adam 家族特性总结", color="white", pad=10)

    plt.suptitle("Adam 变体优化算法对比", color="white", fontsize=14,
                 y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("adam_variants.png"))

    print("\n=== 最终损失（2000步后） ===")
    for name, ls in sorted(losses.items(), key=lambda x: x[1][-1]):
        print(f"  {name:<8}: {ls[-1]:.4f}")

    print("\n[DONE] Adam 变体优化算法完成!")


if __name__ == "__main__":
    adam_variants()
