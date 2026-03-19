"""
学习率调度策略 (Learning Rate Schedulers)
==========================================
从零实现常用学习率调度策略，展示各策略对训练收敛的影响

实现策略：
  1. StepLR       — 每 N 步衰减一次（乘以 gamma）
  2. ExponentialLR — 指数衰减
  3. CosineAnnealing — 余弦退火（含 Warm Restart）
  4. WarmupCosine  — 线性预热 + 余弦衰减（Transformer 常用）
  5. CyclicLR     — 三角形周期性学习率
  6. OneCycleLR   — Super-Convergence 单周期策略
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 调度器实现 ──────────────────────────────

class StepLR:
    """每 step_size 步将 lr 乘以 gamma"""
    def __init__(self, base_lr, step_size, gamma=0.5):
        self.base_lr = base_lr
        self.step_size = step_size
        self.gamma = gamma

    def get_lr(self, t):
        return self.base_lr * (self.gamma ** (t // self.step_size))


class ExponentialLR:
    """指数衰减：lr = base_lr * gamma^t"""
    def __init__(self, base_lr, gamma=0.995):
        self.base_lr = base_lr
        self.gamma = gamma

    def get_lr(self, t):
        return self.base_lr * (self.gamma ** t)


class CosineAnnealingLR:
    """余弦退火：lr 在 [min_lr, base_lr] 之间余弦振荡"""
    def __init__(self, base_lr, T_max, min_lr=1e-6):
        self.base_lr = base_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def get_lr(self, t):
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + np.cos(np.pi * (t % self.T_max) / self.T_max)
        )


class WarmupCosineDecay:
    """线性预热 + 余弦衰减（Transformer/BERT 常用）"""
    def __init__(self, base_lr, warmup_steps, total_steps, min_lr=1e-6):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr

    def get_lr(self, t):
        if t < self.warmup_steps:
            return self.base_lr * t / max(1, self.warmup_steps)
        progress = (t - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
        return self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + np.cos(np.pi * progress)
        )


class CyclicLR:
    """三角形周期性 LR：在 base_lr 和 max_lr 之间线性振荡"""
    def __init__(self, base_lr, max_lr, step_size_up=200, mode="triangular"):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.mode = mode

    def get_lr(self, t):
        cycle = t // (2 * self.step_size_up)
        x = abs(t / self.step_size_up - 2 * cycle - 1)
        if self.mode == "triangular2":
            scale = 1 / (2 ** cycle)
        elif self.mode == "exp_range":
            scale = 0.99994 ** t
        else:  # triangular
            scale = 1.0
        return self.base_lr + (self.max_lr - self.base_lr) * max(0, 1 - x) * scale


class OneCycleLR:
    """
    One-Cycle Policy（Leslie Smith 2018）
    Phase 1: base_lr -> max_lr (pct_start 比例)
    Phase 2: max_lr -> min_lr (1 - pct_start 比例)
    """
    def __init__(self, base_lr, max_lr, total_steps, pct_start=0.3, final_div=100):
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.min_lr = max_lr / final_div

    def get_lr(self, t):
        warmup_steps = int(self.total_steps * self.pct_start)
        if t <= warmup_steps:
            return self.base_lr + (self.max_lr - self.base_lr) * t / warmup_steps
        else:
            progress = (t - warmup_steps) / (self.total_steps - warmup_steps)
            return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (
                1 + np.cos(np.pi * progress)
            )


# ─────────────────────── 模拟训练 ────────────────────────────────

def simulate_training(scheduler, total_steps=1000, noise_scale=0.3, seed=42):
    """
    用调度器驱动 SGD 在二次函数上训练，模拟 loss 曲线
    f(w) = w^2, global min at w=0
    """
    rng = np.random.default_rng(seed)
    w = 5.0
    losses = []
    lrs = []
    for t in range(total_steps):
        lr = scheduler.get_lr(t)
        lrs.append(lr)
        grad = 2 * w + rng.normal(0, noise_scale)  # 带噪声梯度
        w = w - lr * grad
        losses.append(w**2)
    return lrs, losses


def lr_scheduler():
    print("学习率调度策略运行中...\n")

    total_steps = 1000
    base_lr = 0.1
    schedulers = {
        "StepLR":       StepLR(base_lr, step_size=200, gamma=0.5),
        "ExpLR":        ExponentialLR(base_lr, gamma=0.995),
        "CosineAnneal": CosineAnnealingLR(base_lr, T_max=200),
        "WarmupCosine": WarmupCosineDecay(base_lr, warmup_steps=100, total_steps=total_steps),
        "CyclicLR":     CyclicLR(base_lr * 0.1, base_lr, step_size_up=150),
        "OneCycleLR":   OneCycleLR(base_lr * 0.1, base_lr, total_steps=total_steps),
    }

    COLORS = {
        "StepLR": "#e74c3c", "ExpLR": "#3498db",
        "CosineAnneal": "#2ecc71", "WarmupCosine": "#f39c12",
        "CyclicLR": "#9b59b6", "OneCycleLR": "#1abc9c",
    }

    print(f"1. 模拟 {total_steps} 步训练...")
    results = {}
    for name, sched in schedulers.items():
        lrs, losses = simulate_training(sched, total_steps)
        results[name] = {"lr": lrs, "loss": losses}

    # ── 可视化 ─────────────────────────────────────────────────
    print("2. 生成可视化...")
    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    t_axis = np.arange(total_steps)

    # ── 上4格：各调度器 LR + Loss 双轴 ─────────────────────────
    sched_names = list(schedulers.keys())
    positions = [(0, 0), (0, 1), (1, 0), (1, 1)]
    for (r, c), name in zip(positions, sched_names[:4]):
        ax = axes[r, c]
        d = results[name]
        color = COLORS[name]
        ax2 = ax.twinx()
        ax.plot(t_axis, d["lr"], color=color, linewidth=1.8, alpha=0.9, label="LR")
        # 滑动平均 loss
        loss_smooth = np.convolve(d["loss"], np.ones(20)/20, mode="same")
        ax2.plot(t_axis, loss_smooth, color="#aaa", linewidth=1.2, alpha=0.7, linestyle="--", label="Loss")
        ax.set_title(name, color="white", pad=6, fontsize=11)
        ax.set_xlabel("步数", color="#aaa")
        ax.set_ylabel("Learning Rate", color=color)
        ax2.set_ylabel("Loss (smoothed)", color="#aaa")
        ax.tick_params(colors="gray"); ax2.tick_params(colors="gray")
        for sp in ax.spines.values(): sp.set_color("#444")
        # 合并图例
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2,
                  fontsize=7, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图5：所有 LR 曲线对比 ─────────────────────────────────
    ax = axes[2, 0]
    for name, d in results.items():
        ax.plot(t_axis, d["lr"], color=COLORS[name], label=name,
                linewidth=1.5, alpha=0.9)
    ax.set_title("各调度策略 LR 对比", color="white", pad=10)
    ax.set_xlabel("步数", color="#aaa"); ax.set_ylabel("Learning Rate", color="#aaa")
    ax.tick_params(colors="gray"); ax.grid(alpha=0.2, color="#555")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图6：最终 Loss 对比 ────────────────────────────────────
    ax = axes[2, 1]
    for name, d in results.items():
        loss_smooth = np.convolve(d["loss"], np.ones(30)/30, mode="same")
        ax.semilogy(t_axis, np.maximum(loss_smooth, 1e-6), color=COLORS[name],
                    label=name, linewidth=1.5, alpha=0.9)
    ax.set_title("收敛曲线对比（平滑）", color="white", pad=10)
    ax.set_xlabel("步数", color="#aaa"); ax.set_ylabel("Loss (log)", color="#aaa")
    ax.tick_params(colors="gray"); ax.grid(alpha=0.2, color="#555")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    plt.suptitle("学习率调度策略对比", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("lr_scheduler.png"))

    print("\n=== 各调度策略最终 Loss (后100步均值) ===")
    for name, d in sorted(results.items(), key=lambda x: np.mean(x[1]["loss"][-100:])):
        avg_loss = np.mean(d["loss"][-100:])
        final_lr = d["lr"][-1]
        print(f"  {name:<15}: avg_loss={avg_loss:.4f}  final_lr={final_lr:.6f}")

    print("\n[DONE] 学习率调度策略完成!")


if __name__ == "__main__":
    lr_scheduler()
