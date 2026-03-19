"""
SGD 变体优化算法 (SGD Variants)
================================
从零实现经典梯度下降变体，展示各算法在不同损失曲面上的收敛行为

实现算法：
  1. 朴素 SGD (Vanilla SGD)
  2. SGD + Momentum (动量法)
  3. SGD + Nesterov Momentum (NAG)
  4. Adagrad (自适应梯度)
  5. RMSProp (均方根传播)
  6. 对比实验：在 Rosenbrock 函数 + 二次函数上收敛速度对比
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

def rosenbrock(x, y, a=1, b=100):
    """Rosenbrock 香蕉函数：f(x,y) = (a-x)^2 + b*(y-x^2)^2，全局最小值在 (a,a)"""
    return (a - x)**2 + b * (y - x**2)**2

def rosenbrock_grad(x, y, a=1, b=100):
    """Rosenbrock 梯度"""
    dx = -2*(a - x) - 4*b*x*(y - x**2)
    dy = 2*b*(y - x**2)
    return np.array([dx, dy])

def quadratic(w, A, b):
    """二次函数：f(w) = 0.5 * w^T A w - b^T w"""
    return 0.5 * w @ A @ w - b @ w

def quadratic_grad(w, A, b):
    return A @ w - b


# ─────────────────────── 优化器实现 ──────────────────────────────

class VanillaSGD:
    """朴素随机梯度下降"""
    def __init__(self, lr=0.001):
        self.lr = lr

    def step(self, params, grads):
        return params - self.lr * grads

    def reset(self):
        pass


class MomentumSGD:
    """带动量的 SGD：v = mu*v - lr*g;  w = w + v"""
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.mu = momentum
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.mu * self.v - self.lr * grads
        return params + self.v

    def reset(self):
        self.v = None


class NesterovSGD:
    """Nesterov 加速梯度 (NAG)：先看前方再更新"""
    def __init__(self, lr=0.001, momentum=0.9):
        self.lr = lr
        self.mu = momentum
        self.v = None

    def lookahead(self, params):
        """返回前看点（用于计算梯度）"""
        if self.v is None:
            return params
        return params + self.mu * self.v

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.mu * self.v - self.lr * grads
        return params + self.v

    def reset(self):
        self.v = None


class Adagrad:
    """Adagrad：累积历史梯度平方，自适应学习率（稀疏特征效果好，但学习率单调衰减）"""
    def __init__(self, lr=0.01, eps=1e-8):
        self.lr = lr
        self.eps = eps
        self.G = None

    def step(self, params, grads):
        if self.G is None:
            self.G = np.zeros_like(params)
        self.G += grads ** 2
        return params - self.lr / (np.sqrt(self.G) + self.eps) * grads

    def reset(self):
        self.G = None


class RMSProp:
    """RMSProp：指数加权移动平均梯度平方，解决 Adagrad 学习率消失问题"""
    def __init__(self, lr=0.001, rho=0.9, eps=1e-8):
        self.lr = lr
        self.rho = rho
        self.eps = eps
        self.v = None

    def step(self, params, grads):
        if self.v is None:
            self.v = np.zeros_like(params)
        self.v = self.rho * self.v + (1 - self.rho) * grads ** 2
        return params - self.lr / (np.sqrt(self.v) + self.eps) * grads

    def reset(self):
        self.v = None


# ─────────────────────── 优化实验 ────────────────────────────────

def run_optimization(optimizers, grad_fn, x0, n_steps=500, nesterov_grad_fn=None):
    """在给定梯度函数上运行各优化器，记录轨迹"""
    histories = {}
    for name, opt in optimizers.items():
        opt.reset()
        x = x0.copy().astype(float)
        traj = [x.copy()]
        for _ in range(n_steps):
            if name == "Nesterov" and nesterov_grad_fn is not None:
                x_look = opt.lookahead(x)
                g = nesterov_grad_fn(x_look)
            else:
                g = grad_fn(x)
            x = opt.step(x, g)
            traj.append(x.copy())
        histories[name] = np.array(traj)
    return histories


def sgd_variants():
    print("SGD 变体优化算法运行中...\n")

    # ── 实验1：Rosenbrock 函数 ────────────────────────────────────
    print("1. Rosenbrock 函数优化对比...")

    def rosen_grad(xy):
        return rosenbrock_grad(xy[0], xy[1])

    def rosen_grad_nag(xy):
        return rosenbrock_grad(xy[0], xy[1])

    x0_rosen = np.array([-1.5, 1.0])
    optimizers_rosen = {
        "SGD":       VanillaSGD(lr=0.0008),
        "Momentum":  MomentumSGD(lr=0.0008, momentum=0.9),
        "Nesterov":  NesterovSGD(lr=0.0008, momentum=0.9),
        "Adagrad":   Adagrad(lr=0.4),
        "RMSProp":   RMSProp(lr=0.008),
    }
    hist_rosen = run_optimization(
        optimizers_rosen, rosen_grad, x0_rosen,
        n_steps=3000, nesterov_grad_fn=rosen_grad_nag
    )

    # ── 实验2：二次函数 (病态 Hessian) ───────────────────────────
    print("2. 病态二次函数优化对比...")
    np.random.seed(42)
    # 构造条件数 = 100 的矩阵
    A = np.diag([100.0, 1.0])
    b_vec = np.array([10.0, 1.0])
    w_star = np.linalg.solve(A, b_vec)  # 真实最优解

    def quad_grad(w):
        return quadratic_grad(w, A, b_vec)

    x0_quad = np.array([-0.5, -0.5])
    optimizers_quad = {
        "SGD":      VanillaSGD(lr=0.005),
        "Momentum": MomentumSGD(lr=0.005, momentum=0.9),
        "Nesterov": NesterovSGD(lr=0.005, momentum=0.9),
        "Adagrad":  Adagrad(lr=0.5),
        "RMSProp":  RMSProp(lr=0.05),
    }
    hist_quad = run_optimization(
        optimizers_quad, quad_grad, x0_quad, n_steps=300,
        nesterov_grad_fn=quad_grad
    )

    # ── 计算收敛曲线 ──────────────────────────────────────────────
    rosen_losses = {}
    for name, traj in hist_rosen.items():
        rosen_losses[name] = [rosenbrock(p[0], p[1]) for p in traj]

    quad_losses = {}
    for name, traj in hist_quad.items():
        quad_losses[name] = [quadratic(p, A, b_vec) for p in traj]

    # ── 可视化 ────────────────────────────────────────────────────
    print("3. 生成可视化图表...")
    COLORS = {
        "SGD": "#e74c3c", "Momentum": "#3498db",
        "Nesterov": "#2ecc71", "Adagrad": "#f39c12", "RMSProp": "#9b59b6"
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：Rosenbrock 等高线 + 轨迹 ─────────────────────────
    ax = axes[0, 0]
    xs = np.linspace(-2, 1.5, 300)
    ys = np.linspace(-0.5, 1.5, 300)
    X, Y = np.meshgrid(xs, ys)
    Z = rosenbrock(X, Y)
    ax.contourf(X, Y, np.log1p(Z), levels=40, cmap="plasma", alpha=0.7)
    ax.contour(X, Y, np.log1p(Z), levels=20, colors="white", alpha=0.2, linewidths=0.5)
    for name, traj in hist_rosen.items():
        # 仅显示前500步轨迹，避免过密
        n = min(500, len(traj))
        ax.plot(traj[:n, 0], traj[:n, 1], color=COLORS[name],
                alpha=0.8, linewidth=1.2, label=name)
    ax.plot(1, 1, "w*", markersize=12, label="Global Min")
    ax.plot(x0_rosen[0], x0_rosen[1], "wo", markersize=6)
    ax.set_title("Rosenbrock: 优化轨迹", color="white", pad=10)
    ax.set_xlabel("x", color="#aaa"); ax.set_ylabel("y", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=7, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图2：Rosenbrock 损失曲线 ───────────────────────────────
    ax = axes[0, 1]
    for name, losses in rosen_losses.items():
        ax.semilogy(losses, color=COLORS[name], label=name, alpha=0.9, linewidth=1.5)
    ax.set_title("Rosenbrock: 收敛曲线 (log)", color="white", pad=10)
    ax.set_xlabel("迭代步数", color="#aaa"); ax.set_ylabel("Loss (log)", color="#aaa")
    ax.tick_params(colors="gray"); ax.grid(alpha=0.2, color="#555")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图3：Rosenbrock 最终 Loss 柱状图 ──────────────────────
    ax = axes[0, 2]
    final_losses = {n: v[-1] for n, v in rosen_losses.items()}
    names = list(final_losses.keys())
    vals  = [final_losses[n] for n in names]
    bars = ax.bar(names, vals, color=[COLORS[n] for n in names], alpha=0.85, edgecolor="#333")
    ax.set_title("Rosenbrock: 最终 Loss", color="white", pad=10)
    ax.set_ylabel("Final Loss", color="#aaa")
    ax.set_yscale("log")
    ax.tick_params(colors="gray", axis="x", rotation=20)
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.2,
                f"{v:.2f}", ha="center", va="bottom", color="white", fontsize=8)

    # ── 子图4：二次函数等高线 + 轨迹 ────────────────────────────
    ax = axes[1, 0]
    ws = np.linspace(-0.6, 0.15, 200)
    bs = np.linspace(-0.6, 1.1, 200)
    WW, BB = np.meshgrid(ws, bs)
    ZZ = 0.5 * (100 * WW**2 + BB**2) - 10*WW - BB
    ax.contourf(WW, BB, ZZ, levels=40, cmap="coolwarm", alpha=0.7)
    for name, traj in hist_quad.items():
        ax.plot(traj[:, 0], traj[:, 1], color=COLORS[name],
                alpha=0.85, linewidth=1.2, label=name)
    ax.plot(w_star[0], w_star[1], "w*", markersize=12)
    ax.set_title("病态二次函数: 优化轨迹", color="white", pad=10)
    ax.set_xlabel("w0", color="#aaa"); ax.set_ylabel("w1", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=7, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图5：二次函数损失曲线 ──────────────────────────────────
    ax = axes[1, 1]
    for name, losses in quad_losses.items():
        ax.semilogy(losses, color=COLORS[name], label=name, alpha=0.9, linewidth=1.5)
    ax.set_title("病态二次函数: 收敛曲线 (log)", color="white", pad=10)
    ax.set_xlabel("迭代步数", color="#aaa"); ax.set_ylabel("Loss (log)", color="#aaa")
    ax.tick_params(colors="gray"); ax.grid(alpha=0.2, color="#555")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white", framealpha=0.7)

    # ── 子图6：算法特性对比表 ────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["算法", "动量", "自适应LR", "收敛速度", "超参数"],
        ["Vanilla SGD", "否", "否", "慢", "lr"],
        ["Momentum", "是", "否", "中", "lr, mu"],
        ["Nesterov", "是(NAG)", "否", "较快", "lr, mu"],
        ["Adagrad", "否", "是(累积)", "中(衰减)", "lr"],
        ["RMSProp", "否", "是(EMA)", "快", "lr, rho"],
    ]
    col_widths = [0.25, 0.15, 0.2, 0.2, 0.2]
    tbl = ax.table(
        cellText=table_data[1:], colLabels=table_data[0],
        cellLoc="center", loc="center",
        colWidths=col_widths
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white")
        cell.set_edgecolor("#334")
    ax.set_title("算法特性总结", color="white", pad=10)

    plt.suptitle("SGD 变体优化算法对比", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("sgd_variants.png"))

    # ── 打印数值结果 ──────────────────────────────────────────────
    print("\n=== Rosenbrock 最终损失（3000步） ===")
    for name, losses in sorted(rosen_losses.items(), key=lambda x: x[1][-1]):
        print(f"  {name:<12}: {losses[-1]:.4f}")

    print("\n=== 二次函数最终损失（300步） ===")
    for name, losses in sorted(quad_losses.items(), key=lambda x: x[1][-1]):
        print(f"  {name:<12}: {losses[-1]:.6f}")

    print("\n[DONE] SGD 变体优化算法完成!")


if __name__ == "__main__":
    sgd_variants()
