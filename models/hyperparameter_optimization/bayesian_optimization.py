"""
贝叶斯优化 (Bayesian Optimization)
===================================
智能超参数搜索：用概率模型建模目标函数，平衡探索与利用

核心思想：
  1. 用代理模型（如高斯过程）建模目标函数 f(x)
  2. 使用采集函数（如 Expected Improvement）选择下一个采样点
  3. 迭代优化，用少量评估找到全局最优

实现内容：
  1. 高斯过程 (Gaussian Process) 回归
  2. Expected Improvement (EI) 采集函数
  3. 优化合成测试函数（Rosenbrock/Ackley）
  4. 对比随机搜索、网格搜索、贝叶斯优化
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 高斯过程 ─────────────────────────────

class GaussianProcess:
    """
    简化高斯过程回归

    核函数：RBF k(x, x') = σ² exp(-||x-x'||²/(2l²))
    先验：GP(0, K)
    后验：GP(μ*, Σ*)
    """
    def __init__(self, kernel="rbf", alpha=1e-8):
        self.kernel = kernel
        self.alpha = alpha  # 噪声方差

    def rbf_kernel(self, X1, X2, length_scale=1.0, sigma_f=1.0):
        """RBF 核函数"""
        dists = cdist(X1, X2, metric="sqeuclidean")
        K = sigma_f**2 * np.exp(-0.5 / length_scale**2 * dists)
        return K

    def fit(self, X, y):
        """训练 GP：计算核矩阵 K"""
        self.X = X
        self.y = y
        self.K = self.rbf_kernel(X, X) + self.alpha * np.eye(len(X))
        self.L = np.linalg.cholesky(self.K)  # Cholesky 分解

    def predict(self, X_new):
        """预测：后验均值和方差"""
        K_s = self.rbf_kernel(self.X, X_new)
        K_ss = self.rbf_kernel(X_new, X_new) + self.alpha * np.eye(len(X_new))
        K_inv = np.linalg.solve(self.L.T, np.linalg.solve(self.L, np.eye(len(self.X))))

        # 后验均值
        mu = K_s.T @ K_inv @ self.y
        # 后验协方差
        cov = K_ss - K_s.T @ K_inv @ K_s
        return mu, np.diag(cov)


# ─────────────────────── 采集函数 ─────────────────────────────

def expected_improvement(mu, sigma, best_y):
    """
    Expected Improvement (EI)

    EI(x) = E[max(f(x) - f_best, 0)]
    """
    if sigma < 1e-8:
        return 0
    z = (best_y - mu) / sigma
    from scipy.stats import norm
    return sigma * (z * norm.cdf(z) + norm.pdf(z))


# ─────────────────────── 贝叶斯优化 ─────────────────────────

def bayesian_optimization(objective_func, bounds, n_init=5, n_iter=20, random_state=42):
    """
    贝叶斯优化

    参数：
      objective_func: 目标函数 f(x) -> y
      bounds:        变量范围 [(min1, max1), (min2, max2), ...]
      n_init:        初始随机采样数
      n_iter:        迭代次数
    """
    np.random.seed(random_state)
    n_dim = len(bounds)

    # 初始随机采样
    X_init = np.random.rand(n_init, n_dim)
    for i in range(n_dim):
        X_init[:, i] = X_init[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

    y_init = np.array([objective_func(x) for x in X_init])

    X = X_init.copy()
    y = y_init.copy()

    gp = GaussianProcess()
    history = {"X": [X.copy()], "y": [y.copy()]}

    for t in range(n_iter):
        # 1. 拟合 GP
        gp.fit(X, y)

        # 2. 找下一个采样点（最大化 EI）
        best_y = y.min()  # 最小化问题
        best_x = X[np.argmin(y)]

        # 随机候选 + 精确优化
        candidates = np.random.rand(100, n_dim)
        for i in range(n_dim):
            candidates[:, i] = candidates[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]

        mu, sigma = gp.predict(candidates)
        ei = expected_improvement(mu, sigma, best_y)
        next_idx = np.argmax(ei)
        x_next = candidates[next_idx]

        # 3. 评估目标函数
        y_next = objective_func(x_next)

        # 4. 更新数据
        X = np.vstack([X, x_next])
        y = np.append(y, y_next)
        history["X"].append(X.copy())
        history["y"].append(y.copy())

        print(f"  Iter {t}: x={x_next}, y={y_next:.4f}, best_y={y.min():.4f}")

    return X, y, history


# ─────────────────────── 测试函数 ─────────────────────────────

def rosenbrock(x):
    """Rosenbrock 函数（全局最小值在 (1,1)，值=0）"""
    return (1 - x[0])**2 + 100 * (x[1] - x[0]**2)**2

def ackley(x):
    """Ackley 函数（全局最小值在原点，值≈0）"""
    a, b, c = 20, 0.2, 2*np.pi
    sum1 = np.sum(x**2)
    sum2 = np.sum(np.cos(c * x))
    return -a * np.exp(-b * np.sqrt(sum1 / len(x))) - np.exp(sum2 / len(x)) + a + np.exp(1)


# ─────────────────────── 对比其他优化方法 ─────────────────────

def random_search(objective_func, bounds, n_iter, random_state=42):
    """随机搜索"""
    np.random.seed(random_state)
    n_dim = len(bounds)
    X = np.random.rand(n_iter, n_dim)
    for i in range(n_dim):
        X[:, i] = X[:, i] * (bounds[i][1] - bounds[i][0]) + bounds[i][0]
    y = np.array([objective_func(x) for x in X])
    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx]


def grid_search(objective_func, bounds, n_points=20):
    """网格搜索（简化版）"""
    n_dim = len(bounds)
    grids = []
    for i in range(n_dim):
        grids.append(np.linspace(bounds[i][0], bounds[i][1], n_points))
    mesh = np.meshgrid(*grids)
    X = np.column_stack([m.ravel() for m in mesh])
    y = np.array([objective_func(x) for x in X])
    best_idx = np.argmin(y)
    return X[best_idx], y[best_idx], X, y


def bayesian_optimization_main():
    print("贝叶斯优化运行中...\n")

    # ── 目标函数：Rosenbrock ─────────────────────────────────────────
    print("1. 优化 Rosenbrock 函数...")
    bounds_rosen = [(-2, 2), (-2, 2)]

    print("   随机搜索 (50次)...")
    x_rand, y_rand = random_search(rosenbrock, bounds_rosen, n_iter=50)
    print(f"     最优: x={x_rand}, y={y_rand:.4f}")

    print("   贝叶斯优化 (5 init + 20 iter)...")
    X_bo, y_bo, hist_bo = bayesian_optimization(
        rosenbrock, bounds_rosen, n_init=5, n_iter=20
    )
    best_idx = np.argmin(y_bo)
    print(f"     最优: x={X_bo[best_idx]}, y={y_bo[best_idx]:.4f}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n2. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── Rosenbrock 等高线 ────────────────────────────────────────
    ax = axes[0, 0]
    xs = np.linspace(-2, 2, 200)
    ys = np.linspace(-2, 2, 200)
    X, Y = np.meshgrid(xs, ys)
    Z = rosenbrock(np.stack([X, Y], axis=-1).reshape(-1, 2)).reshape(X.shape)
    ax.contourf(X, Y, np.log1p(Z), levels=40, cmap="plasma", alpha=0.7)
    ax.contour(X, Y, np.log1p(Z), levels=20, colors="white", alpha=0.3, linewidths=0.5)
    # 贝叶斯优化采样点
    ax.scatter(X_bo[:, 0], X_bo[:, 1], c="#e74c3c", s=40, alpha=0.8, edgecolors="white", linewidths=0.5)
    ax.plot(1, 1, "w*", markersize=14, label="全局最优")
    ax.set_title("Rosenbrock 等高线 + 贝叶斯采样", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 最优值收敛曲线 ─────────────────────────────────────────
    ax = axes[0, 1]
    best_y_iter = np.minimum.accumulate(y_bo)
    ax.plot(best_y_iter, color="#e74c3c", linewidth=2, label="贝叶斯优化")
    ax.plot(np.cumsum(np.ones(len(y_bo)))[:50],
             np.minimum.accumulate([y_rand] * 50), color="#3498db", linestyle="--", label="随机搜索")
    ax.set_title("最优值收敛曲线", color="white", pad=8)
    ax.set_xlabel("迭代次数", color="#aaa"); ax.set_ylabel("Best y", color="#aaa")
    ax.set_yscale("log")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.grid(alpha=0.2, color="#555")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 采样点热力图（GP 后验） ───────────────────────────────────
    ax = axes[0, 2]
    gp = GaussianProcess()
    gp.fit(X_bo, y_bo)
    grid_x = np.linspace(-2, 2, 50)
    grid_y = np.linspace(-2, 2, 50)
    X_grid, Y_grid = np.meshgrid(grid_x, grid_y)
    test_pts = np.column_stack([X_grid.ravel(), Y_grid.ravel()])
    mu, sigma = gp.predict(test_pts)
    Z_mean = mu.reshape(X_grid.shape)
    im = ax.contourf(X_grid, Y_grid, Z_mean, levels=20, cmap="coolwarm", alpha=0.7)
    ax.scatter(X_bo[:, 0], X_bo[:, 1], c="#e74c3c", s=40, edgecolors="white", linewidths=0.5)
    ax.set_title("GP 后验均值", color="white", pad=8)
    ax.set_xlabel("x1", color="#aaa"); ax.set_ylabel("x2", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im, ax=ax, fraction=0.02).ax.tick_params(colors="gray")

    # ── Ackley 函数优化对比 ─────────────────────────────────────
    print("\n3. 优化 Ackley 函数...")
    bounds_ack = [-5, 5]

    print("   随机搜索...")
    x_rand_ack, y_rand_ack = random_search(
        lambda x: ackley(np.array([x, x])),
        [(bounds_ack, bounds_ack)], n_iter=50
    )

    print("   贝叶斯优化...")
    X_bo_ack, y_bo_ack, _ = bayesian_optimization(
        lambda x: ackley(np.array([x[0], x[0]])),
        [(bounds_ack, bounds_ack)], n_init=5, n_iter=20
    )

    ax = axes[1, 0]
    methods = ["Random Search", "Bayesian Opt"]
    vals = [y_rand_ack, y_bo_ack.min()]
    pal = ["#3498db", "#e74c3c"]
    bars = ax.bar(methods, vals, color=pal, alpha=0.85)
    ax.set_title("Ackley 函数优化结果对比", color="white", pad=8)
    ax.set_ylabel("Best y (越小越好)", color="#aaa")
    ax.set_yscale("log")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v * 1.1,
                f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=9)

    # ── 采集函数 EI 示例 ─────────────────────────────────────────
    ax = axes[1, 1]
    test_x = np.linspace(-2, 2, 100)
    mu_test, sigma_test = gp.predict(np.column_stack([test_x, test_x]))
    best_y_hist = y_bo.min()
    ei_test = expected_improvement(mu_test, sigma_test, best_y_hist)
    ax.plot(test_x, mu_test, color="#3498db", label="GP Mean", linewidth=2)
    ax.fill_between(test_x, mu_test - 2*sigma_test, mu_test + 2*sigma_test,
                    color="#3498db", alpha=0.2)
    ax.plot(test_x, ei_test, color="#e74c3c", label="EI", linewidth=2)
    ax.set_title("GP 预测 & EI 采集函数", color="white", pad=8)
    ax.set_xlabel("x", color="#aaa"); ax.set_ylabel("Value", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.grid(alpha=0.2, color="#555")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 方法对比表 ─────────────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["方法", "需要梯度", "评估次数", "适用场景"],
        ["随机搜索", "否", "中", "实现简单"],
        ["网格搜索", "否", "多(指数级)", "参数少"],
        ["贝叶斯优化", "否", "少", "昂贵黑盒"],
        ["梯度下降", "是", "少", "光滑可导"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   colWidths=[0.2, 0.18, 0.18, 0.44])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white"); cell.set_edgecolor("#334")
    ax.set_title("优化方法对比总结", color="white", pad=10)

    plt.suptitle("贝叶斯优化 (Gaussian Process + EI)", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("bayesian_optimization.png"))

    print("\n[DONE] 贝叶斯优化完成!")


if __name__ == "__main__":
    bayesian_optimization_main()
