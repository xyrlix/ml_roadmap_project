# EM 算法（Expectation-Maximization）
# 通过迭代优化求解含有隐变量的最大似然估计

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from utils import get_results_path, save_and_close


class EMAlgorithm:
    """从零实现 EM 算法（单维高斯混合）"""
    def __init__(self, n_components=2, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None      # π_k
        self.means_ = None        # μ_k
        self.variances_ = None     # σ²_k
        self.responsibilities_ = None  # γ(z_ik)

    def _init_params(self, X):
        """初始化参数（K-Means）"""
        kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = np.array([X[labels == k].mean() for k in range(self.n_components)])
        self.variances_ = np.array([X[labels == k].var() for k in range(self.n_components)])

    def _e_step(self, X):
        """E 步：计算后验概率（责任度）"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            # N(x | μ_k, σ²_k)
            pdf = (1.0 / np.sqrt(2 * np.pi * self.variances_[k])) * \
                   np.exp(-0.5 * (X - self.means_[k])**2 / self.variances_[k])
            responsibilities[:, k] = self.weights_[k] * pdf

        # 归一化
        responsibilities = responsibilities / (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
        return responsibilities

    def _m_step(self, X, responsibilities):
        """M 步：最大化期望完全对数似然"""
        N_k = responsibilities.sum(axis=0)

        # π_k = Σ γ(z_ik) / N
        self.weights_ = N_k / len(X)

        # μ_k = Σ γ(z_ik) x_i / Σ γ(z_ik)
        self.means_ = (responsibilities.T @ X) / (N_k[:, np.newaxis])

        # σ²_k = Σ γ(z_ik) (x_i - μ_k)² / Σ γ(z_ik)
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.variances_[k] = (responsibilities[:, k] * diff**2).sum() / N_k[k]

    def fit(self, X):
        """EM 迭代"""
        self._init_params(X)
        log_likelihoods = []

        for iteration in range(self.max_iter):
            # E 步
            self.responsibilities_ = self._e_step(X)

            # M 步
            old_means = self.means_.copy()
            self._m_step(X, self.responsibilities_)

            # 计算对数似然
            log_likelihood = self._compute_log_likelihood(X)
            log_likelihoods.append(log_likelihood)

            # 收敛判断
            if np.max(np.abs(self.means_ - old_means)) < self.tol:
                break

        return self, log_likelihoods

    def _compute_log_likelihood(self, X):
        """计算对数似然"""
        n_samples = X.shape[0]
        log_likelihood = 0

        for i in range(n_samples):
            sample_ll = 0
            for k in range(self.n_components):
                pdf = (1.0 / np.sqrt(2 * np.pi * self.variances_[k])) * \
                       np.exp(-0.5 * (X[i] - self.means_[k])**2 / self.variances_[k])
                sample_ll += self.weights_[k] * pdf
            log_likelihood += np.log(sample_ll + 1e-10)

        return log_likelihood / n_samples

    def predict(self, X):
        """硬聚类（argmax 责任度）"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)


def em_algorithm():
    """EM 算法实现（单维高斯混合）"""
    print("EM 算法（期望最大化）运行中...\n")

    # 1. 数据准备（单维混合高斯）
    print("1. 生成单维混合高斯数据（2个高斯分布）...")
    np.random.seed(42)
    n_samples = 300

    # 混合高斯
    X1 = np.random.normal(loc=-2, scale=0.8, size=int(n_samples * 0.7))
    X2 = np.random.normal(loc=3, scale=1.2, size=int(n_samples * 0.3))
    X = np.concatenate([X1, X2]).reshape(-1, 1)

    # 打乱
    idx = np.random.permutation(len(X))
    X = X[idx]

    print(f"   样本数: {len(X)}")

    # 2. EM 训练
    print("2. 训练 EM 算法...")
    em = EMAlgorithm(n_components=2, max_iter=200, tol=1e-5)
    em, log_likelihoods = em.fit(X)

    # 输出参数
    print("3. 学习到的参数：")
    for k in range(em.n_components):
        print(f"   Component {k}: π={em.weights_[k]:.4f}, μ={em.means_[k]:.4f}, σ²={em.variances_[k]:.4f}")

    # 3. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：对数似然收敛曲线 ──
    ax = axes[0]
    ax.plot(log_likelihoods, 'bo-', linewidth=1.5, markersize=6)
    ax.set_title('EM 算法：对数似然收敛曲线')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Likelihood')
    ax.grid(True, alpha=0.3)

    # ── 子图2：数据分布 + 拟合高斯 ──
    ax = axes[1]
    ax.scatter(X, np.zeros_like(X), c='steelblue', alpha=0.6, s=30, label='Data')

    x_plot = np.linspace(X.min() - 1, X.max() + 1, 500).reshape(-1, 1)

    for k in range(em.n_components):
        # 绘制拟合的高斯
        pdf = em.weights_[k] * (1.0 / np.sqrt(2 * np.pi * em.variances_[k])) * \
               np.exp(-0.5 * (x_plot - em.means_[k])**2 / em.variances_[k])
        ax.plot(x_plot, pdf, linewidth=2,
                label=f'GMM {k}: μ={em.means_[k][0]:.2f}, σ²={em.variances_[k][0]:.2f}')

    ax.set_title('EM 拟合的高斯混合分布')
    ax.set_xlabel('x')
    ax.set_yticks([])
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图3：责任度热力图（前50个样本）──
    ax = axes[2]
    n_show = min(50, len(X))
    resp_subset = em.responsibilities_[:n_show]

    im = ax.imshow(resp_subset.T, aspect='auto', cmap='RdYlGn', vmin=0, vmax=1)
    fig.colorbar(im, ax=ax)
    ax.set_title(f'责任度矩阵（前{n_show}个样本）')
    ax.set_xlabel('Sample Index')
    ax.set_yticks(range(em.n_components))
    ax.set_yticklabels([f'Component {k}' for k in range(em.n_components)])

    save_path = get_results_path('em_algorithm_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    em_algorithm()
