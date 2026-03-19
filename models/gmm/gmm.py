# 高斯混合模型 (Gaussian Mixture Model)
# 概率聚类模型，使用 EM 算法学习参数

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from utils import get_results_path, save_and_close


class GMMFromScratch:
    """从零实现高斯混合模型（多维）"""
    def __init__(self, n_components=3, max_iter=100, tol=1e-4):
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None
        self.means_ = None
        self.covariances_ = None

    def _init_params(self, X):
        """初始化参数"""
        kmeans = KMeans(n_clusters=self.n_components, n_init=10, random_state=42)
        labels = kmeans.fit_predict(X)

        self.weights_ = np.ones(self.n_components) / self.n_components
        self.means_ = np.array([X[labels == k].mean(axis=0) for k in range(self.n_components)])

        # 协方差初始化（对角阵）
        self.covariances_ = np.array([np.cov(X[labels == k], rowvar=False) + 1e-6 * np.eye(X.shape[1])
                                        for k in range(self.n_components)])

    def _multivariate_gaussian_pdf(self, X, mean, cov):
        """多元高斯 PDF"""
        n = X.shape[1]
        cov_inv = np.linalg.pinv(cov)
        det = np.linalg.det(cov) + 1e-10

        diff = X - mean
        mahalanobis = np.sum(diff @ cov_inv * diff, axis=1)

        return (1.0 / ((2 * np.pi) ** (n/2) * np.sqrt(det))) * np.exp(-0.5 * mahalanobis)

    def _e_step(self, X):
        """E 步：计算责任度"""
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):
            pdf = self._multivariate_gaussian_pdf(X, self.means_[k], self.covariances_[k])
            responsibilities[:, k] = self.weights_[k] * pdf

        # 归一化
        responsibilities = responsibilities / (responsibilities.sum(axis=1, keepdims=True) + 1e-10)
        return responsibilities

    def _m_step(self, X, responsibilities):
        """M 步：更新参数"""
        N_k = responsibilities.sum(axis=0)

        # π_k
        self.weights_ = N_k / len(X)

        # μ_k
        self.means_ = (responsibilities.T @ X) / (N_k[:, np.newaxis])

        # Σ_k
        for k in range(self.n_components):
            diff = X - self.means_[k]
            weighted_cov = (responsibilities[:, k:k+1] * diff).T @ diff
            self.covariances_[k] = weighted_cov / N_k[k] + 1e-6 * np.eye(X.shape[1])

    def fit(self, X):
        """训练 GMM"""
        self._init_params(X)

        for iteration in range(self.max_iter):
            old_means = self.means_.copy()

            # E 步
            responsibilities = self._e_step(X)

            # M 步
            self._m_step(X, responsibilities)

            if np.max(np.abs(self.means_ - old_means)) < self.tol:
                break

        return self

    def predict(self, X):
        """硬聚类"""
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

    def predict_proba(self, X):
        """软聚类（责任度）"""
        return self._e_step(X)


def gmm():
    """高斯混合模型实现（软聚类）"""
    print("高斯混合模型 (GMM) 运行中...\n")

    # 1. 数据准备（三簇球形数据）
    print("1. 准备数据（三簇球形数据）...")
    np.random.seed(42)
    n_samples = 400

    X1 = np.random.randn(int(n_samples * 0.4), 2) * 0.5 + np.array([2, 2])
    X2 = np.random.randn(int(n_samples * 0.35), 2) * 0.8 + np.array([-2, -1])
    X3 = np.random.randn(int(n_samples * 0.25), 2) * 0.6 + np.array([0, -3])
    X = np.vstack([X1, X2, X3])

    # 标准化
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    print(f"   样本数: {len(X)}")

    # 2. 训练手写 GMM
    print("2. 训练手写 GMM (K=3)...")
    gmm_scratch = GMMFromScratch(n_components=3, max_iter=200, tol=1e-5)
    gmm_scratch.fit(X)
    labels_scratch = gmm_scratch.predict(X)
    proba_scratch = gmm_scratch.predict_proba(X)

    sil_scratch = silhouette_score(X, labels_scratch)
    print(f"   轮廓系数: {sil_scratch:.4f}")

    # 3. sklearn 对比
    print("3. sklearn GaussianMixture 对比...")
    gmm_sk = GaussianMixture(n_components=3, max_iter=200, tol=1e-5, random_state=42)
    gmm_sk.fit(X)
    labels_sk = gmm_sk.predict(X)
    sil_sk = silhouette_score(X, labels_sk)
    print(f"   轮廓系数: {sil_sk:.4f}")

    # 4. 组件数敏感性分析
    print("4. 组件数敏感性分析（K=2~6）...")
    k_range = range(2, 7)
    sils, bics, aics = [], [], []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, max_iter=200, random_state=42)
        gmm.fit(X)
        labels = gmm.predict(X)
        sils.append(silhouette_score(X, labels) if len(np.unique(labels)) > 1 else 0)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    best_k_sil = k_range[np.argmax(sils)]
    best_k_bic = k_range[np.argmin(bics)]
    print(f"   Silhouette 最优 K = {best_k_sil}")
    print(f"   BIC 最优 K = {best_k_bic}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：软聚类散点图（责任度）──
    ax = axes[0]
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    for k in range(3):
        mask = labels_scratch == k
        ax.scatter(X[mask, 0], X[mask, 1], c=[colors[k]], s=30,
                   alpha=0.6, edgecolors='k', linewidths=0.3, label=f'Cluster {k}')
        # 绘制均值
        ax.scatter(gmm_scratch.means_[k, 0], gmm_scratch.means_[k, 1],
                   c='red', marker='*', s=300, zorder=10)
    ax.set_title(f'GMM 软聚类结果\n(Silhouette={sil_scratch:.3f})')
    ax.set_xlabel('Feature 1 (Scaled)')
    ax.set_ylabel('Feature 2 (Scaled)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图2：K 值 vs 评估指标 ──
    ax = axes[1]
    ax2 = ax.twinx()
    ax.plot(k_range, sils, 'bo-', label='Silhouette', linewidth=1.5)
    ax2.plot(k_range, bics, 'rs--', label='BIC', linewidth=1.5)
    ax2.plot(k_range, aics, 'gs-.', label='AIC', linewidth=1.5)
    ax.set_xlabel('Number of Components (K)')
    ax.set_ylabel('Silhouette Score', color='blue')
    ax2.set_ylabel('BIC / AIC', color='red')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax.grid(True, alpha=0.3)

    # ── 子图3：BIC/AIC 曲线 ──
    ax = axes[2]
    ax.plot(k_range, bics, 'ro-', label='BIC', linewidth=1.5, markersize=6)
    ax.plot(k_range, aics, 'gs-', label='AIC', linewidth=1.5, markersize=6)
    ax.axvline(x=best_k_bic, color='green', linestyle='--', linewidth=2,
               label=f'Best K={best_k_bic}')
    ax.set_title('GMM：BIC / AIC 曲线')
    ax.set_xlabel('Number of Components (K)')
    ax.set_ylabel('Value')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('gmm_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    gmm()
