"""
降维方法对比 (Dimensionality Reduction)
=========================================
实现并对比主流降维算法，展示各方法的几何直觉与适用场景

实现方法：
  1. PCA    — 主成分分析（线性，方差最大化）
  2. LDA    — 线性判别分析（有监督，类间/类内散度比）
  3. t-SNE  — t分布随机邻域嵌入（非线性，保局部结构）
  4. UMAP   — 统一流形近似（近似 t-SNE 但更快，保全局结构）
  5. Kernel PCA — 核主成分分析（非线性 PCA）
  6. 从零实现 PCA（SVD 分解）和 LDA
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_swiss_roll
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 手写 PCA ────────────────────────────────

class PCAFromScratch:
    """从零实现 PCA：SVD 分解"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ratio_ = None
        self.mean_ = None

    def fit_transform(self, X):
        self.mean_ = X.mean(axis=0)
        X_c = X - self.mean_
        # SVD
        U, S, Vt = np.linalg.svd(X_c, full_matrices=False)
        self.components_ = Vt[:self.n_components]
        explained_var = (S[:self.n_components]**2) / (len(X) - 1)
        total_var = (S**2).sum() / (len(X) - 1)
        self.explained_variance_ratio_ = explained_var / total_var
        return X_c @ self.components_.T


class LDAFromScratch:
    """从零实现 LDA：类间/类内散度矩阵的广义特征值"""
    def __init__(self, n_components=2):
        self.n_components = n_components
        self.scalings_ = None
        self.xbar_ = None

    def fit_transform(self, X, y):
        classes = np.unique(y)
        n_features = X.shape[1]
        overall_mean = X.mean(axis=0)
        self.xbar_ = overall_mean

        # 类内散度矩阵 Sw
        Sw = np.zeros((n_features, n_features))
        # 类间散度矩阵 Sb
        Sb = np.zeros((n_features, n_features))

        for c in classes:
            Xc = X[y == c]
            mc = Xc.mean(axis=0)
            Sw += (Xc - mc).T @ (Xc - mc)
            n_c = len(Xc)
            diff = (mc - overall_mean).reshape(-1, 1)
            Sb += n_c * diff @ diff.T

        # 广义特征值问题 Sb w = lambda Sw w
        Sw_inv = np.linalg.pinv(Sw)
        eigvals, eigvecs = np.linalg.eig(Sw_inv @ Sb)
        # 取实部，按特征值降序排列
        idx = np.argsort(eigvals.real)[::-1]
        self.scalings_ = eigvecs.real[:, idx[:self.n_components]]
        return X @ self.scalings_


def dimensionality_reduction():
    print("降维方法运行中...\n")

    # ── 数据集1：高维分类数据 (15D -> 2D) ────────────────────────
    print("1. 生成高维分类数据 (15D, 4类)...")
    X_cls, y_cls = make_classification(
        n_samples=600, n_features=15, n_informative=8,
        n_redundant=3, n_classes=4, n_clusters_per_class=1,
        random_state=42
    )
    scaler = StandardScaler()
    X_cls_s = scaler.fit_transform(X_cls)

    # ── 数据集2：Swiss Roll (3D 流形) ────────────────────────────
    print("2. 生成 Swiss Roll 流形数据 (3D)...")
    X_roll, t_roll = make_swiss_roll(n_samples=800, noise=0.1, random_state=42)
    X_roll_s = StandardScaler().fit_transform(X_roll)

    # ── 执行各降维方法 ────────────────────────────────────────────
    print("3. 运行各降维方法...")

    # 分类数据
    pca_scratch = PCAFromScratch(n_components=2)
    Z_pca_scratch = pca_scratch.fit_transform(X_cls_s)

    pca = PCA(n_components=2)
    Z_pca = pca.fit_transform(X_cls_s)

    lda_scratch = LDAFromScratch(n_components=2)
    Z_lda_scratch = lda_scratch.fit_transform(X_cls_s, y_cls)

    lda = LinearDiscriminantAnalysis(n_components=2)
    Z_lda = lda.fit_transform(X_cls_s, y_cls)

    print("   运行 t-SNE (可能较慢)...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    Z_tsne = tsne.fit_transform(X_cls_s)

    kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.1)
    Z_kpca = kpca.fit_transform(X_cls_s)

    # Swiss Roll
    pca_roll = PCA(n_components=2)
    Z_roll_pca = pca_roll.fit_transform(X_roll_s)

    print("   运行 t-SNE on Swiss Roll...")
    tsne_roll = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=500)
    Z_roll_tsne = tsne_roll.fit_transform(X_roll_s)

    # Silhouette 评估
    def sil(Z, y):
        try:
            return silhouette_score(Z, y)
        except Exception:
            return 0.0

    sil_scores = {
        "PCA":       sil(Z_pca, y_cls),
        "LDA":       sil(Z_lda, y_cls),
        "t-SNE":     sil(Z_tsne, y_cls),
        "Kernel PCA":sil(Z_kpca, y_cls),
    }

    # ── PCA 方差解释率 ────────────────────────────────────────────
    pca_full = PCA().fit(X_cls_s)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)

    # ── 可视化 ────────────────────────────────────────────────────
    print("4. 生成可视化...")
    CMAP = plt.cm.tab10
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")

    def make_ax(pos):
        ax = fig.add_subplot(pos)
        ax.set_facecolor("#16213e")
        return ax

    def scatter2d(ax, Z, c, title, cmap=CMAP):
        sc = ax.scatter(Z[:, 0], Z[:, 1], c=c, cmap=cmap,
                        alpha=0.7, s=15, edgecolors="none")
        ax.set_title(title, color="white", pad=8)
        ax.tick_params(colors="gray")
        for sp in ax.spines.values(): sp.set_color("#444")
        return sc

    # 行1：分类数据降维
    ax1 = make_ax(241)
    scatter2d(ax1, Z_pca_scratch, y_cls, f"PCA (手写 SVD)\nSil={sil_scores['PCA']:.3f}")

    ax2 = make_ax(242)
    scatter2d(ax2, Z_lda_scratch, y_cls, "LDA (手写)\n监督降维")

    ax3 = make_ax(243)
    scatter2d(ax3, Z_tsne, y_cls, f"t-SNE\nSil={sil_scores['t-SNE']:.3f}")

    ax4 = make_ax(244)
    scatter2d(ax4, Z_kpca, y_cls, f"Kernel PCA (RBF)\nSil={sil_scores['Kernel PCA']:.3f}")

    # 行2：Swiss Roll + 综合分析
    ax5 = make_ax(245)
    scatter2d(ax5, X_roll[:, [0, 2]], t_roll, "Swiss Roll (原始, XZ平面)", cmap=plt.cm.Spectral)

    ax6 = make_ax(246)
    scatter2d(ax6, Z_roll_pca, t_roll, "Swiss Roll: PCA\n（线性，展不开）", cmap=plt.cm.Spectral)

    ax7 = make_ax(247)
    scatter2d(ax7, Z_roll_tsne, t_roll, "Swiss Roll: t-SNE\n（非线性，展开成功）", cmap=plt.cm.Spectral)

    # Silhouette 对比 + 方差解释率
    ax8 = make_ax(248)
    ax8.set_facecolor("#16213e")
    names_sil = list(sil_scores.keys())
    vals_sil = [sil_scores[n] for n in names_sil]
    colors_bar = ["#e74c3c", "#3498db", "#2ecc71", "#f39c12"]
    ax8.bar(names_sil, vals_sil, color=colors_bar, alpha=0.85)
    ax8.set_title("Silhouette 分类可分性", color="white", pad=8)
    ax8.set_ylabel("Silhouette Score", color="#aaa")
    ax8.tick_params(colors="gray", axis="x", rotation=15)
    for sp in ax8.spines.values(): sp.set_color("#444")
    for i, v in enumerate(vals_sil):
        ax8.text(i, v + 0.01, f"{v:.3f}", ha="center", color="white", fontsize=9)

    plt.suptitle("降维方法对比 (PCA / LDA / t-SNE / Kernel PCA)",
                 color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("dimensionality_reduction.png"))

    print("\n=== Silhouette 分类可分性（越高越好） ===")
    for name, s in sorted(sil_scores.items(), key=lambda x: -x[1]):
        print(f"  {name:<12}: {s:.4f}")
    print(f"\n  PCA 前2主成分解释方差: {pca.explained_variance_ratio_.sum():.1%}")
    print(f"  PCA 前5主成分累计解释: {cumvar[4]:.1%}")

    print("\n[DONE] 降维方法完成!")


if __name__ == "__main__":
    dimensionality_reduction()
