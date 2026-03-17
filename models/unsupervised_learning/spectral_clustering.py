# 谱聚类模型
# 利用图的拉普拉斯矩阵特征向量进行聚类，擅长处理非凸形状

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import SpectralClustering, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, make_circles
from utils import generate_clustering_data, get_results_path, save_and_close


def spectral_clustering():
    """谱聚类实现：对比非凸数据与 K-Means 的差异"""
    print("谱聚类模型运行中...\n")

    # 1. 数据准备：月牙形、同心圆、球形
    print("1. 准备三种形状数据集...")
    np.random.seed(42)
    X_moons, _ = make_moons(n_samples=200, noise=0.06, random_state=42)
    X_circles, _ = make_circles(n_samples=200, noise=0.04, factor=0.45, random_state=42)
    X_blobs, _ = generate_clustering_data(n_samples=200, n_clusters=3, random_state=42)

    datasets = [
        (X_moons, "月牙形 (n_clusters=2)", 2),
        (X_circles, "同心圆 (n_clusters=2)", 2),
        (X_blobs, "球形数据 (n_clusters=3)", 3),
    ]

    # 2. 谱聚类 vs K-Means 对比
    print("2. 谱聚类 vs K-Means 对比...")
    fig, axes = plt.subplots(3, 2, figsize=(12, 15))

    for row, (X, title, n_c) in enumerate(datasets):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        # K-Means
        km = KMeans(n_clusters=n_c, random_state=42, n_init=10)
        labels_km = km.fit_predict(X_s)
        sil_km = silhouette_score(X_s, labels_km)

        # 谱聚类（RBF 核）
        sc = SpectralClustering(n_clusters=n_c, affinity='rbf',
                                gamma=1.0, random_state=42, n_init=10)
        labels_sc = sc.fit_predict(X_s)
        sil_sc = silhouette_score(X_s, labels_sc)

        print(f"   [{title}]  K-Means 轮廓={sil_km:.3f}  谱聚类 轮廓={sil_sc:.3f}")

        for col, (labels, method, sil) in enumerate([
            (labels_km, 'K-Means', sil_km),
            (labels_sc, 'Spectral', sil_sc)
        ]):
            ax = axes[row, col]
            ax.scatter(X_s[:, 0], X_s[:, 1], c=labels,
                       cmap='Set1', s=30, edgecolors='k',
                       linewidths=0.3, alpha=0.85)
            ax.set_title(f'{method} — {title}\nSilhouette={sil:.3f}')
            ax.set_xlabel('F1 (Scaled)')
            ax.set_ylabel('F2 (Scaled)')
            ax.grid(True, alpha=0.3)

    plt.suptitle('谱聚类 vs K-Means：非凸形状数据对比', fontsize=14, y=1.01)
    save_path = get_results_path('spectral_clustering_result.png')
    save_and_close(save_path)
    print(f"\n   图表已保存: {save_path}")

    # 3. 不同 affinity 核函数对比（月牙形数据）
    print("3. 不同相似度核函数对比（月牙形）...")
    X_s = StandardScaler().fit_transform(X_moons)
    affinities = ['rbf', 'nearest_neighbors', 'cosine']
    gamma_vals  = [1.0, None, None]

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for i, (aff, gamma) in enumerate(zip(affinities, gamma_vals)):
        kwargs = {'affinity': aff, 'n_clusters': 2, 'random_state': 42, 'n_init': 10}
        if gamma is not None:
            kwargs['gamma'] = gamma
        sc = SpectralClustering(**kwargs)
        try:
            labels = sc.fit_predict(X_s)
            sil = silhouette_score(X_s, labels)
            status = f'Silhouette={sil:.3f}'
        except Exception as e:
            labels = np.zeros(len(X_s), dtype=int)
            status = f'Error: {str(e)[:30]}'

        axes2[i].scatter(X_s[:, 0], X_s[:, 1], c=labels,
                         cmap='Set1', s=30, edgecolors='k', linewidths=0.3)
        axes2[i].set_title(f'Affinity={aff}\n{status}')
        axes2[i].grid(True, alpha=0.3)

    plt.suptitle('谱聚类不同相似度核函数（月牙形数据）', fontsize=13)
    save_path2 = get_results_path('spectral_clustering_affinity.png')
    save_and_close(save_path2)
    print(f"   核函数对比图已保存: {save_path2}")


if __name__ == "__main__":
    spectral_clustering()
