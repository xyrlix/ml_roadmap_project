# DBSCAN 密度聚类模型
# 基于密度的空间聚类算法，能够发现任意形状簇并识别噪声点

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from sklearn.datasets import make_moons, make_circles
from utils import get_results_path, save_and_close


def dbscan():
    """DBSCAN 密度聚类实现"""
    print("DBSCAN 密度聚类模型运行中...\n")

    # 1. 数据准备 —— 使用非线性分布数据体现 DBSCAN 优势
    print("1. 准备数据（月牙形 + 环形 + 球形）...")
    np.random.seed(42)
    X_moons, y_moons = make_moons(n_samples=200, noise=0.08, random_state=42)
    X_circles, y_circles = make_circles(n_samples=200, noise=0.05,
                                         factor=0.5, random_state=42)
    X_blobs = np.vstack([
        np.random.randn(80, 2) * 0.4 + [0, 0],
        np.random.randn(80, 2) * 0.4 + [3, 3],
        np.random.randn(30, 2) * 2.5          # 噪声区域
    ])
    datasets = [
        (X_moons, "月牙形数据"),
        (X_circles, "同心圆数据"),
        (X_blobs, "含噪声的球形数据")
    ]

    # DBSCAN 超参数（针对各数据集微调）
    dbscan_params = [
        {'eps': 0.2, 'min_samples': 5},
        {'eps': 0.15, 'min_samples': 5},
        {'eps': 0.5, 'min_samples': 5},
    ]

    # 2. 训练 & 评估
    print("2. 训练 DBSCAN 模型...")
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    for col, ((X, title), params) in enumerate(zip(datasets, dbscan_params)):
        scaler = StandardScaler()
        X_s = scaler.fit_transform(X)

        model = DBSCAN(**params)
        labels = model.fit_predict(X_s)

        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = (labels == -1).sum()
        print(f"   [{title}] 簇数: {n_clusters}  噪声点: {n_noise}/{len(X)}")

        sil = silhouette_score(X_s, labels) if n_clusters > 1 else float('nan')

        # ── 上行：聚类散点图 ──
        ax = axes[0, col]
        unique_labels = sorted(set(labels))
        cmap = plt.cm.tab10
        for k in unique_labels:
            mask = labels == k
            if k == -1:
                ax.scatter(X_s[mask, 0], X_s[mask, 1], c='black',
                           marker='x', s=50, label='Noise', zorder=3)
            else:
                color = cmap(k / max(n_clusters, 1))
                ax.scatter(X_s[mask, 0], X_s[mask, 1], c=[color],
                           s=30, edgecolors='k', linewidths=0.3,
                           alpha=0.8, label=f'Cluster {k}')
        ax.set_title(f'{title}\neps={params["eps"]}, min_samples={params["min_samples"]}')
        ax.set_xlabel('Feature 1 (Scaled)')
        ax.set_ylabel('Feature 2 (Scaled)')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

        # ── 下行：统计信息条形图 ──
        ax2 = axes[1, col]
        cluster_sizes = [(labels == k).sum() for k in range(n_clusters)]
        if cluster_sizes:
            ax2.bar(range(n_clusters), cluster_sizes,
                    color=plt.cm.tab10(np.linspace(0, 1, n_clusters)),
                    edgecolor='white')
            ax2.set_xticks(range(n_clusters))
            ax2.set_xticklabels([f'C{i}' for i in range(n_clusters)])
        ax2.bar(n_clusters, n_noise, color='gray', label='Noise')
        stats_text = (f'Clusters: {n_clusters}\n'
                      f'Noise: {n_noise}\n'
                      f'Silhouette: {sil:.3f}')
        ax2.text(0.97, 0.97, stats_text, transform=ax2.transAxes,
                 ha='right', va='top', fontsize=9,
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax2.set_title(f'簇大小分布 & 统计')
        ax2.set_xlabel('Cluster / Noise')
        ax2.set_ylabel('Sample Count')
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('DBSCAN 密度聚类 — 不同数据集对比', fontsize=14, y=1.01)
    save_path = get_results_path('dbscan_result.png')
    save_and_close(save_path)
    print(f"\n   图表已保存: {save_path}")

    # 3. 参数敏感性分析
    print("3. eps 参数敏感性分析（月牙形数据）...")
    X_s = StandardScaler().fit_transform(X_moons)
    eps_values = [0.05, 0.1, 0.15, 0.2, 0.3, 0.5]
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 9))
    for i, eps in enumerate(eps_values):
        ax = axes2[i // 3, i % 3]
        model = DBSCAN(eps=eps, min_samples=5)
        labels = model.fit_predict(X_s)
        n_c = len(set(labels)) - (1 if -1 in labels else 0)
        n_n = (labels == -1).sum()
        ax.scatter(X_s[:, 0], X_s[:, 1], c=labels,
                   cmap='tab10', s=20, edgecolors='k', linewidths=0.2)
        ax.set_title(f'eps={eps}  簇:{n_c} 噪声:{n_n}')
        ax.axis('off')

    plt.suptitle('DBSCAN eps 参数敏感性（月牙形，min_samples=5）', fontsize=13)
    save_path2 = get_results_path('dbscan_eps_sensitivity.png')
    save_and_close(save_path2)
    print(f"   eps 敏感性图已保存: {save_path2}")


if __name__ == "__main__":
    dbscan()
