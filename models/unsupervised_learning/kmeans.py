# K-Means 聚类模型
# 经典无监督聚类算法：最小化簇内距离平方和

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.preprocessing import StandardScaler
from utils import generate_clustering_data, get_results_path, save_and_close


def kmeans():
    """K-Means 聚类算法实现"""
    print("K-Means 聚类模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y_true = generate_clustering_data(n_samples=400, n_clusters=4, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   样本数: {X.shape[0]}  特征数: {X.shape[1]}")

    # 2. 用肘部法则选择最优 K
    print("2. 肘部法则选择最优 K...")
    inertias, silhouettes = [], []
    k_range = range(2, 10)
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        silhouettes.append(silhouette_score(X_scaled, labels))

    best_k = k_range[np.argmax(silhouettes)]
    print(f"   轮廓系数最优 K = {best_k}")

    # 3. 训练最终模型
    print(f"3. 使用 K={best_k} 训练模型...")
    model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)

    # 4. 评估
    print("4. 模型评估...")
    sil = silhouette_score(X_scaled, labels)
    ch  = calinski_harabasz_score(X_scaled, labels)
    print(f"   轮廓系数 (Silhouette Score): {sil:.4f}   [越大越好，范围 -1~1]")
    print(f"   CH 指数 (Calinski-Harabasz): {ch:.2f}   [越大越好]")
    print(f"   簇内惯性 (Inertia): {model.inertia_:.2f}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：肘部法则 + 轮廓系数 ──
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(list(k_range), inertias, 'bo-', label='Inertia', linewidth=1.5)
    ax2.plot(list(k_range), silhouettes, 'rs--', label='Silhouette', linewidth=1.5)
    ax.axvline(x=best_k, color='green', linestyle=':', linewidth=2,
               label=f'Best K={best_k}')
    ax.set_xlabel('Number of Clusters K')
    ax.set_ylabel('Inertia', color='blue')
    ax2.set_ylabel('Silhouette Score', color='red')
    ax.set_title('肘部法则 & 轮廓系数')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    ax.grid(True, alpha=0.3)

    # ── 子图2：聚类结果 ──
    ax = axes[1]
    colors = plt.cm.tab10(np.linspace(0, 1, best_k))
    for i in range(best_k):
        mask = labels == i
        ax.scatter(X_scaled[mask, 0], X_scaled[mask, 1],
                   c=[colors[i]], label=f'Cluster {i}', s=40,
                   edgecolors='k', linewidths=0.3, alpha=0.8)
    centers = model.cluster_centers_
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='*',
               s=300, zorder=10, label='Centroids', edgecolors='black')
    ax.set_title(f'K-Means 聚类结果 (K={best_k})')
    ax.set_xlabel('Feature 1 (Scaled)')
    ax.set_ylabel('Feature 2 (Scaled)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图3：Voronoi 区域（预测面） ──
    ax = axes[2]
    x_min, x_max = X_scaled[:, 0].min() - 0.5, X_scaled[:, 0].max() + 0.5
    y_min, y_max = X_scaled[:, 1].min() - 0.5, X_scaled[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap='tab10',
                levels=np.arange(-0.5, best_k + 0.5, 1))
    ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels,
               cmap='tab10', s=20, edgecolors='k', linewidths=0.2, alpha=0.7)
    ax.scatter(centers[:, 0], centers[:, 1], c='red', marker='*',
               s=250, zorder=10, edgecolors='black')
    ax.set_title('Voronoi 划分区域')
    ax.set_xlabel('Feature 1 (Scaled)')
    ax.set_ylabel('Feature 2 (Scaled)')
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('kmeans_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    kmeans()
