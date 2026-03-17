# 层次聚类模型
# 自下而上（凝聚型）的层次聚类，生成树状图展示聚类过程

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
from scipy.cluster.hierarchy import dendrogram, linkage
from utils import generate_clustering_data, get_results_path, save_and_close


def hierarchical_clustering():
    """层次聚类（凝聚型）实现"""
    print("层次聚类模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y_true = generate_clustering_data(n_samples=200, n_clusters=4, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   样本数: {X.shape[0]}")

    # 2. 比较不同链接方式
    print("2. 比较不同链接方式（Ward / Complete / Average / Single）...")
    linkage_methods = ['ward', 'complete', 'average', 'single']
    n_clusters = 4

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    results = {}
    for i, method in enumerate(linkage_methods):
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        labels = model.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results[method] = {'labels': labels, 'silhouette': sil}
        print(f"   {method:8s}  轮廓系数: {sil:.4f}")

        # 上行：散点图
        ax = axes[0, i]
        ax.scatter(X_scaled[:, 0], X_scaled[:, 1], c=labels,
                   cmap='tab10', s=30, edgecolors='k', linewidths=0.3, alpha=0.8)
        ax.set_title(f'Linkage: {method}\nSilhouette={sil:.3f}')
        ax.set_xlabel('F1 (Scaled)')
        ax.set_ylabel('F2 (Scaled)')
        ax.grid(True, alpha=0.3)

        # 下行：树状图（用 scipy linkage 计算）
        ax2 = axes[1, i]
        Z = linkage(X_scaled, method=method)
        dendrogram(Z, ax=ax2, no_labels=True, color_threshold=Z[-(n_clusters-1), 2],
                   above_threshold_color='gray', truncate_mode='lastp', p=30)
        ax2.set_title(f'Dendrogram ({method})')
        ax2.set_xlabel('Sample Index')
        ax2.set_ylabel('Distance')
        ax2.axhline(y=Z[-(n_clusters-1), 2], color='red',
                    linestyle='--', label='Cut line')
        ax2.legend(fontsize=7)
        ax2.grid(True, alpha=0.3, axis='y')

    plt.suptitle('层次聚类：不同链接方式对比', fontsize=14, y=1.01)
    save_path = get_results_path('hierarchical_clustering_result.png')
    save_and_close(save_path)
    print(f"\n   图表已保存: {save_path}")

    # 3. 详细展示最佳方法
    best_method = max(results, key=lambda k: results[k]['silhouette'])
    print(f"\n3. 最佳链接方式: {best_method} "
          f"(轮廓系数={results[best_method]['silhouette']:.4f})")

    # 4. 树状图 + 不同 K 的轮廓系数
    print("4. K 值敏感性分析...")
    Z_ward = linkage(X_scaled, method='ward')
    k_range = range(2, 9)
    sil_scores = []
    for k in k_range:
        model_k = AgglomerativeClustering(n_clusters=k, linkage='ward')
        lbs = model_k.fit_predict(X_scaled)
        sil_scores.append(silhouette_score(X_scaled, lbs))

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # 完整树状图
    ax = axes2[0]
    dendrogram(Z_ward, ax=ax, no_labels=True,
               color_threshold=Z_ward[-(n_clusters-1), 2],
               above_threshold_color='gray')
    ax.axhline(y=Z_ward[-(n_clusters-1), 2], color='red',
               linestyle='--', linewidth=1.5, label=f'K={n_clusters} cut')
    ax.set_title('Ward 树状图（完整）')
    ax.set_xlabel('Sample Index')
    ax.set_ylabel('Merge Distance')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # K 值 vs 轮廓系数
    ax2 = axes2[1]
    ax2.plot(list(k_range), sil_scores, 'bo-', linewidth=2, markersize=8)
    ax2.axvline(x=sil_scores.index(max(sil_scores)) + 2,
                color='red', linestyle='--',
                label=f'Best K={sil_scores.index(max(sil_scores))+2}')
    ax2.set_title('K 值 vs 轮廓系数 (Ward)')
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    save_path2 = get_results_path('hierarchical_dendrogram.png')
    save_and_close(save_path2)
    print(f"   详细树状图已保存: {save_path2}")


if __name__ == "__main__":
    hierarchical_clustering()
