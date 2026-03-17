# LOF 局部离群因子异常检测模型
# 比较样本局部密度与邻居的密度差异，局部密度显著低的样本为异常

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, roc_auc_score
from utils import get_results_path, save_and_close


def generate_data(n_normal=300, n_anomaly=30, seed=42):
    rng = np.random.default_rng(seed)
    # 三个密度不同的正常簇
    X_normal = np.vstack([
        rng.multivariate_normal([0, 0], [[0.5, 0.1], [0.1, 0.5]], n_normal // 3),
        rng.multivariate_normal([5, 0], [[1.5, 0], [0, 0.3]], n_normal // 3),
        rng.multivariate_normal([2, 5], [[0.3, 0], [0, 1.0]], n_normal // 3),
    ])
    X_anomaly = rng.uniform(-3, 9, (n_anomaly, 2))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([1] * n_normal + [-1] * n_anomaly)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def lof():
    """LOF 局部离群因子异常检测实现"""
    print("LOF 局部离群因子模型运行中...\n")

    # 1. 数据准备
    print("1. 准备多密度聚类数据...")
    X, y_true = generate_data()
    X = StandardScaler().fit_transform(X)
    anomaly_rate = (y_true == -1).mean()
    print(f"   样本数: {len(X)}  异常率: {anomaly_rate:.1%}")

    # 2. LOF 检测
    print("2. 训练 LOF 模型...")
    model = LocalOutlierFactor(
        n_neighbors=20,
        contamination=anomaly_rate,
        novelty=False     # 无监督模式
    )
    y_pred = model.fit_predict(X)   # 1=正常, -1=异常
    lof_scores = -model.negative_outlier_factor_   # 越大越异常

    precision = (y_pred[y_true == -1] == -1).mean()
    recall    = (y_pred[y_true == -1] == -1).mean()  # simplified
    f1  = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
    auc = roc_auc_score(y_true == -1, lof_scores)
    print(f"   F1={f1:.4f}  AUC={auc:.4f}")

    # 3. n_neighbors 参数分析
    print("3. n_neighbors 参数分析...")
    k_range = [5, 10, 15, 20, 30, 50, 70]
    k_f1, k_auc = [], []
    for k in k_range:
        m = LocalOutlierFactor(n_neighbors=k,
                               contamination=anomaly_rate, novelty=False)
        yp = m.fit_predict(X)
        sc = -m.negative_outlier_factor_
        k_f1.append(f1_score(y_true, yp, pos_label=-1, zero_division=0))
        k_auc.append(roc_auc_score(y_true == -1, sc))
        print(f"   k={k:3d}  F1={k_f1[-1]:.4f}  AUC={k_auc[-1]:.4f}")

    # 4. 与其他算法对比
    print("4. 与孤立森林对比...")
    from sklearn.ensemble import IsolationForest
    m_if = IsolationForest(contamination=anomaly_rate,
                           n_estimators=100, random_state=42)
    m_if.fit(X)
    yp_if = m_if.predict(X)
    sc_if = -m_if.score_samples(X)
    f1_if  = f1_score(y_true, yp_if, pos_label=-1, zero_division=0)
    auc_if = roc_auc_score(y_true == -1, sc_if)
    print(f"   LOF:              F1={f1:.4f}  AUC={auc:.4f}")
    print(f"   Isolation Forest: F1={f1_if:.4f}  AUC={auc_if:.4f}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：LOF 异常检测结果（气泡大小=LOF分数）──
    ax = axes[0]
    # 正常点
    mask_n = y_pred == 1
    ax.scatter(X[mask_n, 0], X[mask_n, 1], c='steelblue',
               s=20, alpha=0.5, label=f'Normal ({mask_n.sum()})')
    # 预测异常（气泡大小与LOF分数成正比）
    mask_a = y_pred == -1
    sizes  = lof_scores[mask_a] * 30
    ax.scatter(X[mask_a, 0], X[mask_a, 1], c='red',
               s=np.clip(sizes, 30, 300),
               alpha=0.8, edgecolors='darkred', linewidths=0.8,
               zorder=5, label=f'Anomaly ({mask_a.sum()})')
    # 真实异常点标出边框
    true_anom = y_true == -1
    ax.scatter(X[true_anom, 0], X[true_anom, 1],
               facecolors='none', edgecolors='gold',
               s=80, linewidths=1.5, zorder=6, label='True Anomaly')
    ax.set_title(f'LOF 检测结果 (k=20)\nF1={f1:.3f}  AUC={auc:.3f}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── 子图2：n_neighbors 敏感性 ──
    ax = axes[1]
    ax_r = ax.twinx()
    ax.plot(k_range, k_f1, 'bo-', linewidth=1.8, label='F1')
    ax_r.plot(k_range, k_auc, 'rs--', linewidth=1.8, label='AUC')
    ax.set_xlabel('n_neighbors (k)')
    ax.set_ylabel('F1 Score', color='blue')
    ax_r.set_ylabel('AUC', color='red')
    ax.set_title('n_neighbors 参数敏感性')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图3：LOF vs IForest 对比 ──
    ax = axes[2]
    methods = ['LOF', 'Isolation\nForest']
    f1_vals  = [f1, f1_if]
    auc_vals = [auc, auc_if]
    x_pos = np.arange(len(methods))
    width = 0.3
    bars1 = ax.bar(x_pos - width/2, f1_vals, width,
                   label='F1', color=['steelblue', 'darkorange'],
                   edgecolor='white')
    bars2 = ax.bar(x_pos + width/2, auc_vals, width,
                   label='AUC', color=['royalblue', 'coral'],
                   edgecolor='white')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=10)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.set_title('LOF vs Isolation Forest')
    ax.set_ylim(0, 1.15)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    save_path = get_results_path('lof_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    lof()
