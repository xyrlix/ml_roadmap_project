# One-Class SVM 异常检测模型
# 学习正常数据的紧密超球面，将偏离的样本判定为异常

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from utils import get_results_path, save_and_close


def generate_data(n_normal=300, n_anomaly=30, seed=42):
    rng = np.random.default_rng(seed)
    X_normal = np.vstack([
        rng.multivariate_normal([0, 0], [[1, 0.4], [0.4, 1]], n_normal // 2),
        rng.multivariate_normal([4, 4], [[0.8, -0.2], [-0.2, 0.8]], n_normal // 2)
    ])
    X_anomaly = rng.uniform(-5, 10, (n_anomaly, 2))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([1] * n_normal + [-1] * n_anomaly)
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def one_class_svm():
    """One-Class SVM 异常检测实现"""
    print("One-Class SVM 模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y_true = generate_data()
    X = StandardScaler().fit_transform(X)
    anomaly_rate = (y_true == -1).mean()
    print(f"   样本数: {len(X)}  异常率: {anomaly_rate:.1%}")

    # 仅用正常样本训练 One-Class SVM
    X_normal = X[y_true == 1]

    # 2. 比较不同核函数
    print("2. 比较不同核函数（rbf / poly / sigmoid）...")
    kernels = ['rbf', 'poly', 'sigmoid']
    nu_val  = anomaly_rate       # nu ≈ 异常比例上界

    results = {}
    for kernel in kernels:
        m = OneClassSVM(kernel=kernel, nu=nu_val, gamma='scale')
        m.fit(X_normal)
        y_pred = m.predict(X)   # 1=正常, -1=异常
        scores = m.score_samples(X)
        f1  = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
        auc = roc_auc_score(y_true == -1, -scores)
        results[kernel] = {'pred': y_pred, 'scores': scores, 'f1': f1, 'auc': auc}
        print(f"   kernel={kernel:8s}  F1={f1:.4f}  AUC={auc:.4f}")

    # 3. nu 参数敏感性
    print("3. nu 参数敏感性分析（rbf 核）...")
    nu_range = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]
    nu_f1, nu_auc = [], []
    for nu in nu_range:
        m = OneClassSVM(kernel='rbf', nu=nu, gamma='scale')
        m.fit(X_normal)
        yp = m.predict(X)
        sc = m.score_samples(X)
        nu_f1.append(f1_score(y_true, yp, pos_label=-1, zero_division=0))
        nu_auc.append(roc_auc_score(y_true == -1, -sc))

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    best_kernel = max(results, key=lambda k: results[k]['f1'])
    best = results[best_kernel]

    # ── 子图1：最佳核函数决策边界 ──
    ax = axes[0]
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
        np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200)
    )
    m_best = OneClassSVM(kernel=best_kernel, nu=nu_val, gamma='scale')
    m_best.fit(X_normal)
    Z = m_best.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=15, cmap='RdYlGn', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2,
               colors='black', linestyles='--')
    mask_n = best['pred'] == 1
    mask_a = best['pred'] == -1
    ax.scatter(X[mask_n, 0], X[mask_n, 1], c='steelblue', s=20,
               alpha=0.5, label=f'Normal ({mask_n.sum()})')
    ax.scatter(X[mask_a, 0], X[mask_a, 1], c='red', s=60,
               marker='x', linewidths=1.5, zorder=5,
               label=f'Anomaly ({mask_a.sum()})')
    ax.set_title(f'One-Class SVM ({best_kernel} 核)\n'
                 f'F1={best["f1"]:.3f}  AUC={best["auc"]:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图2：核函数对比柱状图 ──
    ax = axes[1]
    k_names = list(results.keys())
    f1_vals  = [results[k]['f1']  for k in k_names]
    auc_vals = [results[k]['auc'] for k in k_names]
    x_pos = np.arange(len(k_names))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2, f1_vals, width,
                   label='F1', color='steelblue', edgecolor='white')
    bars2 = ax.bar(x_pos + width/2, auc_vals, width,
                   label='AUC', color='darkorange', edgecolor='white')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.01,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(k_names)
    ax.set_title('不同核函数 F1/AUC 对比')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图3：nu 参数敏感性 ──
    ax = axes[2]
    ax_r = ax.twinx()
    ax.plot(nu_range, nu_f1, 'bo-', linewidth=1.8, label='F1')
    ax_r.plot(nu_range, nu_auc, 'rs--', linewidth=1.8, label='AUC')
    ax.axvline(x=anomaly_rate, color='green', linestyle=':',
               label=f'True rate={anomaly_rate:.2f}')
    ax.set_xlabel('nu (Anomaly Upper Bound)')
    ax.set_ylabel('F1 Score', color='blue')
    ax_r.set_ylabel('AUC', color='red')
    ax.set_title('nu 参数敏感性分析')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax_r.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('one_class_svm_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    one_class_svm()
