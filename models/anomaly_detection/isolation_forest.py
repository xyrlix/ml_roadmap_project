# 孤立森林异常检测模型
# 通过随机分割空间隔离异常点，异常点需要更少的分割次数

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from utils import get_results_path, save_and_close


def generate_anomaly_data(n_normal=300, n_anomaly=30, seed=42):
    """生成正常数据（多峰高斯）+ 异常数据（均匀分布）"""
    rng = np.random.default_rng(seed)
    # 正常数据：两个高斯簇
    X_normal = np.vstack([
        rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_normal // 2),
        rng.multivariate_normal([5, 5], [[1, -0.3], [-0.3, 0.8]], n_normal // 2)
    ])
    # 异常点：散落在边缘
    X_anomaly = rng.uniform(-6, 12, (n_anomaly, 2))
    X = np.vstack([X_normal, X_anomaly])
    y = np.array([1] * n_normal + [-1] * n_anomaly)  # 1=正常, -1=异常
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def isolation_forest():
    """孤立森林（Isolation Forest）异常检测实现"""
    print("孤立森林异常检测模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y_true = generate_anomaly_data(n_normal=300, n_anomaly=30)
    X = StandardScaler().fit_transform(X)
    anomaly_rate = (y_true == -1).mean()
    print(f"   样本数: {len(X)}  异常率: {anomaly_rate:.1%}")

    # 2. 训练孤立森林
    print("2. 训练孤立森林模型...")
    model = IsolationForest(
        n_estimators=200,
        contamination=anomaly_rate,
        max_samples='auto',
        random_state=42,
        n_jobs=-1
    )
    model.fit(X)
    y_pred  = model.predict(X)          # 1=正常, -1=异常
    scores  = model.score_samples(X)    # 越负越异常

    # 3. 评估
    print("3. 模型评估...")
    precision = precision_score(y_true, y_pred, pos_label=-1, zero_division=0)
    recall    = recall_score(y_true, y_pred, pos_label=-1, zero_division=0)
    f1        = f1_score(y_true, y_pred, pos_label=-1, zero_division=0)
    auc       = roc_auc_score(y_true == -1, -scores)
    print(f"   Precision: {precision:.4f}  Recall: {recall:.4f}  "
          f"F1: {f1:.4f}  AUC: {auc:.4f}")

    # 4. 污染率敏感性分析
    print("4. 污染率（contamination）参数分析...")
    contam_range = [0.01, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25]
    f1_scores, auc_scores = [], []
    for c in contam_range:
        m = IsolationForest(n_estimators=100, contamination=c,
                            random_state=42, n_jobs=-1)
        m.fit(X)
        yp = m.predict(X)
        sc = m.score_samples(X)
        f1_scores.append(f1_score(y_true, yp, pos_label=-1, zero_division=0))
        auc_scores.append(roc_auc_score(y_true == -1, -sc))
    print(f"   最佳 F1 对应 contamination={contam_range[np.argmax(f1_scores)]:.2f}")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：决策边界 ──
    ax = axes[0]
    xx, yy = np.meshgrid(
        np.linspace(X[:, 0].min()-0.5, X[:, 0].max()+0.5, 200),
        np.linspace(X[:, 1].min()-0.5, X[:, 1].max()+0.5, 200)
    )
    Z = model.decision_function(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=20, cmap='RdYlGn', alpha=0.6)
    ax.contour(xx, yy, Z, levels=[0], linewidths=2, colors='black',
               linestyles='--')
    # 正常点
    mask_n = y_pred == 1
    ax.scatter(X[mask_n, 0], X[mask_n, 1], c='steelblue', s=20,
               alpha=0.6, label=f'Normal ({mask_n.sum()})')
    # 预测异常点
    mask_a = y_pred == -1
    ax.scatter(X[mask_a, 0], X[mask_a, 1], c='red', s=60,
               marker='x', linewidths=1.5, zorder=5,
               label=f'Anomaly ({mask_a.sum()})')
    ax.set_title(f'孤立森林决策边界\nF1={f1:.3f}  AUC={auc:.3f}')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ── 子图2：异常分数分布 ──
    ax = axes[1]
    ax.hist(scores[y_true == 1], bins=30, alpha=0.6,
            color='steelblue', label='Normal', density=True)
    ax.hist(scores[y_true == -1], bins=15, alpha=0.6,
            color='tomato', label='Anomaly', density=True)
    ax.axvline(x=np.percentile(scores, anomaly_rate * 100),
               color='black', linestyle='--', label='Threshold')
    ax.set_title('异常分数分布')
    ax.set_xlabel('Anomaly Score')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图3：污染率 vs F1/AUC ──
    ax = axes[2]
    ax2 = ax.twinx()
    ax.plot(contam_range, f1_scores, 'bo-', linewidth=1.8, label='F1')
    ax2.plot(contam_range, auc_scores, 'rs--', linewidth=1.8, label='AUC')
    ax.axvline(x=anomaly_rate, color='green', linestyle=':',
               label=f'True rate={anomaly_rate:.2f}')
    ax.set_xlabel('Contamination Rate')
    ax.set_ylabel('F1 Score', color='blue')
    ax2.set_ylabel('AUC', color='red')
    ax.set_title('污染率参数敏感性分析')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('isolation_forest_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    isolation_forest()
