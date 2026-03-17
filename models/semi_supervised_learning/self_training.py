# 自训练模型（Self-Training）
# 用有标签数据训练初始模型，再将高置信度预测结果作为伪标签迭代扩充训练集

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from utils import generate_classification_data, get_results_path, save_and_close


def self_training():
    """自训练（Self-Training）半监督学习实现"""
    print("自训练模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据（10% 有标签 + 90% 无标签）...")
    np.random.seed(42)
    X, y = generate_classification_data(n_samples=500, n_features=10,
                                         n_informative=6, n_redundant=2,
                                         random_state=42)
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # 模拟半监督场景：仅保留 10% 已标注
    LABELED_RATE = 0.10
    n_labeled = int(len(X) * LABELED_RATE)
    labeled_idx = np.random.choice(len(X), n_labeled, replace=False)
    unlabeled_idx = np.setdiff1d(np.arange(len(X)), labeled_idx)

    y_semi = np.full_like(y, -1)          # 无标签设为 -1
    y_semi[labeled_idx] = y[labeled_idx]   # 有标签样本

    print(f"   总样本: {len(X)}  已标注: {n_labeled}  无标签: {len(unlabeled_idx)}")

    # 2. 多种基学习器的 Self-Training 对比
    print("2. 使用不同基学习器进行 Self-Training...")
    base_models = {
        'SVC (rbf)':   SVC(kernel='rbf', probability=True, random_state=42),
        'LogReg':      LogisticRegression(max_iter=500, random_state=42),
    }

    # 全监督基线（仅用有标签数据训练，在全集上测试）
    baselines = {}
    for name, base in base_models.items():
        base_clone = base.__class__(**base.get_params())
        base_clone.fit(X[labeled_idx], y[labeled_idx])
        baselines[name] = accuracy_score(y, base_clone.predict(X))
        print(f"   [基线-仅有标签] {name}: {baselines[name]:.4f}")

    # Self-Training
    st_results = {}
    for name, base in base_models.items():
        st = SelfTrainingClassifier(
            base_estimator=base.__class__(**base.get_params()),
            threshold=0.75,      # 置信度阈值
            max_iter=10,
            verbose=False
        )
        st.fit(X, y_semi)
        acc = accuracy_score(y[unlabeled_idx], st.predict(X[unlabeled_idx]))
        st_results[name] = acc
        print(f"   [Self-Training]  {name}: 无标签准确率={acc:.4f}  "
              f"(迭代={st.n_iter_}  增加了 "
              f"{(st.labeled_iter_ != 0).sum() - n_labeled} 个伪标签)")

    # 3. 迭代过程追踪（LogReg 为例）
    print("3. 追踪伪标签增长过程...")
    iter_accs, iter_sizes = [], []
    thresholds = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]
    for thresh in thresholds:
        st = SelfTrainingClassifier(
            base_estimator=LogisticRegression(max_iter=500, random_state=42),
            threshold=thresh,
            max_iter=15,
            verbose=False
        )
        st.fit(X, y_semi)
        acc_ul = accuracy_score(y[unlabeled_idx], st.predict(X[unlabeled_idx]))
        n_pseudo = int((st.labeled_iter_ != 0).sum()) - n_labeled
        iter_accs.append(acc_ul)
        iter_sizes.append(n_pseudo)
        print(f"   threshold={thresh:.2f}  无标签准确率={acc_ul:.4f}  "
              f"伪标签数={n_pseudo}")

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：基线 vs Self-Training 准确率对比 ──
    ax = axes[0]
    names = list(base_models.keys())
    x_pos = np.arange(len(names))
    width = 0.35
    bars1 = ax.bar(x_pos - width/2,
                   [baselines[n] for n in names],
                   width, label='Supervised (10% labeled)',
                   color='steelblue', edgecolor='white')
    bars2 = ax.bar(x_pos + width/2,
                   [st_results[n] for n in names],
                   width, label='Self-Training',
                   color='darkorange', edgecolor='white')
    for bar in list(bars1) + list(bars2):
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=10)
    ax.set_title('基线 vs Self-Training 准确率')
    ax.set_ylabel('Accuracy (on unlabeled)')
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图2：阈值 vs 准确率 & 伪标签数量 ──
    ax  = axes[1]
    ax2 = ax.twinx()
    ax.plot(thresholds, iter_accs, 'bo-', linewidth=1.8, label='Unlabeled Acc')
    ax2.bar(thresholds, iter_sizes, width=0.03, alpha=0.4,
            color='darkorange', label='Pseudo Labels')
    ax.set_xlabel('Confidence Threshold')
    ax.set_ylabel('Unlabeled Accuracy', color='blue')
    ax2.set_ylabel('# Pseudo Labels Added', color='darkorange')
    ax.set_title('置信度阈值影响分析（LogReg）')
    lines1, labs1 = ax.get_legend_handles_labels()
    lines2, labs2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labs1 + labs2, loc='lower left')
    ax.grid(True, alpha=0.3)

    # ── 子图3：二维投影（PCA 前两维）──
    from sklearn.decomposition import PCA
    ax3 = axes[2]
    pca = PCA(n_components=2, random_state=42)
    X_2d = pca.fit_transform(X)

    # Self-Training 最终预测
    st_final = SelfTrainingClassifier(
        base_estimator=LogisticRegression(max_iter=500, random_state=42),
        threshold=0.75, max_iter=10
    )
    st_final.fit(X, y_semi)
    y_final_pred = st_final.predict(X)

    ax3.scatter(X_2d[labeled_idx, 0], X_2d[labeled_idx, 1],
                c=y[labeled_idx], cmap='coolwarm', s=80,
                edgecolors='black', linewidths=1.0,
                zorder=5, label='Labeled (真实)')
    ax3.scatter(X_2d[unlabeled_idx, 0], X_2d[unlabeled_idx, 1],
                c=y_final_pred[unlabeled_idx], cmap='coolwarm',
                s=20, alpha=0.4, label='Unlabeled (伪标签)')
    ax3.set_title('PCA 二维投影：Self-Training 结果')
    ax3.set_xlabel('PC1')
    ax3.set_ylabel('PC2')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    save_path = get_results_path('self_training_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    self_training()
