# 半监督 SVM 模型（S3VM 思想）
# 通过瞬态标签传播实现半监督 SVM，利用无标签数据改善决策边界

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from sklearn.semi_supervised import LabelSpreading
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.datasets import make_moons
from utils import get_results_path, save_and_close


def semi_supervised_svm():
    """半监督 SVM 实现（使用 LabelSpreading + SVM 两阶段策略）"""
    print("半监督 SVM 模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据（非线性月牙形）...")
    np.random.seed(42)
    X_full, y_full = make_moons(n_samples=400, noise=0.15, random_state=42)
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # 2. 构建不同标注比例的实验
    labeled_rates  = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
    acc_supervised  = []   # 仅有标签 SVM
    acc_semi_svm    = []   # 半监督 SVM

    print("2. 对比不同标注比例下：纯 SVM vs 半监督 SVM...")
    for rate in labeled_rates:
        n_lab = max(6, int(len(X_full) * rate))
        labeled_idx = []
        for cls in range(2):
            cls_idx = np.where(y_full == cls)[0]
            labeled_idx.extend(
                np.random.choice(cls_idx, n_lab // 2, replace=False).tolist())
        unlabeled_idx = np.setdiff1d(np.arange(len(X_full)), labeled_idx)

        # 纯监督 SVM（仅用有标签）
        svm_sup = SVC(kernel='rbf', C=10, gamma='scale', probability=True)
        svm_sup.fit(X_full[labeled_idx], y_full[labeled_idx])
        acc_sup = accuracy_score(y_full[unlabeled_idx],
                                  svm_sup.predict(X_full[unlabeled_idx]))
        acc_supervised.append(acc_sup)

        # 两阶段半监督 SVM:
        #   Step1: LabelSpreading 给无标签样本分配软标签
        #   Step2: 用全部样本（含软标签）重训 SVM
        y_mix = np.full(len(X_full), -1, dtype=int)
        for i in labeled_idx:
            y_mix[i] = y_full[i]

        ls = LabelSpreading(kernel='rbf', gamma=20, alpha=0.2, max_iter=500)
        ls.fit(X_full, y_mix)
        y_soft = ls.predict(X_full)   # 所有样本的预测标签

        svm_semi = SVC(kernel='rbf', C=10, gamma='scale')
        svm_semi.fit(X_full, y_soft)
        acc_semi = accuracy_score(y_full[unlabeled_idx],
                                   svm_semi.predict(X_full[unlabeled_idx]))
        acc_semi_svm.append(acc_semi)
        print(f"   rate={rate:.0%}  Supervised={acc_sup:.3f}  "
              f"Semi-SVM={acc_semi:.3f}")

    # 3. 决策边界对比（选取 rate=0.1）
    print("3. 可视化决策边界（标注率=10%）...")
    rate_vis = 0.10
    n_lab = max(6, int(len(X_full) * rate_vis))
    labeled_idx = []
    for cls in range(2):
        cls_idx = np.where(y_full == cls)[0]
        labeled_idx.extend(
            np.random.choice(cls_idx, n_lab // 2, replace=False).tolist())
    unlabeled_idx = np.setdiff1d(np.arange(len(X_full)), labeled_idx)

    y_mix = np.full(len(X_full), -1, dtype=int)
    for i in labeled_idx:
        y_mix[i] = y_full[i]

    ls = LabelSpreading(kernel='rbf', gamma=20, alpha=0.2)
    ls.fit(X_full, y_mix)
    y_soft_vis = ls.predict(X_full)

    svm_sup_vis  = SVC(kernel='rbf', C=10, gamma='scale')
    svm_sup_vis.fit(X_full[labeled_idx], y_full[labeled_idx])
    svm_semi_vis = SVC(kernel='rbf', C=10, gamma='scale')
    svm_semi_vis.fit(X_full, y_soft_vis)

    # 网格预测
    xx, yy = np.meshgrid(
        np.linspace(X_full[:, 0].min()-0.5, X_full[:, 0].max()+0.5, 200),
        np.linspace(X_full[:, 1].min()-0.5, X_full[:, 1].max()+0.5, 200)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]

    # 4. 可视化
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：纯 SVM 决策边界 ──
    ax = axes[0]
    Z_sup = svm_sup_vis.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z_sup, alpha=0.25, cmap='coolwarm')
    ax.scatter(X_full[unlabeled_idx, 0], X_full[unlabeled_idx, 1],
               c='lightgray', s=20, alpha=0.5, label='Unlabeled')
    for cls, c in enumerate(['royalblue', 'tomato']):
        mask = np.array(labeled_idx)[
            y_full[np.array(labeled_idx)] == cls]
        ax.scatter(X_full[mask, 0], X_full[mask, 1],
                   c=c, s=80, edgecolors='k',
                   linewidths=0.8, zorder=5, label=f'Labeled cls={cls}')
    ax.set_title(f'纯监督 SVM (10%标注)\nAcc={acc_supervised[1]:.3f}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── 子图2：半监督 SVM 决策边界 ──
    ax = axes[1]
    Z_semi = svm_semi_vis.predict(grid).reshape(xx.shape)
    ax.contourf(xx, yy, Z_semi, alpha=0.25, cmap='coolwarm')
    ax.scatter(X_full[:, 0], X_full[:, 1], c=y_full,
               cmap='coolwarm', s=20, alpha=0.5, edgecolors='k',
               linewidths=0.2)
    ax.scatter(X_full[labeled_idx, 0], X_full[labeled_idx, 1],
               c='yellow', s=100, edgecolors='k', linewidths=1.0,
               zorder=5, label='Labeled', marker='*')
    ax.set_title(f'半监督 SVM (10%标注)\nAcc={acc_semi_svm[1]:.3f}')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

    # ── 子图3：准确率 vs 标注率 ──
    ax = axes[2]
    ax.plot(labeled_rates, acc_supervised, 'bo-', linewidth=1.8,
            label='Supervised SVM', markersize=8)
    ax.plot(labeled_rates, acc_semi_svm, 'rs-', linewidth=1.8,
            label='Semi-Supervised SVM', markersize=8)
    ax.fill_between(labeled_rates,
                    acc_supervised, acc_semi_svm,
                    alpha=0.15, color='green', label='Semi-SVM 提升')
    ax.set_xlabel('Labeled Rate')
    ax.set_ylabel('Accuracy (on unlabeled data)')
    ax.set_title('标注率 vs 准确率对比')
    ax.legend()
    ax.set_xlim(0, 0.55)
    ax.set_ylim(0.5, 1.0)
    ax.grid(True, alpha=0.3)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:.0%}'))

    save_path = get_results_path('semi_supervised_svm_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    semi_supervised_svm()
