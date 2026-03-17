# 标签传播模型（Label Propagation）
# 利用数据流形结构，将少量有标签样本的标签传播到无标签样本

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.semi_supervised import LabelPropagation, LabelSpreading
from sklearn.datasets import make_moons, make_circles
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from utils import get_results_path, save_and_close


def label_propagation():
    """标签传播半监督学习实现"""
    print("标签传播模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据（月牙形非线性数据）...")
    np.random.seed(42)
    X_full, y_full = make_moons(n_samples=300, noise=0.12, random_state=42)
    scaler = StandardScaler()
    X_full = scaler.fit_transform(X_full)

    # 2. 构建半监督场景：仅保留少量已标注样本
    labeled_rates = [0.05, 0.10, 0.20]  # 标注比例
    fig, axes = plt.subplots(2, 3, figsize=(18, 11))

    print("2. 比较不同标注比例下的标签传播效果...")
    results = {}
    for col, rate in enumerate(labeled_rates):
        n_labeled = max(4, int(len(X_full) * rate))
        # 构建 y_mix：无标签样本设为 -1
        y_mix = np.full_like(y_full, -1)
        labeled_idx = np.random.choice(
            np.where(y_full == 0)[0],  n_labeled // 2, replace=False).tolist() + \
            np.random.choice(
            np.where(y_full == 1)[0], n_labeled // 2, replace=False).tolist()
        for i in labeled_idx:
            y_mix[i] = y_full[i]

        # 标签传播
        model_lp = LabelPropagation(kernel='rbf', gamma=20, max_iter=1000)
        model_lp.fit(X_full, y_mix)
        y_pred_lp = model_lp.predict(X_full)

        # 仅对真实无标签样本评估
        unlabeled_idx = np.where(y_mix == -1)[0]
        acc = accuracy_score(y_full[unlabeled_idx], y_pred_lp[unlabeled_idx])
        results[rate] = acc
        print(f"   标注比例={rate:.0%}  已标注={n_labeled}  "
              f"无标签准确率={acc:.4f}")

        # ── 上行：标注情况 ──
        ax = axes[0, col]
        # 未标注
        ax.scatter(X_full[y_mix == -1, 0], X_full[y_mix == -1, 1],
                   c='lightgray', s=30, alpha=0.6, label='Unlabeled')
        # 已标注
        for cls, color in enumerate(['royalblue', 'tomato']):
            mask = (y_mix == cls)
            ax.scatter(X_full[mask, 0], X_full[mask, 1],
                       c=color, s=100, edgecolors='k',
                       linewidths=0.8, zorder=5, label=f'Labeled cls={cls}')
        ax.set_title(f'已标注比例 {rate:.0%} ({n_labeled} 个)')
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

        # ── 下行：传播结果 ──
        ax2 = axes[1, col]
        ax2.scatter(X_full[:, 0], X_full[:, 1], c=y_pred_lp,
                    cmap='coolwarm', s=30, edgecolors='k',
                    linewidths=0.2, alpha=0.8)
        ax2.set_title(f'标签传播结果（无标签准确率={acc:.3f}）')
        ax2.grid(True, alpha=0.3)

    plt.suptitle('标签传播（Label Propagation）— 不同标注比例对比', fontsize=14, y=1.01)
    save_path = get_results_path('label_propagation_result.png')
    save_and_close(save_path)
    print(f"\n   图表已保存: {save_path}")

    # 3. LabelPropagation vs LabelSpreading 对比
    print("3. LabelPropagation vs LabelSpreading 对比...")
    n_labeled = 30
    y_mix = np.full_like(y_full, -1)
    for cls in range(2):
        idxs = np.random.choice(np.where(y_full == cls)[0],
                                n_labeled // 2, replace=False)
        for i in idxs:
            y_mix[i] = y_full[i]

    models_semi = {
        'LabelPropagation (rbf)':  LabelPropagation(kernel='rbf', gamma=20),
        'LabelSpreading (rbf)':    LabelSpreading(kernel='rbf', gamma=20, alpha=0.2),
        'LabelSpreading (knn)':    LabelSpreading(kernel='knn', n_neighbors=7),
    }
    unlabeled_idx = np.where(y_mix == -1)[0]

    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4))
    for i, (name, m) in enumerate(models_semi.items()):
        m.fit(X_full, y_mix)
        y_pred = m.predict(X_full)
        acc_ul = accuracy_score(y_full[unlabeled_idx], y_pred[unlabeled_idx])
        print(f"   {name}: 无标签准确率={acc_ul:.4f}")
        axes2[i].scatter(X_full[:, 0], X_full[:, 1], c=y_pred,
                         cmap='coolwarm', s=25, edgecolors='k',
                         linewidths=0.2, alpha=0.8)
        axes2[i].set_title(f'{name}\nUnlabeled Acc={acc_ul:.3f}')
        axes2[i].grid(True, alpha=0.3)

    plt.suptitle('LabelPropagation vs LabelSpreading', fontsize=12)
    save_path2 = get_results_path('label_propagation_compare.png')
    save_and_close(save_path2)
    print(f"   对比图已保存: {save_path2}")


if __name__ == "__main__":
    label_propagation()
