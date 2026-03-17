# 随机森林集成学习模型（增强版）
# Bagging 思想：多棵决策树并行投票，降低方差

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, learning_curve, cross_val_score
from sklearn.metrics import accuracy_score
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, save_and_close)


def random_forest():
    """随机森林分类器（集成学习版）实现"""
    print("随机森林集成学习模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=500, n_features=10,
                                         n_informative=6, n_redundant=2,
                                         random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   训练集: {X_train.shape[0]} 样本  测试集: {X_test.shape[0]} 样本")
    print(f"   特征数: {X.shape[1]}")

    # 2. 训练随机森林（启用 OOB 误差）
    print("2. 训练随机森林模型（n_estimators=200，启用 OOB）...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        oob_score=True,       # 袋外误差估计
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    print(f"   OOB 袋外得分: {model.oob_score_:.4f}")

    # 3. 预测 & 评估
    print("3. 模型预测 & 评估...")
    y_pred = model.predict(X_test)
    accuracy, cm = print_classification_report(y_test, y_pred, "随机森林（集成版）")

    # 4. 不同 n_estimators 下的 OOB 误差曲线
    print("4. 计算 OOB 误差随树数量的变化...")
    estimator_range = [10, 20, 50, 100, 150, 200]
    oob_errors, test_accs = [], []
    for n in estimator_range:
        rf = RandomForestClassifier(n_estimators=n, oob_score=True,
                                    random_state=42, n_jobs=-1)
        rf.fit(X_train, y_train)
        oob_errors.append(1 - rf.oob_score_)
        test_accs.append(accuracy_score(y_test, rf.predict(X_test)))

    # 5. 特征重要性
    importances = model.feature_importances_
    feat_names = [f'F{i+1}' for i in range(len(importances))]
    sorted_idx = np.argsort(importances)[::-1]

    # 6. 交叉验证学习曲线
    print("5. 计算学习曲线...")
    train_sizes, train_scores, val_scores = learning_curve(
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X, y, cv=5, scoring='accuracy',
        train_sizes=np.linspace(0.1, 1.0, 8), n_jobs=-1
    )
    train_mean = train_scores.mean(axis=1)
    train_std  = train_scores.std(axis=1)
    val_mean   = val_scores.mean(axis=1)
    val_std    = val_scores.std(axis=1)

    # 7. 可视化
    print("6. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：OOB 误差 & 测试准确率 vs 树数量 ──
    ax = axes[0]
    ax2 = ax.twinx()
    ax.plot(estimator_range, oob_errors, 'bo-', label='OOB Error', linewidth=1.8)
    ax2.plot(estimator_range, test_accs, 'rs--', label='Test Acc', linewidth=1.8)
    ax.set_xlabel('Number of Trees')
    ax.set_ylabel('OOB Error', color='blue')
    ax2.set_ylabel('Test Accuracy', color='red')
    ax.set_title('OOB 误差 & 测试准确率 vs 树数量')
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc='center right')
    ax.grid(True, alpha=0.3)

    # ── 子图2：特征重要性 ──
    ax = axes[1]
    colors = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(importances)))
    ax.barh(range(len(importances)),
            importances[sorted_idx][::-1],
            color=colors, edgecolor='white')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx[::-1]])
    ax.set_title('特征重要性（Gini）')
    ax.set_xlabel('Importance Score')
    ax.grid(True, alpha=0.3, axis='x')

    # ── 子图3：学习曲线 ──
    ax = axes[2]
    ax.plot(train_sizes, train_mean, 'o-', color='steelblue',
            label='Train Accuracy', linewidth=1.8)
    ax.fill_between(train_sizes, train_mean - train_std,
                    train_mean + train_std, alpha=0.15, color='steelblue')
    ax.plot(train_sizes, val_mean, 's--', color='darkorange',
            label='CV Accuracy', linewidth=1.8)
    ax.fill_between(train_sizes, val_mean - val_std,
                    val_mean + val_std, alpha=0.15, color='darkorange')
    ax.set_title('学习曲线（5-fold CV）')
    ax.set_xlabel('Training Size')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('ensemble_random_forest_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    random_forest()
