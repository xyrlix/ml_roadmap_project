# AdaBoost 集成学习模型
# 自适应提升算法：通过迭代调整样本权重，聚焦难分类样本

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, plot_decision_boundary, save_and_close)


def adaboost():
    """AdaBoost 自适应提升分类器实现"""
    print("AdaBoost 模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=300, n_features=2,
                                         n_informative=2, n_redundant=0,
                                         random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"   训练集: {X_train.shape[0]} 样本  测试集: {X_test.shape[0]} 样本")

    # 2. 模型训练
    print("2. 训练 AdaBoost 模型...")
    # 基学习器：深度为1的决策树桩（Decision Stump）
    base_estimator = DecisionTreeClassifier(max_depth=1)
    model = AdaBoostClassifier(
        estimator=base_estimator,
        n_estimators=100,      # 提升轮数
        learning_rate=0.5,     # 每个弱学习器的权重收缩
        algorithm='SAMME',
        random_state=42
    )
    model.fit(X_train, y_train)
    print(f"   实际使用弱学习器数: {len(model.estimators_)}")

    # 3. 预测
    print("3. 模型预测...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)

    # 4. 评估
    print("4. 模型评估...")
    accuracy, cm = print_classification_report(y_test, y_pred, "AdaBoost")

    # 5. 可视化
    print("5. 可视化结果...")

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：决策边界 ──
    ax = axes[0]
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.05),
                         np.arange(y_min, y_max, 0.05))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.35, cmap='coolwarm')
    scatter = ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test,
                         cmap='coolwarm', edgecolors='k', s=60)
    ax.set_title('AdaBoost 决策边界（测试集）')
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    fig.colorbar(scatter, ax=ax)

    # ── 子图2：分类器权重（每轮提升的弱学习器权重）──
    ax = axes[1]
    ax.plot(range(1, len(model.estimator_weights_) + 1),
            model.estimator_weights_, marker='o', color='steelblue', linewidth=1.5)
    ax.set_title('各弱学习器权重 (α)')
    ax.set_xlabel('Estimator Index')
    ax.set_ylabel('Weight α')
    ax.grid(True, alpha=0.5)

    # ── 子图3：随迭代次数增加的测试准确率 ──
    ax = axes[2]
    staged_acc = [accuracy_score(y_test, pred)
                  for pred in model.staged_predict(X_test)]
    ax.plot(range(1, len(staged_acc) + 1), staged_acc,
            color='darkorange', linewidth=1.5)
    ax.axhline(y=accuracy, color='green', linestyle='--', label=f'Final Acc={accuracy:.3f}')
    ax.set_title('AdaBoost 迭代准确率曲线')
    ax.set_xlabel('Number of Estimators')
    ax.set_ylabel('Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.5)

    save_path = get_results_path('adaboost_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    adaboost()
