# XGBoost 集成学习模型（增强版）
# 梯度提升树（GBDT），支持正则化、缺失值处理，速度快精度高

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.metrics import accuracy_score, roc_curve, auc
from xgboost import XGBClassifier
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, save_and_close)


def xgboost():
    """XGBoost 梯度提升分类器（集成学习版）实现"""
    print("XGBoost 集成学习模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=600, n_features=12,
                                         n_informative=7, n_redundant=3,
                                         random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"   训练集: {X_train.shape[0]} 样本  测试集: {X_test.shape[0]} 样本")
    print(f"   特征数: {X.shape[1]}")

    # 2. 模型训练（含早停）
    print("2. 训练 XGBoost 模型（含早停）...")
    model = XGBClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        eval_metric='logloss',
        early_stopping_rounds=20,
        random_state=42,
        verbosity=0
    )
    model.fit(X_train, y_train,
              eval_set=[(X_train, y_train), (X_test, y_test)],
              verbose=False)
    print(f"   最佳迭代轮次: {model.best_iteration}")

    # 3. 预测 & 评估
    print("3. 模型预测 & 评估...")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    accuracy, cm = print_classification_report(y_test, y_pred, "XGBoost（集成版）")

    # ROC 曲线
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)
    print(f"   AUC: {roc_auc:.4f}")

    # 4. 不同超参数对比（学习率）
    print("4. 不同学习率对比...")
    lr_values = [0.01, 0.05, 0.1, 0.2, 0.5]
    lr_accs = []
    for lr in lr_values:
        m = XGBClassifier(n_estimators=100, learning_rate=lr,
                          max_depth=4, random_state=42, verbosity=0,
                          eval_metric='logloss')
        m.fit(X_train, y_train)
        lr_accs.append(accuracy_score(y_test, m.predict(X_test)))

    # 5. 特征重要性（gain 类型）
    importances = model.feature_importances_
    feat_names = [f'F{i+1}' for i in range(len(importances))]
    sorted_idx = np.argsort(importances)[::-1]

    # 6. 训练历史（evals_result）
    evals = model.evals_result()
    train_loss = evals['validation_0']['logloss']
    val_loss   = evals['validation_1']['logloss']

    # 7. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：训练/验证损失曲线 ──
    ax = axes[0]
    ax.plot(train_loss, label='Train Loss', color='steelblue', linewidth=1.5)
    ax.plot(val_loss, label='Val Loss', color='darkorange', linewidth=1.5)
    ax.axvline(x=model.best_iteration, color='green', linestyle='--',
               label=f'Best iter={model.best_iteration}')
    ax.set_title('XGBoost 训练损失曲线')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Log Loss')
    ax.legend()
    ax.grid(True, alpha=0.4)

    # ── 子图2：特征重要性 ──
    ax = axes[1]
    ax.barh(range(len(importances)),
            importances[sorted_idx][::-1],
            color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx[::-1]])
    ax.set_title('特征重要性（F-score Weight）')
    ax.set_xlabel('Importance')
    ax.grid(True, alpha=0.3, axis='x')

    # ── 子图3：ROC 曲线 + 学习率对比 ──
    ax = axes[2]
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC (AUC={roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC 曲线')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('ensemble_xgboost_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")

    # 额外图：学习率对比柱状图
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    bars = ax2.bar([str(lr) for lr in lr_values], lr_accs,
                   color=plt.cm.Blues(np.linspace(0.4, 0.9, len(lr_values))),
                   edgecolor='white')
    for bar, acc in zip(bars, lr_accs):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                 f'{acc:.3f}', ha='center', va='bottom', fontsize=10)
    ax2.set_title('不同学习率下的测试准确率（n_estimators=100）')
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Test Accuracy')
    ax2.set_ylim(0, 1.05)
    ax2.grid(True, alpha=0.3, axis='y')
    save_path2 = get_results_path('ensemble_xgboost_lr_compare.png')
    save_and_close(save_path2)
    print(f"   学习率对比图已保存: {save_path2}")


if __name__ == "__main__":
    xgboost()
