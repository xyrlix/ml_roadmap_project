# LightGBM 集成学习模型
# 基于梯度提升决策树（GBDT）的高效实现，微软开源框架

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
import lightgbm as lgb
from utils import (generate_classification_data, get_results_path,
                   print_classification_report, save_and_close)


def lightgbm():
    """LightGBM 梯度提升分类器实现"""
    print("LightGBM 模型运行中...\n")

    # 1. 数据准备
    print("1. 准备数据...")
    X, y = generate_classification_data(n_samples=500, n_features=10,
                                         n_informative=6, n_redundant=2,
                                         random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    print(f"   训练集: {X_train.shape[0]} 样本  测试集: {X_test.shape[0]} 样本")
    print(f"   特征数: {X.shape[1]}")

    # 转为 LightGBM Dataset
    train_data = lgb.Dataset(X_train, label=y_train)
    valid_data = lgb.Dataset(X_test, label=y_test, reference=train_data)

    # 2. 模型配置与训练
    print("2. 训练 LightGBM 模型...")
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'random_state': 42
    }

    callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False),
                 lgb.log_evaluation(period=-1)]  # 静默训练

    evals_result = {}
    model = lgb.train(
        params,
        train_data,
        num_boost_round=300,
        valid_sets=[train_data, valid_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks + [lgb.record_evaluation(evals_result)]
    )
    print(f"   最佳迭代轮次: {model.best_iteration}")

    # 3. 预测
    print("3. 模型预测...")
    y_prob = model.predict(X_test, num_iteration=model.best_iteration)
    y_pred = (y_prob >= 0.5).astype(int)

    # 4. 评估
    print("4. 模型评估...")
    accuracy, cm = print_classification_report(y_test, y_pred, "LightGBM")

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：训练曲线（损失） ──
    ax = axes[0]
    ax.plot(evals_result['train']['binary_logloss'], label='Train Loss', color='steelblue')
    ax.plot(evals_result['valid']['binary_logloss'], label='Valid Loss', color='darkorange')
    ax.axvline(x=model.best_iteration - 1, color='green',
               linestyle='--', label=f'Best iter={model.best_iteration}')
    ax.set_title('LightGBM 训练损失曲线')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Binary Log Loss')
    ax.legend()
    ax.grid(True, alpha=0.5)

    # ── 子图2：特征重要性（split次数）──
    ax = axes[1]
    importance = model.feature_importance(importance_type='split')
    feat_names = [f'F{i+1}' for i in range(len(importance))]
    sorted_idx = np.argsort(importance)[::-1]
    ax.barh(range(len(importance)),
            importance[sorted_idx][::-1],
            color='steelblue', edgecolor='white')
    ax.set_yticks(range(len(importance)))
    ax.set_yticklabels([feat_names[i] for i in sorted_idx[::-1]])
    ax.set_title('特征重要性 (Split Count)')
    ax.set_xlabel('Importance')
    ax.grid(True, alpha=0.3, axis='x')

    # ── 子图3：预测概率分布 ──
    ax = axes[2]
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.6,
            color='royalblue', label='Class 0', density=True)
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.6,
            color='tomato', label='Class 1', density=True)
    ax.axvline(x=0.5, color='black', linestyle='--', label='Threshold=0.5')
    ax.set_title('预测概率分布')
    ax.set_xlabel('Predicted Probability')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('lightgbm_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    lightgbm()
