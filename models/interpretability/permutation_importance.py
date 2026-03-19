"""
排列特征重要性 (Permutation Feature Importance)
==================================================
模型无关的特征重要性评估方法：随机打乱某个特征，观察性能下降幅度

核心原理：
  对于特征 j：
  1. 记录基准性能 s0 = score(X_test, y_test)
  2. 随机打乱 X_test 的第 j 列得到 X_perm
  3. 重新计算 s_perm = score(X_perm, y_test)
  4. 重要性 = s0 - s_perm（下降越多越重要）
  重复多次取均值，减少随机性

优点：
  - 模型无关（适用任何黑盒）
  - 考虑特征间交互
  - 不依赖模型内部结构
缺点：
  - 相关特征可能相互稀释
  - 计算成本 O(n_feat * n_repeats)
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier,
                               RandomForestRegressor)
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, r2_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 排列重要性实现 ──────────────────────────

def permutation_importance(model, X, y, metric_fn, n_repeats=10, random_state=42):
    """
    计算排列特征重要性

    参数：
      model:     已训练模型（有 predict 方法）
      X:         测试集特征 (n_samples, n_features)
      y:         测试集标签
      metric_fn: 评估函数 f(y_true, y_pred) -> float
      n_repeats: 重复打乱次数
      random_state: 随机种子

    返回：
      importances:  (n_features, n_repeats) 数组，每列为一次重复的重要性
      baseline:     基准性能
    """
    rng = np.random.default_rng(random_state)
    n_feat = X.shape[1]

    # 基准性能
    baseline = metric_fn(y, model.predict(X))

    importances = np.zeros((n_feat, n_repeats))
    for j in range(n_feat):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_score = metric_fn(y, model.predict(X_perm))
            importances[j, r] = baseline - perm_score  # 性能下降

    return importances, baseline


def permutation_importance_proba(model, X, y, metric_fn, n_repeats=10, random_state=42):
    """支持 predict_proba 的版本（用于 AUC 等概率指标）"""
    from sklearn.metrics import roc_auc_score
    rng = np.random.default_rng(random_state)
    n_feat = X.shape[1]

    baseline = metric_fn(y, model.predict_proba(X)[:, 1])
    importances = np.zeros((n_feat, n_repeats))
    for j in range(n_feat):
        for r in range(n_repeats):
            X_perm = X.copy()
            X_perm[:, j] = rng.permutation(X_perm[:, j])
            perm_score = metric_fn(y, model.predict_proba(X_perm)[:, 1])
            importances[j, r] = baseline - perm_score
    return importances, baseline


def permutation_importance_main():
    print("排列特征重要性运行中...\n")

    np.random.seed(42)

    # ── 数据集1：分类（10个特征，4个真实有效） ────────────────────
    print("1. 分类任务：比较 RF / GBM / LR 的排列重要性...")
    feature_names_cls = [f"Feature_{i:02d}" for i in range(10)]
    X_cls, y_cls = make_classification(
        n_samples=1000, n_features=10, n_informative=4,
        n_redundant=2, n_repeated=1, random_state=42
    )
    X_tr_c, X_te_c, y_tr_c, y_te_c = train_test_split(X_cls, y_cls, test_size=0.3, random_state=42)
    scaler_c = StandardScaler()
    X_tr_c_s = scaler_c.fit_transform(X_tr_c)
    X_te_c_s = scaler_c.transform(X_te_c)

    models_cls = {
        "RandomForest": RandomForestClassifier(n_estimators=100, random_state=42),
        "GBM":          GradientBoostingClassifier(n_estimators=100, random_state=42),
        "LogisticReg":  LogisticRegression(max_iter=500, random_state=42),
    }
    for m in models_cls.values():
        m.fit(X_tr_c_s, y_tr_c)

    pi_results_cls = {}
    for name, model in models_cls.items():
        imp, base = permutation_importance(
            model, X_te_c_s, y_te_c, accuracy_score, n_repeats=10
        )
        pi_results_cls[name] = {"imp": imp, "base": base}
        print(f"   {name:<14} 基准Acc={base:.4f}")

    # ── 数据集2：回归（8个特征） ──────────────────────────────────
    print("\n2. 回归任务：排列重要性 vs 内置重要性...")
    feature_names_reg = [f"X{i}" for i in range(8)]
    X_reg, y_reg = make_regression(
        n_samples=800, n_features=8, n_informative=5,
        noise=0.5, random_state=42
    )
    X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(X_reg, y_reg, test_size=0.3, random_state=42)
    scaler_r = StandardScaler()
    X_tr_r_s = scaler_r.fit_transform(X_tr_r)
    X_te_r_s = scaler_r.transform(X_te_r)

    rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_reg.fit(X_tr_r_s, y_tr_r)
    base_r2 = r2_score(y_te_r, rf_reg.predict(X_te_r_s))
    print(f"   RandomForest Regression R²={base_r2:.4f}")

    imp_reg, _ = permutation_importance(
        rf_reg, X_te_r_s, y_te_r, r2_score, n_repeats=15
    )

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n3. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    COLORS = {"RandomForest": "#e74c3c", "GBM": "#3498db", "LogisticReg": "#2ecc71"}

    # ── 子图1~3：各分类模型的排列重要性 ─────────────────────────
    for ax, (name, res) in zip(axes[0, :], pi_results_cls.items()):
        imp = res["imp"]
        means = imp.mean(axis=1)
        stds  = imp.std(axis=1)
        order = np.argsort(means)[::-1]

        ax.barh(
            [feature_names_cls[i] for i in order[::-1]],
            means[order[::-1]],
            xerr=stds[order[::-1]],
            color=COLORS[name], alpha=0.85,
            capsize=3, error_kw={"ecolor": "#aaa", "elinewidth": 0.8}
        )
        ax.axvline(0, color="white", linewidth=0.8, alpha=0.6)
        ax.set_title(f"{name}\n排列重要性 (Acc drop)", color="white", pad=8)
        ax.set_xlabel("Performance Drop (Acc)", color="#aaa")
        ax.tick_params(colors="gray")
        for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：回归排列重要性 箱线图 ────────────────────────────
    ax = axes[1, 0]
    order_r = np.argsort(imp_reg.mean(axis=1))[::-1]
    bp_data = [imp_reg[i] for i in order_r]
    bp = ax.boxplot(bp_data, vert=False, patch_artist=True)
    pal = plt.cm.viridis(np.linspace(0.3, 0.9, len(order_r)))
    for patch, color in zip(bp["boxes"], pal):
        patch.set_facecolor(color); patch.set_alpha(0.8)
    for element in ["whiskers", "caps", "medians", "fliers"]:
        for line in bp[element]:
            line.set_color("#aaa")
    ax.set_yticks(range(1, len(order_r) + 1))
    ax.set_yticklabels([feature_names_reg[i] for i in order_r], color="white")
    ax.axvline(0, color="#e74c3c", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_title("回归 RF 排列重要性分布", color="white", pad=8)
    ax.set_xlabel("R² Drop", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图5：排列重要性 vs RF 内置重要性对比 ───────────────────
    ax = axes[1, 1]
    perm_mean = imp_reg.mean(axis=1)
    rf_imp    = rf_reg.feature_importances_
    perm_norm = perm_mean / (perm_mean.sum() + 1e-10)
    rf_norm   = rf_imp / rf_imp.sum()
    x_pos = np.arange(len(feature_names_reg))
    w = 0.35
    ax.bar(x_pos - w/2, perm_norm, w, color="#e74c3c", alpha=0.85, label="排列重要性")
    ax.bar(x_pos + w/2, rf_norm,   w, color="#3498db", alpha=0.85, label="RF 内置重要性")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(feature_names_reg, color="white")
    ax.set_title("排列重要性 vs RF 内置重要性", color="white", pad=8)
    ax.set_ylabel("归一化重要性", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图6：方法对比总结 ──────────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["方法", "模型无关", "考虑交互", "计算速度", "偏差"],
        ["排列重要性", "是", "是", "慢", "相关特征稀释"],
        ["RF内置重要性", "否(树)", "部分", "快", "高基数偏差"],
        ["SHAP", "是", "是", "很慢", "近似误差"],
        ["LIME", "是", "局部", "中等", "邻域不真实"],
        ["权重系数", "否(线性)", "否", "很快", "共线性"],
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center",
                   colWidths=[0.28, 0.18, 0.18, 0.18, 0.28])
    tbl.auto_set_font_size(False); tbl.set_fontsize(7.5)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white"); cell.set_edgecolor("#334")
    ax.set_title("特征重要性方法对比", color="white", pad=10)

    plt.suptitle("排列特征重要性 (Permutation Feature Importance)",
                 color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("permutation_importance.png"))

    print("\n=== 分类任务排列重要性 (RF) ===")
    rf_imp_cls = pi_results_cls["RandomForest"]["imp"].mean(axis=1)
    for i in np.argsort(rf_imp_cls)[::-1]:
        print(f"  {feature_names_cls[i]}: {rf_imp_cls[i]:.4f}")

    print("\n[DONE] 排列特征重要性完成!")


if __name__ == "__main__":
    permutation_importance_main()
