"""
特征选择 (Feature Selection)
==============================
从零 + sklearn 实现常用特征选择方法，展示如何找出对模型最有用的特征

实现方法：
  1. Filter 方法：方差过滤、互信息、卡方检验、F-统计量
  2. Wrapper 方法：递归特征消除 (RFE)
  3. Embedded 方法：L1 正则化（Lasso）、树模型特征重要性
  4. 对比实验：各方法选出的特征子集在测试集上的性能
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, mutual_info_classif,
    chi2, f_classif, RFE, SelectFromModel
)
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


def feature_selection():
    print("特征选择方法运行中...\n")

    # ── 生成数据：20个特征，其中只有8个真正有用 ──────────────────
    print("1. 生成含冗余特征的数据集...")
    N_SAMPLES   = 800
    N_FEATURES  = 20
    N_INFORMATIVE = 8
    N_REDUNDANT   = 4
    N_REPEATED    = 2   # 复制特征
    N_USELESS     = N_FEATURES - N_INFORMATIVE - N_REDUNDANT - N_REPEATED

    X, y = make_classification(
        n_samples=N_SAMPLES, n_features=N_FEATURES,
        n_informative=N_INFORMATIVE, n_redundant=N_REDUNDANT,
        n_repeated=N_REPEATED, n_classes=2, random_state=42
    )
    feat_names = [f"F{i:02d}" for i in range(N_FEATURES)]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)
    X_pos = X - X.min(axis=0)  # 卡方检验需要非负

    print(f"   数据: {N_SAMPLES}样本 x {N_FEATURES}特征  "
          f"(有效:{N_INFORMATIVE}, 冗余:{N_REDUNDANT}, 复制:{N_REPEATED}, 噪声:{N_USELESS})")

    K = 8  # 选 8 个特征

    # ── 各方法特征选择 ────────────────────────────────────────────
    print("2. 运行各种特征选择方法...")

    # 1) 方差过滤
    var_sel = VarianceThreshold(threshold=0.5)
    var_sel.fit(X_train_s)
    var_support = var_sel.get_support()

    # 2) 互信息
    mi_sel = SelectKBest(mutual_info_classif, k=K)
    mi_sel.fit(X_train, y_train)
    mi_scores = mi_sel.scores_

    # 3) F-统计量
    f_sel = SelectKBest(f_classif, k=K)
    f_sel.fit(X_train, y_train)
    f_scores = f_sel.scores_

    # 4) 卡方检验（非负）
    X_pos_train = X_train - X_train.min(axis=0)
    chi2_sel = SelectKBest(chi2, k=K)
    chi2_sel.fit(X_pos_train, y_train)
    chi2_scores = chi2_sel.scores_

    # 5) RFE（递归特征消除）
    rfe_estimator = LogisticRegression(max_iter=500, random_state=42)
    rfe_sel = RFE(rfe_estimator, n_features_to_select=K, step=1)
    rfe_sel.fit(X_train_s, y_train)
    rfe_support = rfe_sel.support_
    rfe_ranking = rfe_sel.ranking_

    # 6) L1 Lasso 嵌入式
    lasso = LogisticRegression(C=0.1, penalty="l1", solver="saga",
                                max_iter=1000, random_state=42)
    l1_sel = SelectFromModel(lasso, prefit=False, max_features=K)
    l1_sel.fit(X_train_s, y_train)
    l1_support = l1_sel.get_support()

    # 7) 树模型特征重要性
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_sel = SelectFromModel(rf, prefit=False, max_features=K)
    rf_sel.fit(X_train, y_train)
    rf_support = rf_sel.get_support()
    rf.fit(X_train, y_train)
    rf_importances = rf.feature_importances_

    # ── 各方法性能评估 ────────────────────────────────────────────
    print("3. 评估各特征集上的分类性能...")

    def eval_features(support, X_tr, X_te, y_tr, y_te, name):
        if support.sum() == 0:
            return 0.5
        clf = LogisticRegression(max_iter=500, random_state=42)
        clf.fit(X_tr[:, support], y_tr)
        return accuracy_score(y_te, clf.predict(X_te[:, support]))

    methods = {
        "All Features":    np.ones(N_FEATURES, dtype=bool),
        "Variance":        var_support,
        "Mutual Info":     mi_sel.get_support(),
        "F-Statistic":     f_sel.get_support(),
        "Chi2":            chi2_sel.get_support(),
        "RFE":             rfe_support,
        "L1 (Lasso)":      l1_support,
        "RandomForest":    rf_support,
    }

    accs = {}
    for name, support in methods.items():
        acc = eval_features(support, X_train_s, X_test_s, y_train, y_test, name)
        accs[name] = acc
        print(f"   {name:<16}: {support.sum():2d} 特征 -> Acc={acc:.4f}")

    # ── 可视化 ───────────────────────────────────────────────────
    print("4. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # 归一化得分辅助
    def normalize(scores):
        mn, mx = scores.min(), scores.max()
        return (scores - mn) / (mx - mn + 1e-12)

    # ── 子图1：互信息得分 ────────────────────────────────────────
    ax = axes[0, 0]
    order = np.argsort(mi_scores)[::-1]
    colors = ["#e74c3c" if mi_sel.get_support()[i] else "#555" for i in order]
    ax.bar(range(N_FEATURES), mi_scores[order], color=colors, alpha=0.85)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels([feat_names[i] for i in order], rotation=45, ha="right",
                        color="white", fontsize=7)
    ax.set_title("互信息得分 (MI)", color="white", pad=10)
    ax.set_ylabel("MI Score", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    ax.text(0.65, 0.92, "红=被选中", transform=ax.transAxes, color="#e74c3c", fontsize=8)

    # ── 子图2：F-统计量得分 ──────────────────────────────────────
    ax = axes[0, 1]
    order_f = np.argsort(f_scores)[::-1]
    colors_f = ["#3498db" if f_sel.get_support()[i] else "#555" for i in order_f]
    ax.bar(range(N_FEATURES), f_scores[order_f], color=colors_f, alpha=0.85)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels([feat_names[i] for i in order_f], rotation=45, ha="right",
                        color="white", fontsize=7)
    ax.set_title("F-统计量得分", color="white", pad=10)
    ax.set_ylabel("F Score", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图3：RF 特征重要性 ─────────────────────────────────────
    ax = axes[0, 2]
    order_rf = np.argsort(rf_importances)[::-1]
    colors_rf = ["#2ecc71" if rf_support[i] else "#555" for i in order_rf]
    ax.bar(range(N_FEATURES), rf_importances[order_rf], color=colors_rf, alpha=0.85)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels([feat_names[i] for i in order_rf], rotation=45, ha="right",
                        color="white", fontsize=7)
    ax.set_title("随机森林特征重要性", color="white", pad=10)
    ax.set_ylabel("Importance", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：方法精度对比 ──────────────────────────────────────
    ax = axes[1, 0]
    names_acc = list(accs.keys())
    vals_acc  = [accs[n] for n in names_acc]
    pal = plt.cm.viridis(np.linspace(0.3, 0.9, len(names_acc)))
    bars = ax.barh(names_acc, vals_acc, color=pal, alpha=0.85)
    ax.set_title("各特征选择方法分类准确率", color="white", pad=10)
    ax.set_xlabel("Test Accuracy", color="#aaa")
    ax.set_xlim(0.4, 1.0)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, vals_acc):
        ax.text(v + 0.002, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", color="white", fontsize=8)

    # ── 子图5：RFE 排名热力图 ────────────────────────────────────
    ax = axes[1, 1]
    rank_norm = 1 / rfe_ranking
    ax.bar(feat_names, rank_norm,
           color=["#f39c12" if rfe_support[i] else "#555" for i in range(N_FEATURES)],
           alpha=0.85)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", color="white", fontsize=7)
    ax.set_title("RFE 特征排名（越高越重要）", color="white", pad=10)
    ax.set_ylabel("1 / Rank", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图6：选择重叠热力图 ────────────────────────────────────
    ax = axes[1, 2]
    method_names = ["MI", "F-stat", "Chi2", "RFE", "L1", "RF"]
    support_matrix = np.array([
        mi_sel.get_support().astype(int),
        f_sel.get_support().astype(int),
        chi2_sel.get_support().astype(int),
        rfe_support.astype(int),
        l1_support.astype(int),
        rf_support.astype(int),
    ])
    im = ax.imshow(support_matrix, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")
    ax.set_yticks(range(len(method_names)))
    ax.set_yticklabels(method_names, color="white", fontsize=9)
    ax.set_xticks(range(N_FEATURES))
    ax.set_xticklabels(feat_names, rotation=45, ha="right", color="white", fontsize=7)
    ax.set_title("特征选择一致性热力图", color="white", pad=10)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).ax.tick_params(colors="gray")

    plt.suptitle("特征选择方法对比", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("feature_selection.png"))

    print("\n[DONE] 特征选择完成!")


if __name__ == "__main__":
    feature_selection()
