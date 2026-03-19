"""
LIME 模型解释 (Local Interpretable Model-agnostic Explanations)
================================================================
从零实现 LIME，用局部线性代理模型解释黑盒预测

核心思想：
  对某个待解释的样本 x：
  1. 在 x 的邻域内随机扰动生成 N 个样本
  2. 用黑盒模型预测这些样本的标签
  3. 按距离 x 的相似性加权
  4. 拟合一个加权线性回归（代理模型），其系数即为解释

优势：模型无关、直观易懂
局限：局部近似、扰动样本分布可能不真实
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge
from sklearn.metrics import accuracy_score
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── LIME 实现 ───────────────────────────────

class LIMEExplainer:
    """
    LIME：局部可解释模型无关解释

    参数：
      n_samples:    邻域扰动样本数
      kernel_width: 相似性核宽度（高斯核 sigma）
      n_features:   解释使用的最大特征数
    """
    def __init__(self, n_samples=500, kernel_width=0.75, n_features=None):
        self.n_samples    = n_samples
        self.kernel_width = kernel_width
        self.n_features   = n_features

    def _kernel(self, distances):
        """指数核（高斯相似性）：exp(-d^2 / (2*sigma^2))"""
        return np.exp(-(distances ** 2) / (2 * self.kernel_width ** 2))

    def explain_instance(self, x, predict_fn, training_data, random_state=42):
        """
        解释单个样本 x 的预测

        返回：
          coefs:      特征系数（重要性 + 方向）
          intercept:  截距
          local_pred: 线性代理模型的预测
          r2:         代理模型在邻域样本上的 R²
        """
        rng = np.random.default_rng(random_state)
        n_feat = len(x)
        if self.n_features is None:
            self.n_features = n_feat

        # 1. 生成扰动样本（在训练数据分布上扰动）
        std = training_data.std(axis=0) + 1e-8
        perturbations = rng.normal(x, std * 0.1, size=(self.n_samples, n_feat))

        # 2. 黑盒预测（取正类概率）
        preds = predict_fn(perturbations)

        # 3. 计算到 x 的欧氏距离，然后核加权
        distances = np.sqrt(((perturbations - x) ** 2).sum(axis=1))
        distances_norm = distances / distances.std()
        weights = self._kernel(distances_norm)

        # 4. 加权线性回归（Ridge）
        reg = Ridge(alpha=1.0)
        reg.fit(perturbations, preds, sample_weight=weights)

        # 5. 评估局部近似质量
        local_pred = reg.predict(perturbations)
        ss_res = ((weights * (preds - local_pred)**2)).sum()
        ss_tot = ((weights * (preds - preds.mean())**2)).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)

        # 选择最重要的 n_features 个特征（绝对系数最大）
        coefs = reg.coef_.copy()
        if self.n_features < n_feat:
            top_idx = np.argsort(np.abs(coefs))[::-1][:self.n_features]
            mask = np.zeros(n_feat, dtype=bool)
            mask[top_idx] = True
            coefs[~mask] = 0.0

        return coefs, reg.intercept_, reg.predict(x.reshape(1, -1))[0], r2


def lime_explainer():
    print("LIME 模型解释运行中...\n")

    # ── 数据准备 ──────────────────────────────────────────────────
    print("1. 生成贷款违约预测数据集...")
    np.random.seed(42)
    N = 800
    feature_names = [
        "信用评分", "月收入", "负债比率", "工龄",
        "贷款金额", "历史逾期次数", "资产净值", "家庭人口"
    ]
    n_feat = len(feature_names)

    X, y = make_classification(
        n_samples=N, n_features=n_feat, n_informative=6,
        n_redundant=0, n_classes=2, random_state=42, weights=[0.65, 0.35]
    )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # ── 训练黑盒模型 ──────────────────────────────────────────────
    print("2. 训练随机森林（黑盒模型）...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_s, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test_s))
    print(f"   Test Accuracy: {acc:.4f}")

    def predict_fn(X_new):
        return rf.predict_proba(X_new)[:, 1]

    # ── LIME 解释多个样本 ─────────────────────────────────────────
    print("3. LIME 解释测试集前15个样本...")
    lime = LIMEExplainer(n_samples=800, kernel_width=0.5, n_features=n_feat)

    n_explain = 15
    all_coefs = np.zeros((n_explain, n_feat))
    all_r2    = np.zeros(n_explain)
    all_preds = np.zeros(n_explain)

    for i in range(n_explain):
        coefs, intercept, local_p, r2 = lime.explain_instance(
            X_test_s[i], predict_fn, X_train_s, random_state=i
        )
        all_coefs[i] = coefs
        all_r2[i]    = r2
        all_preds[i] = local_p
        true_pred = predict_fn(X_test_s[i:i+1])[0]
        print(f"   样本{i:2d}: 真实预测={true_pred:.3f}  LIME近似={local_p:.3f}  R²={r2:.3f}  y={y_test[i]}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n4. 生成 LIME 可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：单样本 LIME 解释（正/负特征条形图） ───────────────
    for plot_idx, (ax, sample_idx) in enumerate(
        zip([axes[0, 0], axes[0, 1]], [0, 1])
    ):
        coefs = all_coefs[sample_idx]
        true_pred = predict_fn(X_test_s[sample_idx:sample_idx+1])[0]
        order = np.argsort(coefs)
        colors = ["#e74c3c" if c > 0 else "#3498db" for c in coefs[order]]
        ax.barh([feature_names[i] for i in order], coefs[order],
                color=colors, alpha=0.85)
        ax.axvline(0, color="white", linewidth=0.8)
        ax.set_title(
            f"LIME 解释 (样本#{sample_idx})\n"
            f"黑盒预测={true_pred:.3f}, R²={all_r2[sample_idx]:.3f}  y={y_test[sample_idx]}",
            color="white", pad=8, fontsize=9
        )
        ax.set_xlabel("LIME 系数（贡献度）", color="#aaa")
        ax.tick_params(colors="gray")
        for sp in ax.spines.values(): sp.set_color("#444")
        ax.text(0.05, 0.95, "红=增加违约风险\n蓝=降低违约风险",
                transform=ax.transAxes, color="white", fontsize=7, va="top")

    # ── 子图3：所有样本 LIME 热力图 ──────────────────────────────
    ax = axes[0, 2]
    im = ax.imshow(all_coefs.T, aspect="auto", cmap="RdBu_r",
                   interpolation="nearest",
                   vmin=-np.abs(all_coefs).max(), vmax=np.abs(all_coefs).max())
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(feature_names, color="white", fontsize=8)
    ax.set_xlabel("样本编号", color="#aaa")
    ax.set_title("LIME 系数热力图（所有样本）", color="white", pad=8)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im, ax=ax, fraction=0.03).ax.tick_params(colors="gray")

    # ── 子图4：R² 分布（LIME 近似质量） ─────────────────────────
    ax = axes[1, 0]
    ax.bar(range(n_explain), all_r2,
           color=plt.cm.RdYlGn(all_r2), alpha=0.85)
    ax.axhline(0.7, color="white", linestyle="--", alpha=0.7, label="R²=0.7 阈值")
    ax.set_title("LIME 局部近似质量 (R²)", color="white", pad=8)
    ax.set_xlabel("样本编号", color="#aaa"); ax.set_ylabel("R²", color="#aaa")
    ax.tick_params(colors="gray")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图5：LIME 预测 vs 黑盒预测 ────────────────────────────
    ax = axes[1, 1]
    true_preds = np.array([predict_fn(X_test_s[i:i+1])[0] for i in range(n_explain)])
    ax.scatter(true_preds, all_preds, c=y_test[:n_explain], cmap="RdYlGn",
               s=80, alpha=0.85, edgecolors="white", linewidths=0.5)
    lim = [min(true_preds.min(), all_preds.min()) - 0.05,
           max(true_preds.max(), all_preds.max()) + 0.05]
    ax.plot(lim, lim, "--", color="#aaa", alpha=0.7, label="理想对角线")
    ax.set_title("LIME 近似 vs 真实预测", color="white", pad=8)
    ax.set_xlabel("黑盒预测概率", color="#aaa")
    ax.set_ylabel("LIME 近似预测", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图6：全局特征重要性（LIME 系数均值绝对值） ──────────────
    ax = axes[1, 2]
    global_imp = np.abs(all_coefs).mean(axis=0)
    order_g = np.argsort(global_imp)
    pal = plt.cm.plasma(np.linspace(0.3, 0.9, n_feat))
    ax.barh([feature_names[i] for i in order_g], global_imp[order_g], color=pal, alpha=0.85)
    ax.set_title("全局 LIME 特征重要性（平均）", color="white", pad=8)
    ax.set_xlabel("mean |LIME coef|", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    plt.suptitle("LIME 局部可解释模型解释", color="white", fontsize=14,
                 y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("lime_explainer.png"))

    print("\n=== 全局 LIME 特征重要性 ===")
    global_imp = np.abs(all_coefs).mean(axis=0)
    for i in np.argsort(global_imp)[::-1]:
        print(f"  {feature_names[i]:<10}: {global_imp[i]:.4f}")

    print("\n[DONE] LIME 解释完成!")


if __name__ == "__main__":
    lime_explainer()
