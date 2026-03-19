"""
倾向评分匹配 (Propensity Score Matching, PSM)
=============================================
观察性研究中通过匹配消除选择偏差，估计平均处理效应（ATE / ATT），纯 NumPy 实现

核心思想：
  在观察性数据中，处理组和对照组可能系统性不同（选择偏差）。
  PSM 通过以下步骤模拟随机实验：
    1. 估计倾向评分：P(T=1 | X)（用 Logistic 回归）
    2. 按倾向评分匹配处理组和对照组（1:1 最近邻匹配）
    3. 检验协变量平衡性（匹配前后标准化差异）
    4. 在匹配样本上估计平均处理效应 (ATT)

示例场景：评估"职业培训项目"对收入的因果效应
  - T = 1 参加培训, T = 0 未参加
  - Y = 收入（对数）
  - X = 年龄、教育年限、性别、婚姻状态等
  - 选择偏差：本来就有动力的人更可能参加培训并赚更多
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ─────────────────────── 数据生成 ───────────────────────────────

COVARIATE_NAMES = ["age", "education", "female", "married",
                   "prior_income", "experience"]


def generate_psm_data(n: int = 1500, true_att: float = 0.15,
                      seed: int = 42) -> dict:
    """
    生成职业培训效应估计数据

    DGP:
      X = [age, education, female, married, prior_income, experience]
      P(T=1|X) = logistic(α₀ + α·X)   倾向评分
      T ~ Bernoulli(P)
      Y(0) = β₀ + β·X + ε              潜在结果（未接受培训）
      Y(1) = Y(0) + true_ATT + γ·female （异质性处理效应）
      Y_obs = T*Y(1) + (1-T)*Y(0)
    """
    rng = np.random.default_rng(seed)
    n   = n

    # 协变量
    age         = rng.normal(35, 8, n).clip(18, 65)
    education   = rng.normal(12, 3, n).clip(6, 20)
    female      = rng.binomial(1, 0.45, n).astype(float)
    married     = rng.binomial(1, 0.55, n).astype(float)
    prior_income = rng.normal(30000, 8000, n).clip(10000, 80000)
    experience  = rng.normal(8, 5, n).clip(0, 40)

    X_raw = np.column_stack([age, education, female, married,
                              prior_income, experience])

    # 倾向评分（有偏的参与决策：年轻、教育少、低收入者更可能参加）
    logit_p = (-0.5
               - 0.02 * (age - 35)
               + 0.05 * (education - 12)
               - 0.10 * female
               + 0.00 * married
               - 0.000015 * (prior_income - 30000)
               + 0.01 * experience
               + rng.normal(0, 0.5, n))

    true_ps = 1 / (1 + np.exp(-logit_p))
    T = rng.binomial(1, true_ps).astype(float)

    # 潜在结果
    Y0 = (9.5
          + 0.005 * (age - 35)
          + 0.04  * (education - 12)
          - 0.05  * female
          + 0.03  * married
          + 0.000003 * (prior_income - 30000)
          + 0.008 * experience
          + rng.normal(0, 0.3, n))

    Y1 = Y0 + true_att + 0.05 * female   # 异质性处理效应

    Y_obs = np.where(T == 1, Y1, Y0)

    # 标准化协变量（供 Logistic 使用）
    X_mean = X_raw.mean(axis=0)
    X_std  = X_raw.std(axis=0) + 1e-8
    X      = (X_raw - X_mean) / X_std

    return {
        "Y": Y_obs, "T": T, "X": X, "X_raw": X_raw,
        "true_ps": true_ps, "true_att": true_att,
        "n": n, "X_mean": X_mean, "X_std": X_std,
    }


# ─────────────────────── Logistic 回归（倾向评分估计）────────────

class LogisticRegression:
    def __init__(self, lr: float = 0.1, n_epochs: int = 100, l2: float = 1e-3,
                 seed: int = 42):
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.rng = np.random.default_rng(seed)
        self.W = None
        self.b = None
        self.losses = []

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -50, 50)))

    def fit(self, X: np.ndarray, y: np.ndarray):
        n, d = X.shape
        self.W = self.rng.normal(0, 0.01, d).astype(np.float32)
        self.b = 0.0
        self.losses = []

        for epoch in range(self.n_epochs):
            z     = X @ self.W + self.b
            probs = self._sigmoid(z)
            loss  = -np.mean(y * np.log(probs + 1e-10) +
                             (1 - y) * np.log(1 - probs + 1e-10))
            self.losses.append(loss)
            residuals = probs - y
            self.W -= self.lr * (X.T @ residuals / n + self.l2 * self.W)
            self.b -= self.lr * residuals.mean()
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._sigmoid(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X) >= 0.5).astype(int)


# ─────────────────────── 最近邻匹配 ─────────────────────────────

def nearest_neighbor_matching(ps: np.ndarray, T: np.ndarray,
                               caliper: float = None) -> tuple:
    """
    1:1 最近邻匹配（有放回，按倾向评分距离）
    返回匹配对的 (treated_idx, control_idx)
    """
    treated_idx = np.where(T == 1)[0]
    control_idx = np.where(T == 0)[0]

    matched_treated = []
    matched_control = []
    used_controls   = set()   # 不放回匹配

    for ti in treated_idx:
        ps_t = ps[ti]
        dists = np.abs(ps[control_idx] - ps_t)
        # 排除已用控制组
        order = np.argsort(dists)
        for rank in order:
            ci = control_idx[rank]
            if ci in used_controls:
                continue
            dist = dists[rank]
            if caliper is not None and dist > caliper:
                break
            matched_treated.append(ti)
            matched_control.append(ci)
            used_controls.add(ci)
            break

    return np.array(matched_treated), np.array(matched_control)


# ─────────────────────── 协变量平衡检验 ─────────────────────────

def standardized_mean_difference(X: np.ndarray, T: np.ndarray,
                                  matched_t: np.ndarray = None,
                                  matched_c: np.ndarray = None) -> dict:
    """
    计算每个协变量的标准化均值差 (SMD)
    SMD < 0.1 通常认为平衡良好
    """
    results_before, results_after = {}, {}

    for j in range(X.shape[1]):
        xj     = X[:, j]
        t_mask = T == 1
        c_mask = T == 0

        # 匹配前
        mean_t = xj[t_mask].mean()
        mean_c = xj[c_mask].mean()
        var_t  = xj[t_mask].var()
        var_c  = xj[c_mask].var()
        pooled_std = np.sqrt((var_t + var_c) / 2 + 1e-10)
        smd_before = abs(mean_t - mean_c) / pooled_std
        results_before[j] = smd_before

        # 匹配后
        if matched_t is not None and matched_c is not None:
            mean_t2 = xj[matched_t].mean()
            mean_c2 = xj[matched_c].mean()
            var_t2  = xj[matched_t].var()
            var_c2  = xj[matched_c].var()
            pooled2 = np.sqrt((var_t2 + var_c2) / 2 + 1e-10)
            smd_after = abs(mean_t2 - mean_c2) / pooled2
            results_after[j] = smd_after

    return results_before, results_after


# ─────────────────────── ATT 估计 ────────────────────────────────

def estimate_att(Y: np.ndarray, matched_treated: np.ndarray,
                 matched_control: np.ndarray) -> dict:
    """平均处理效应（ATT）= 匹配处理组 - 匹配对照组，逐对差值的均值"""
    diffs  = Y[matched_treated] - Y[matched_control]
    att    = diffs.mean()
    se_att = diffs.std() / np.sqrt(len(diffs))
    t_stat = att / (se_att + 1e-10)
    p_val  = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(diffs) - 1))
    ci_low  = att - 1.96 * se_att
    ci_high = att + 1.96 * se_att
    return {
        "att": att, "se": se_att, "t": t_stat,
        "p": p_val, "ci": (ci_low, ci_high),
        "n_matched": len(diffs),
    }


# ─────────────────────── 可视化 ─────────────────────────────────

def plot_results(data: dict, ps_model: LogisticRegression,
                 estimated_ps: np.ndarray,
                 matched_t: np.ndarray, matched_c: np.ndarray,
                 att_result: dict, naive_diff: float,
                 smd_before: dict, smd_after: dict,
                 save_path: str = "results/propensity_score_matching_results.png"):

    fig = plt.figure(figsize=(18, 14))
    fig.patch.set_facecolor("#1a1a2e")

    Y, T = data["Y"], data["T"]
    true_att = data["true_att"]

    # ── 1. 倾向评分分布（匹配前）──
    ax1 = fig.add_subplot(2, 3, 1)
    ax1.hist(estimated_ps[T==1], bins=30, alpha=0.65, color="#FF6B6B",
             label="Treated", density=True)
    ax1.hist(estimated_ps[T==0], bins=30, alpha=0.65, color="#4ECDC4",
             label="Control", density=True)
    ax1.set_title("Propensity Score Distribution\n(Before Matching)",
                  color="white", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Estimated P(T=1|X)", color="white")
    ax1.set_ylabel("Density", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)
    for s in ax1.spines.values(): s.set_edgecolor("#444")

    # ── 2. 倾向评分分布（匹配后）──
    ax2 = fig.add_subplot(2, 3, 2)
    ax2.hist(estimated_ps[matched_t], bins=20, alpha=0.65,
             color="#FF6B6B", label="Matched Treated", density=True)
    ax2.hist(estimated_ps[matched_c], bins=20, alpha=0.65,
             color="#4ECDC4", label="Matched Control", density=True)
    ax2.set_title("Propensity Score Distribution\n(After Matching)",
                  color="white", fontsize=9, fontweight="bold")
    ax2.set_xlabel("Estimated P(T=1|X)", color="white")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    ax2.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)
    for s in ax2.spines.values(): s.set_edgecolor("#444")

    # ── 3. 标准化均值差（Love Plot）──
    ax3 = fig.add_subplot(2, 3, 3)
    smd_b = [smd_before[j] for j in range(len(COVARIATE_NAMES))]
    smd_a = [smd_after[j]  for j in range(len(COVARIATE_NAMES))]
    y_pos = range(len(COVARIATE_NAMES))
    ax3.barh(y_pos, smd_b, height=0.35, color="#FF6B6B", alpha=0.8, label="Before")
    ax3.barh([y + 0.38 for y in y_pos], smd_a, height=0.35,
             color="#4ECDC4", alpha=0.8, label="After")
    ax3.axvline(0.1, color="#FFA07A", linestyle="--", linewidth=1.5, label="SMD=0.1")
    ax3.set_yticks([y + 0.19 for y in y_pos])
    ax3.set_yticklabels(COVARIATE_NAMES, color="white", fontsize=9)
    ax3.set_title("Covariate Balance (Love Plot)\nSMD Before vs After",
                  color="white", fontsize=9, fontweight="bold")
    ax3.set_xlabel("Standardized Mean Difference", color="white")
    ax3.set_facecolor("#16213e")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)
    for s in ax3.spines.values(): s.set_edgecolor("#444")

    # ── 4. ATT 估计量对比 ──
    ax4 = fig.add_subplot(2, 3, 4)
    names  = ["True ATT", "Naive Diff", "PSM ATT"]
    vals   = [true_att, naive_diff, att_result["att"]]
    ses    = [0, 0, att_result["se"]]
    colors = ["#888", "#FF6B6B", "#4ECDC4"]
    bars   = ax4.bar(names, vals, yerr=ses, color=colors, alpha=0.85,
                     error_kw={"color": "white", "capsize": 6})
    ax4.axhline(true_att, color="#FFA07A", linestyle="--",
                linewidth=2, label=f"True ATT={true_att}")
    ax4.set_title("Treatment Effect Estimates", color="white", fontsize=10, fontweight="bold")
    ax4.set_ylabel("Effect on Log-Wage", color="white")
    ax4.set_facecolor("#16213e")
    ax4.tick_params(colors="white")
    ax4.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)
    for s in ax4.spines.values(): s.set_edgecolor("#444")
    for bar, v in zip(bars, vals):
        ax4.text(bar.get_x() + bar.get_width()/2, v + 0.002, f"{v:.4f}",
                 ha="center", color="white", fontsize=9)

    # ── 5. 倾向评分学习曲线 ──
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.plot(ps_model.losses, color="#FF6B6B", linewidth=2)
    ax5.set_title("Propensity Score Model (Logistic) Loss",
                  color="white", fontsize=9, fontweight="bold")
    ax5.set_xlabel("Epoch", color="white")
    ax5.set_ylabel("Binary Cross-Entropy", color="white")
    ax5.set_facecolor("#16213e")
    ax5.tick_params(colors="white")
    for s in ax5.spines.values(): s.set_edgecolor("#444")
    ax5.grid(True, alpha=0.2)

    # ── 6. 结果摘要 ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    ax6.set_facecolor("#16213e")

    pct_balance = (np.mean(list(smd_after.values())) /
                   (np.mean(list(smd_before.values())) + 1e-10)) * 100
    well_balanced = sum(v < 0.1 for v in smd_after.values())

    info = [
        "PSM Results Summary",
        "═" * 32,
        f"Sample size       : {data['n']}",
        f"  Treated (T=1)   : {int(T.sum())}",
        f"  Control (T=0)   : {int((1-T).sum())}",
        f"  Matched pairs   : {att_result['n_matched']}",
        "",
        f"True ATT          : {true_att:.4f}",
        "",
        "Naive Estimator (no matching):",
        f"  Bias = {naive_diff-true_att:+.4f}",
        f"  Diff = {naive_diff:.4f}",
        "",
        "PSM Estimator:",
        f"  ATT  = {att_result['att']:.4f} ± {att_result['se']:.4f}",
        f"  Bias = {att_result['att']-true_att:+.4f}",
        f"  95%CI: [{att_result['ci'][0]:.4f}, {att_result['ci'][1]:.4f}]",
        f"  p-val = {att_result['p']:.4f}",
        "",
        "Balance Check:",
        f"  SMD < 0.1  : {well_balanced}/{len(COVARIATE_NAMES)} covariates",
        f"  Avg SMD ↓  : {np.mean(list(smd_before.values())):.3f} → "
        f"{np.mean(list(smd_after.values())):.3f}",
    ]
    ax6.text(0.03, 0.97, "\n".join(info), transform=ax6.transAxes,
             fontsize=8.5, color="white", va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.6))

    plt.suptitle("Propensity Score Matching (PSM) — Causal Inference in Observational Studies",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ─────────────────────── 主函数 ─────────────────────────────────

def propensity_score_matching():
    """倾向评分匹配完整流程"""
    print("=" * 60)
    print("   倾向评分匹配 (Propensity Score Matching, PSM)")
    print("=" * 60)

    # ── 1. 数据 ──
    print("\n[1/5] 生成职业培训效应观察性数据...")
    data = generate_psm_data(n=1500, true_att=0.15, seed=42)
    Y, T, X = data["Y"], data["T"], data["X"]
    n = data["n"]
    print(f"  总样本: {n}  处理组: {int(T.sum())}  对照组: {int((T==0).sum())}")
    print(f"  真实 ATT: {data['true_att']:.4f}")

    # ── 2. 朴素估计（无匹配）──
    print("\n[2/5] 朴素均值差（忽略选择偏差）...")
    naive_diff = Y[T==1].mean() - Y[T==0].mean()
    print(f"  E[Y|T=1] - E[Y|T=0] = {naive_diff:.4f}")
    print(f"  偏差: {naive_diff - data['true_att']:+.4f}（选择偏差）")

    # ── 3. 估计倾向评分 ──
    print("\n[3/5] Logistic 回归估计倾向评分...")
    ps_model = LogisticRegression(lr=0.5, n_epochs=100, l2=1e-3, seed=42)
    ps_model.fit(X, T)
    estimated_ps = ps_model.predict_proba(X)
    ps_acc = (ps_model.predict(X) == T).mean()
    print(f"  倾向评分模型分类准确率: {ps_acc:.4f}")
    print(f"  倾向评分范围: [{estimated_ps.min():.4f}, {estimated_ps.max():.4f}]")

    # ── 4. 最近邻匹配 ──
    print("\n[4/5] 1:1 最近邻匹配（卡尺 = 0.05 × std(ps)）...")
    caliper = 0.05 * estimated_ps.std()
    matched_t, matched_c = nearest_neighbor_matching(
        estimated_ps, T, caliper=caliper)
    print(f"  成功匹配对数: {len(matched_t)}（处理组共 {int(T.sum())} 个）")

    # 协变量平衡检验
    smd_before, smd_after = standardized_mean_difference(
        X, T, matched_t, matched_c)
    print("\n  协变量平衡检验 (SMD):")
    print(f"  {'协变量':<14} {'匹配前':>8} {'匹配后':>8} {'改善':>8}")
    print("  " + "-" * 42)
    for j, name in enumerate(COVARIATE_NAMES):
        b, a = smd_before[j], smd_after[j]
        print(f"  {name:<14} {b:>8.4f} {a:>8.4f} {b-a:>+8.4f}"
              + (" [DONE]" if a < 0.1 else " !"))

    # ── 5. ATT 估计 ──
    print("\n[5/5] 估计 ATT（平均处理效应）...")
    att_result = estimate_att(Y, matched_t, matched_c)
    print(f"  PSM ATT = {att_result['att']:.4f} ± {att_result['se']:.4f}")
    print(f"  95% CI  : [{att_result['ci'][0]:.4f}, {att_result['ci'][1]:.4f}]")
    print(f"  t-stat  : {att_result['t']:.3f},  p = {att_result['p']:.4f}")
    print(f"  偏差    : {att_result['att'] - data['true_att']:+.4f}")

    # 可视化
    print("\n生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(data, ps_model, estimated_ps,
                 matched_t, matched_c, att_result,
                 naive_diff, smd_before, smd_after,
                 save_path="results/propensity_score_matching_results.png")

    print("\n[DONE] 倾向评分匹配完成!")
    return att_result


if __name__ == "__main__":
    propensity_score_matching()
