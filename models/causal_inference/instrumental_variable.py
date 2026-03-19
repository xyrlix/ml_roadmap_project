"""
工具变量法 (Instrumental Variable, IV)
=======================================
因果推断中处理内生性（混淆因素/反向因果）的经典方法，纯 NumPy 实现

核心思想：
  当处理变量 D 与误差项相关（存在混淆）时，OLS 估计因果效应有偏。
  工具变量 Z 需满足：
    1. 相关性（Relevance）:    Z 与 D 相关 (Cov(Z,D) ≠ 0)
    2. 外生性（Exogeneity）:   Z 与误差项 ε 不相关 (Cov(Z,ε) = 0)
    3. 排他性（Exclusivity）:  Z 只通过 D 影响 Y（无直接路径）

估计方法：
  - 两阶段最小二乘 (2SLS / TSLS)
  - Wald 估计量（单一二元 IV 的简化估计）

示例场景：教育回报率估计
  - Y  = 工资（对数）
  - D  = 受教育年限（内生：聪明的人既赚得多又受教育更多）
  - Z  = 出生季度（Angrist & Krueger 1991 经典 IV：出生在年初者强制上学更早）
  - X  = 控制变量（年龄、地区等）
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ─────────────────────── 数据生成 ───────────────────────────────

def generate_iv_data(n: int = 2000, true_beta: float = 0.08,
                     seed: int = 42) -> dict:
    """
    生成教育-工资 IV 场景数据（带混淆变量 ability）

    数据生成过程 (DGP)：
        Z ~ Bernoulli(0.5)          工具变量（出生季度代理）
        ability ~ N(0,1)            不可观测能力（混淆）
        X ~ N(0,1)                  可观测控制变量（年龄等）
        D = 10 + 2*Z + 3*ability + 0.5*X + ε_D   受教育年限（内生）
        Y = 5 + β*D + 2*ability + 0.3*X + ε_Y    工资（对数）

    OLS 会高估 β（能力越高同时受教育更多、工资更高，正向混淆）
    """
    rng = np.random.default_rng(seed)
    Z       = rng.binomial(1, 0.5, n).astype(float)
    ability = rng.normal(0, 1, n)
    X       = rng.normal(0, 1, n)
    eps_D   = rng.normal(0, 1, n)
    eps_Y   = rng.normal(0, 1, n)

    D = 10 + 2 * Z + 3 * ability + 0.5 * X + eps_D
    Y = 5 + true_beta * D + 2 * ability + 0.3 * X + eps_Y

    return {
        "Y": Y, "D": D, "Z": Z, "X": X,
        "ability": ability,  # 不可观测，仅用于验证
        "true_beta": true_beta,
        "n": n
    }


# ─────────────────────── OLS 估计 ───────────────────────────────

def ols(Y: np.ndarray, X_mat: np.ndarray) -> tuple:
    """
    OLS 估计: β = (X'X)^{-1} X'Y
    返回 (coef, se, t_stats, p_values)
    """
    n, k = X_mat.shape
    beta = np.linalg.lstsq(X_mat, Y, rcond=None)[0]
    resid = Y - X_mat @ beta
    sigma2 = (resid ** 2).sum() / (n - k)
    cov_beta = sigma2 * np.linalg.pinv(X_mat.T @ X_mat)
    se    = np.sqrt(np.diag(cov_beta))
    t     = beta / (se + 1e-10)
    p     = 2 * (1 - stats.t.cdf(np.abs(t), df=n - k))
    return beta, se, t, p


# ─────────────────────── 两阶段最小二乘 (2SLS) ──────────────────

def tsls(Y: np.ndarray, D: np.ndarray, Z: np.ndarray,
         X_exog: np.ndarray = None) -> dict:
    """
    2SLS 估计量：
      第一阶段: D ~ Z + X   → 得到 D_hat
      第二阶段: Y ~ D_hat + X  → 因果效应

    参数:
      Y      : 结果变量 (n,)
      D      : 内生处理变量 (n,)
      Z      : 工具变量 (n,) 或 (n, n_iv)
      X_exog : 外生控制变量 (n, k)（可为 None）
    """
    n = len(Y)
    ones = np.ones(n)

    # 构造设计矩阵
    if X_exog is None:
        Z1_mat = np.column_stack([ones, Z])   # 第一阶段：[1, Z]
        X2_mat = np.column_stack([ones])       # 第二阶段控制变量 placeholder
    else:
        Z1_mat = np.column_stack([ones, Z, X_exog])
        X2_mat = np.column_stack([ones, X_exog])

    # ── 第一阶段：D ~ Z + X ──
    beta1, se1, t1, p1 = ols(D, Z1_mat)
    D_hat = Z1_mat @ beta1
    resid1 = D - D_hat

    # F-stat（工具变量强度检验）
    # H0: Z 的系数 = 0（弱工具变量检验）
    n1, k1 = Z1_mat.shape
    n_iv   = 1 if Z.ndim == 1 else Z.shape[1]
    k_exog = k1 - n_iv - 1  # 控制变量数
    rss_r  = (D - (np.column_stack([ones] if X_exog is None else
                                    [ones, X_exog]) @
                    ols(D, np.column_stack([ones] if X_exog is None else
                                            [ones, X_exog]))[0]) ** 2).sum()
    rss_u  = resid1 @ resid1
    f_stat = ((rss_r - rss_u) / n_iv) / (rss_u / (n1 - k1))

    # ── 第二阶段：Y ~ D_hat + X ──
    if X_exog is None:
        X2_second = np.column_stack([ones, D_hat])
    else:
        X2_second = np.column_stack([ones, D_hat, X_exog])
    beta2, _, _, _ = ols(Y, X2_second)

    # 修正标准误（使用实际残差而非 D_hat 残差）
    if X_exog is None:
        X2_actual = np.column_stack([ones, D])
    else:
        X2_actual = np.column_stack([ones, D, X_exog])
    resid2_actual = Y - X2_actual @ beta2
    sigma2_iv = (resid2_actual ** 2).sum() / (n - X2_second.shape[1])
    # IV/2SLS 协方差矩阵（Sandwich）
    X2H = X2_second
    cov_iv = sigma2_iv * np.linalg.pinv(X2H.T @ X2H)
    se_iv = np.sqrt(np.diag(cov_iv))
    t_iv  = beta2 / (se_iv + 1e-10)
    p_iv  = 2 * (1 - stats.t.cdf(np.abs(t_iv), df=n - X2_second.shape[1]))

    return {
        "beta_iv":   beta2[1],       # D 的系数（因果效应）
        "se_iv":     se_iv[1],
        "t_iv":      t_iv[1],
        "p_iv":      p_iv[1],
        "first_stage_f": f_stat,
        "first_stage_beta": beta1,
        "D_hat":     D_hat,
        "resid1":    resid1,
        "resid2":    resid2_actual,
    }


# ─────────────────────── Wald 估计量 ────────────────────────────

def wald_estimator(Y: np.ndarray, D: np.ndarray, Z: np.ndarray) -> dict:
    """
    Wald 估计量（仅适用于单一二元工具变量）：
      β_Wald = Cov(Z, Y) / Cov(Z, D) = E[Y|Z=1]-E[Y|Z=0] / E[D|Z=1]-E[D|Z=0]
    """
    mask1 = Z == 1
    mask0 = Z == 0
    Ey1  = Y[mask1].mean()
    Ey0  = Y[mask0].mean()
    Ed1  = D[mask1].mean()
    Ed0  = D[mask0].mean()
    beta_wald = (Ey1 - Ey0) / (Ed1 - Ed0 + 1e-10)

    # Delta method for SE
    n1, n0 = mask1.sum(), mask0.sum()
    var_num = Y[mask1].var() / n1 + Y[mask0].var() / n0
    var_den = D[mask1].var() / n1 + D[mask0].var() / n0
    denom   = (Ed1 - Ed0) ** 2 + 1e-10
    se_wald = np.sqrt(var_num / denom + (beta_wald ** 2) * var_den / denom)

    return {
        "beta_wald": beta_wald,
        "se_wald":   se_wald,
        "E_Y_Z1": Ey1, "E_Y_Z0": Ey0,
        "E_D_Z1": Ed1, "E_D_Z0": Ed0,
    }


# ─────────────────────── 可视化 ─────────────────────────────────

def plot_results(data: dict, ols_res: dict, tsls_res: dict, wald_res: dict,
                 save_path: str = "results/instrumental_variable_results.png"):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")
    Y, D, Z, X = data["Y"], data["D"], data["Z"], data["X"]
    true_beta   = data["true_beta"]

    # ── 1. 散点图：OLS vs IV 估计 ──
    ax1 = fig.add_subplot(2, 3, 1)
    # 原始散点（教育 vs 工资）
    ax1.scatter(D[Z==0], Y[Z==0], alpha=0.15, s=10, color="#4ECDC4", label="Z=0")
    ax1.scatter(D[Z==1], Y[Z==1], alpha=0.15, s=10, color="#FF6B6B", label="Z=1")
    # OLS 拟合线
    x_range = np.linspace(D.min(), D.max(), 100)
    y_ols   = ols_res["intercept"] + ols_res["beta"] * x_range
    y_iv    = ols_res["intercept"] + tsls_res["beta_iv"] * x_range
    ax1.plot(x_range, y_ols, color="#FFA07A", linewidth=2, label=f"OLS β={ols_res['beta']:.3f}")
    ax1.plot(x_range, y_iv,  color="#98D8C8",  linewidth=2, linestyle="--",
             label=f"IV β={tsls_res['beta_iv']:.3f}")
    ax1.axhline(0, color="#444", linewidth=0.5)
    ax1.set_title("Education vs Log-Wage", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Years of Education (D)", color="white")
    ax1.set_ylabel("Log Wage (Y)", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    ax1.legend(facecolor="#0f3460", labelcolor="white", fontsize=7)
    for s in ax1.spines.values(): s.set_edgecolor("#444")

    # ── 2. 第一阶段：Z → D ──
    ax2 = fig.add_subplot(2, 3, 2)
    means_d = [D[Z==0].mean(), D[Z==1].mean()]
    sems_d  = [D[Z==0].std()/np.sqrt((Z==0).sum()),
               D[Z==1].std()/np.sqrt((Z==1).sum())]
    ax2.bar([0, 1], means_d, color=["#4ECDC4", "#FF6B6B"], alpha=0.85, width=0.4)
    ax2.errorbar([0, 1], means_d, yerr=sems_d, fmt="none", color="white", capsize=5)
    ax2.set_xticks([0, 1])
    ax2.set_xticklabels(["Z=0 (late birth)", "Z=1 (early birth)"], color="white", fontsize=9)
    ax2.set_title(f"1st Stage: Z → D  (F={tsls_res['first_stage_f']:.1f})",
                  color="white", fontsize=10, fontweight="bold")
    ax2.set_ylabel("Mean Education (years)", color="white")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    for s in ax2.spines.values(): s.set_edgecolor("#444")
    ax2.set_ylim(min(means_d)*0.95, max(means_d)*1.05)

    # ── 3. 估计量对比 ──
    ax3 = fig.add_subplot(2, 3, 3)
    estimates = {
        "True β": true_beta,
        "OLS":    ols_res["beta"],
        "2SLS":   tsls_res["beta_iv"],
        "Wald":   wald_res["beta_wald"],
    }
    ses = {
        "True β": 0,
        "OLS":    ols_res["se"],
        "2SLS":   tsls_res["se_iv"],
        "Wald":   wald_res["se_wald"],
    }
    colors  = ["#888", "#FF6B6B", "#4ECDC4", "#45B7D1"]
    names   = list(estimates.keys())
    vals    = list(estimates.values())
    se_vals = list(ses.values())
    ax3.barh(names, vals, xerr=se_vals, color=colors, alpha=0.85,
             error_kw={"color": "white", "capsize": 5})
    ax3.axvline(true_beta, color="#FFA07A", linestyle="--",
                linewidth=2, label=f"True β={true_beta}")
    ax3.set_title("Causal Effect Estimates", color="white", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Coefficient (Return to Education)", color="white")
    ax3.set_facecolor("#16213e")
    ax3.tick_params(colors="white")
    for s in ax3.spines.values(): s.set_edgecolor("#444")
    ax3.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)

    # ── 4. OLS 残差 vs 能力（显示内生性）──
    ax4 = fig.add_subplot(2, 3, 4)
    ax4.scatter(data["ability"], ols_res["residuals"], alpha=0.15, s=8,
                color="#FF6B6B", label="OLS resid")
    ax4.axhline(0, color="white", linewidth=0.8)
    rho_ols = np.corrcoef(data["ability"], ols_res["residuals"])[0, 1]
    ax4.set_title(f"OLS Residual vs Ability\n(ρ={rho_ols:.3f} → OLS 内生性!)",
                  color="white", fontsize=9, fontweight="bold")
    ax4.set_xlabel("Unobserved Ability", color="white")
    ax4.set_ylabel("OLS Residual", color="white")
    ax4.set_facecolor("#16213e")
    ax4.tick_params(colors="white")
    for s in ax4.spines.values(): s.set_edgecolor("#444")
    ax4.grid(True, alpha=0.15)

    # ── 5. 2SLS 残差 vs 能力 ──
    ax5 = fig.add_subplot(2, 3, 5)
    ax5.scatter(data["ability"], tsls_res["resid2"], alpha=0.15, s=8,
                color="#4ECDC4", label="IV resid")
    ax5.axhline(0, color="white", linewidth=0.8)
    rho_iv = np.corrcoef(data["ability"], tsls_res["resid2"])[0, 1]
    ax5.set_title(f"IV Residual vs Ability\n(ρ={rho_iv:.3f} → IV 有效消除偏差)",
                  color="white", fontsize=9, fontweight="bold")
    ax5.set_xlabel("Unobserved Ability", color="white")
    ax5.set_ylabel("IV Residual", color="white")
    ax5.set_facecolor("#16213e")
    ax5.tick_params(colors="white")
    for s in ax5.spines.values(): s.set_edgecolor("#444")
    ax5.grid(True, alpha=0.15)

    # ── 6. Hausmann 检验说明 ──
    ax6 = fig.add_subplot(2, 3, 6)
    ax6.axis("off")
    ax6.set_facecolor("#16213e")

    # 简单 Hausman 检验统计量
    hausman_t  = (ols_res["beta"] - tsls_res["beta_iv"]) / \
                  max(1e-10, np.sqrt(abs(tsls_res["se_iv"]**2 - ols_res["se"]**2)))
    hausman_p  = 2 * (1 - stats.norm.cdf(abs(hausman_t)))

    info = [
        "Instrumental Variable Results",
        "═" * 32,
        f"True Causal Effect (β): {true_beta:.4f}",
        "",
        "OLS (naïve, biased):",
        f"  β = {ols_res['beta']:.4f} ± {ols_res['se']:.4f}",
        f"  Bias = {ols_res['beta']-true_beta:+.4f}",
        "",
        "2SLS (corrected):",
        f"  β = {tsls_res['beta_iv']:.4f} ± {tsls_res['se_iv']:.4f}",
        f"  Bias = {tsls_res['beta_iv']-true_beta:+.4f}",
        f"  1st-stage F = {tsls_res['first_stage_f']:.1f}",
        f"  (F > 10 → strong instrument)",
        "",
        "Wald Estimator:",
        f"  β = {wald_res['beta_wald']:.4f} ± {wald_res['se_wald']:.4f}",
        "",
        "Hausman Test (OLS vs IV):",
        f"  H-stat = {hausman_t:.2f}, p = {hausman_p:.4f}",
        f"  {'→ Endogeneity confirmed (reject OLS)' if hausman_p < 0.05 else '→ OLS acceptable'}",
    ]
    ax6.text(0.03, 0.97, "\n".join(info), transform=ax6.transAxes,
             fontsize=8.5, color="white", va="top", fontfamily="monospace",
             bbox=dict(boxstyle="round", facecolor="#0f3460", alpha=0.6))

    plt.suptitle("Instrumental Variable (IV) — 2SLS Causal Inference",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ─────────────────────── 主函数 ─────────────────────────────────

def instrumental_variable():
    """工具变量法完整流程"""
    print("=" * 60)
    print("   工具变量法 (Instrumental Variable, IV) — 2SLS")
    print("=" * 60)

    # ── 1. 生成数据 ──
    print("\n[1/4] 生成教育-工资 IV 场景数据...")
    data = generate_iv_data(n=2000, true_beta=0.08, seed=42)
    Y, D, Z, X = data["Y"], data["D"], data["Z"], data["X"]
    n   = data["n"]
    print(f"  样本量: {n}")
    print(f"  真实因果效应 (β_true): {data['true_beta']:.4f}")
    print(f"  教育均值: {D.mean():.2f}, 工资均值: {Y.mean():.2f}")
    print(f"  工具变量 Z=1 比例: {Z.mean():.3f}")

    # ── 2. OLS（有偏基准）──
    print("\n[2/4] OLS 估计（有偏，未控制能力）...")
    ones = np.ones(n)
    X_ols = np.column_stack([ones, D, X])
    beta_ols, se_ols, t_ols, p_ols = ols(Y, X_ols)
    print(f"  OLS β_D = {beta_ols[1]:.4f} ± {se_ols[1]:.4f}  "
          f"(t={t_ols[1]:.2f}, p={p_ols[1]:.4f})")
    print(f"  偏差: {beta_ols[1] - data['true_beta']:+.4f}  "
          f"(OLS 高估了真实效应！)")

    ols_res = {
        "beta":      beta_ols[1],
        "se":        se_ols[1],
        "intercept": beta_ols[0],
        "residuals": Y - X_ols @ beta_ols,
    }

    # ── 3. 2SLS ──
    print("\n[3/4] 2SLS 估计...")
    tsls_res = tsls(Y, D, Z, X_exog=X)
    print(f"  第一阶段 F 统计量: {tsls_res['first_stage_f']:.2f}",
          "（>10 为强工具变量）" if tsls_res["first_stage_f"] > 10 else "（弱工具变量风险！）")
    print(f"  2SLS β_D = {tsls_res['beta_iv']:.4f} ± {tsls_res['se_iv']:.4f}  "
          f"(t={tsls_res['t_iv']:.2f}, p={tsls_res['p_iv']:.4f})")
    print(f"  偏差: {tsls_res['beta_iv'] - data['true_beta']:+.4f}")

    # ── 4. Wald 估计量 ──
    print("\n[4/4] Wald 估计量（单二元 IV 的直接估计）...")
    wald_res = wald_estimator(Y, D, Z)
    print(f"  E[D|Z=1] - E[D|Z=0] = {wald_res['E_D_Z1']:.3f} - {wald_res['E_D_Z0']:.3f}"
          f" = {wald_res['E_D_Z1']-wald_res['E_D_Z0']:.3f}")
    print(f"  E[Y|Z=1] - E[Y|Z=0] = {wald_res['E_Y_Z1']:.3f} - {wald_res['E_Y_Z0']:.3f}"
          f" = {wald_res['E_Y_Z1']-wald_res['E_Y_Z0']:.3f}")
    print(f"  Wald β = {wald_res['beta_wald']:.4f} ± {wald_res['se_wald']:.4f}")

    print("\n  方法对比:")
    print(f"  {'方法':<12} {'估计值':>8} {'偏差':>8} {'SE':>8}")
    print("  " + "-" * 38)
    for name, est, se in [
        ("True", data["true_beta"], 0),
        ("OLS",  ols_res["beta"],   ols_res["se"]),
        ("2SLS", tsls_res["beta_iv"], tsls_res["se_iv"]),
        ("Wald", wald_res["beta_wald"], wald_res["se_wald"]),
    ]:
        print(f"  {name:<12} {est:>8.4f} {est-data['true_beta']:>+8.4f} {se:>8.4f}")

    # 可视化
    print("\n生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(data, ols_res, tsls_res, wald_res,
                 save_path="results/instrumental_variable_results.png")

    print("\n[DONE] 工具变量法完成!")
    return tsls_res, wald_res


if __name__ == "__main__":
    instrumental_variable()
