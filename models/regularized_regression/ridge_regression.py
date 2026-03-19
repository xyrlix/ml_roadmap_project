# 岭回归 (Ridge Regression)
# L2 正则化回归：J(w) = MSE(w) + α * ||w||²₂

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import get_results_path, save_and_close


def generate_polynomial_data(n_samples=100, degree=3, noise=0.1, seed=42):
    """生成多项式回归数据"""
    np.random.seed(seed)
    X = np.linspace(0, 10, n_samples)
    y = 2 * X - 0.5 * X**2 + 0.05 * X**3 + np.random.normal(0, noise, n_samples)
    return X.reshape(-1, 1), y


# ───────────────────────────── 手写岭回归 ─────────────────────────────

class RidgeRegressionFromScratch:
    """从零实现岭回归（闭式解）"""
    def __init__(self, alpha=1.0):
        self.alpha = alpha  # L2 正则化强度
        self.coef_ = None
        self.intercept_ = None

    def fit(self, X, y):
        """闭式解: w = (XᵀX + αI)⁻¹ Xᵀy"""
        # 添加偏置项（全1列）
        X_with_bias = np.column_stack([X, np.ones(X.shape[0])])
        n_features = X_with_bias.shape[1]

        # (XᵀX + αI)
        XtX = X_with_bias.T @ X_with_bias
        XtX_reg = XtX + self.alpha * np.eye(n_features)

        # 求逆
        XtX_reg_inv = np.linalg.inv(XtX_reg)

        # w = (XᵀX + αI)⁻¹ Xᵀy
        w = XtX_reg_inv @ X_with_bias.T @ y

        self.coef_ = w[:-1]
        self.intercept_ = w[-1]
        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def ridge_regression():
    """岭回归实现（α 敏感性分析）"""
    print("岭回归 (Ridge Regression) 运行中...\n")

    # 1. 数据准备（多项式特征）
    print("1. 准备数据（多项式回归）...")
    X, y = generate_polynomial_data(n_samples=120, degree=3, noise=1.5)

    # 多项式扩展
    poly = PolynomialFeatures(degree=5, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.25, random_state=42)

    # 标准化（正则化对尺度敏感）
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}  特征数: {X_train.shape[1]}")

    # 2. α 敏感性分析
    print("2. α 敏感性分析（α=0~100）...")
    alphas = np.logspace(-3, 2, 30)  # 0.001 ~ 100
    train_mses, test_mses, coefs_norm = [], [], []

    for alpha in alphas:
        model = RidgeRegressionFromScratch(alpha=alpha)
        model.fit(X_train, y_train)
        train_mses.append(mean_squared_error(y_train, model.predict(X_train)))
        test_mses.append(mean_squared_error(y_test, model.predict(X_test)))
        coefs_norm.append(np.linalg.norm(model.coef_))

    best_alpha = alphas[np.argmin(test_mses)]
    print(f"   最优 α = {best_alpha:.4f} (测试集 MSE={min(test_mses):.4f})")

    # 3. 最优 α 训练 & 可视化
    print(f"3. 训练最优 α={best_alpha:.4f} 模型...")
    model_ridge = RidgeRegressionFromScratch(alpha=best_alpha)
    model_ridge.fit(X_train, y_train)
    y_pred_ridge = model_ridge.predict(X_test)

    # OLS 对比（无正则化）
    model_ols = LinearRegression()
    model_ols.fit(X_train, y_train)
    y_pred_ols = model_ols.predict(X_test)

    mse_ridge = mean_squared_error(y_test, y_pred_ridge)
    r2_ridge = r2_score(y_test, y_pred_ridge)
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    r2_ols = r2_score(y_test, y_pred_ols)

    print(f"   Ridge (α={best_alpha:.4f}): MSE={mse_ridge:.4f}, R²={r2_ridge:.4f}")
    print(f"   OLS (α=0)           : MSE={mse_ols:.4f}, R²={r2_ols:.4f}")

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：α vs MSE 曲线 ──
    ax = axes[0]
    ax.semilogx(alphas, train_mses, 'bo-', label='Train MSE', linewidth=1.5)
    ax.semilogx(alphas, test_mses, 'rs-', label='Test MSE', linewidth=1.5)
    ax.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2,
               label=f'Best α={best_alpha:.3f}')
    ax.set_title('岭回归：α 敏感性分析')
    ax.set_xlabel('α (log scale)')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # ── 子图2：α vs 系数范数 ──
    ax = axes[1]
    ax.semilogx(alphas, coefs_norm, 'g-', linewidth=1.5)
    ax.set_title('α vs ||w||₂（L2 范数）')
    ax.set_xlabel('α (log scale)')
    ax.set_ylabel('Coefficient Norm')
    ax.grid(True, alpha=0.3, which='both')

    # ── 子图3：预测曲线对比 ──
    ax = axes[2]
    X_plot = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
    X_plot_poly = poly.transform(X_plot)
    X_plot_scaled = scaler.transform(X_plot_poly)

    y_plot_ridge = model_ridge.predict(X_plot_scaled)
    y_plot_ols = model_ols.predict(X_plot_scaled)

    ax.scatter(X, y, c='steelblue', alpha=0.6, s=30, label='Data')
    ax.plot(X_plot, y_plot_ridge, 'r-', linewidth=2, label=f'Ridge (α={best_alpha:.2f})')
    ax.plot(X_plot, y_plot_ols, 'k--', linewidth=1.5, label='OLS (α=0)')
    ax.set_title('回归预测曲线对比')
    ax.set_xlabel('X')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('ridge_regression_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    ridge_regression()
