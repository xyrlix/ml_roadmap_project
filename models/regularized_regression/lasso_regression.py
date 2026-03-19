# Lasso 回归 (Least Absolute Shrinkage and Selection Operator)
# L1 正则化回归：J(w) = MSE(w) + α * ||w||₁

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from utils import get_results_path, save_and_close


# ───────────────────────────── 手写 Lasso（坐标下降法）────────────────────────────

class LassoRegressionFromScratch:
    """从零实现 Lasso 回归（坐标下降法）"""
    def __init__(self, alpha=1.0, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.coef_ = None
        self.intercept_ = None

    def _soft_threshold(self, x, alpha):
        """软阈值函数：sign(x) * max(|x| - α, 0)"""
        return np.sign(x) * np.maximum(np.abs(x) - alpha, 0)

    def fit(self, X, y):
        """坐标下降法"""
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = np.mean(y)

        # 中心化 y 和 X（减去均值）
        y_centered = y - self.intercept_

        for iteration in range(self.max_iter):
            coef_old = self.coef_.copy()

            for j in range(n_features):
                # 预测当前特征 j 的残差
                y_pred_partial = X @ self.coef_
                residual = y_centered - y_pred_partial

                # 坐标下降更新 w_j
                rho = X[:, j] @ residual
                w_j = self._soft_threshold(rho / (X[:, j] ** 2).sum(), self.alpha)

                self.coef_[j] = w_j

            # 收敛判断
            if np.max(np.abs(self.coef_ - coef_old)) < self.tol:
                break

        return self

    def predict(self, X):
        return X @ self.coef_ + self.intercept_


def lasso_regression():
    """Lasso 回归实现（α 敏感性 + 特征选择）"""
    print("Lasso 回归 (L1 正则化) 运行中...\n")

    # 1. 数据准备（高维稀疏特征）
    print("1. 准备数据（高维多项式回归，模拟特征选择）...")
    from .ridge_regression import generate_polynomial_data
    X, y = generate_polynomial_data(n_samples=120, degree=3, noise=1.5)

    # 高维多项式扩展（模拟过拟合场景）
    poly = PolynomialFeatures(degree=8, include_bias=False)
    X_poly = poly.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_poly, y, test_size=0.25, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    print(f"   训练集: {X_train.shape[0]}  测试集: {X_test.shape[0]}  特征数: {X_train.shape[1]}")

    # 2. α 敏感性分析
    print("2. α 敏感性分析（α=0.001~10）...")
    alphas = np.logspace(-3, 1, 25)  # 0.001 ~ 10
    train_mses, test_mses, coefs_norm = [], [], []
    n_nonzeros = []

    for alpha in alphas:
        model = LassoRegressionFromScratch(alpha=alpha, max_iter=2000, tol=1e-5)
        model.fit(X_train, y_train)
        train_mses.append(mean_squared_error(y_train, model.predict(X_train)))
        test_mses.append(mean_squared_error(y_test, model.predict(X_test)))
        coefs_norm.append(np.linalg.norm(model.coef_))
        n_nonzeros.append((np.abs(model.coef_) > 1e-6).sum())

    best_alpha = alphas[np.argmin(test_mses)]
    print(f"   最优 α = {best_alpha:.4f} (测试集 MSE={min(test_mses):.4f})")

    # 3. 最优 α 训练 & 可视化
    print(f"3. 训练最优 α={best_alpha:.4f} 模型...")
    model_lasso = LassoRegressionFromScratch(alpha=best_alpha, max_iter=2000)
    model_lasso.fit(X_train, y_train)
    y_pred_lasso = model_lasso.predict(X_test)

    # OLS 对比
    model_ols = LinearRegression()
    model_ols.fit(X_train, y_train)
    y_pred_ols = model_ols.predict(X_test)

    mse_lasso = mean_squared_error(y_test, y_pred_lasso)
    r2_lasso = r2_score(y_test, y_pred_lasso)
    mse_ols = mean_squared_error(y_test, y_pred_ols)
    r2_ols = r2_score(y_test, y_pred_ols)

    print(f"   Lasso (α={best_alpha:.4f}): MSE={mse_lasso:.4f}, R²={r2_lasso:.4f}")
    print(f"   OLS   (α=0)          : MSE={mse_ols:.4f}, R²={r2_ols:.4f}")
    print(f"   Lasso 非零系数数: {(np.abs(model_lasso.coef_) > 1e-6).sum()}/{len(model_lasso.coef_)}")

    # 4. 可视化
    print("4. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：α vs MSE 曲线 ──
    ax = axes[0]
    ax.semilogx(alphas, train_mses, 'bo-', label='Train MSE', linewidth=1.5)
    ax.semilogx(alphas, test_mses, 'rs-', label='Test MSE', linewidth=1.5)
    ax.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2,
               label=f'Best α={best_alpha:.3f}')
    ax.set_title('Lasso：α 敏感性分析')
    ax.set_xlabel('α (log scale)')
    ax.set_ylabel('MSE')
    ax.legend()
    ax.grid(True, alpha=0.3, which='both')

    # ── 子图2：α vs 非零系数数（特征选择效果）──
    ax = axes[1]
    ax.semilogx(alphas, n_nonzeros, 'g-', linewidth=1.5, markersize=4)
    ax.set_title('α vs 非零系数数（特征选择）')
    ax.set_xlabel('α (log scale)')
    ax.set_ylabel('Number of Non-zero Coefficients')
    ax.grid(True, alpha=0.3, which='both')

    # ── 子图3：系数绝对值（特征重要性）──
    ax = axes[2]
    feat_names = [f'X^{i}' for i in range(len(model_lasso.coef_))]
    lasso_coefs = np.abs(model_lasso.coef_)
    ols_coefs = np.abs(model_ols.coef_)

    x = np.arange(len(feat_names))
    width = 0.35
    bars1 = ax.bar(x - width/2, ols_coefs, width, label='OLS (α=0)',
                   color='#3498db', alpha=0.85, edgecolor='white')
    bars2 = ax.bar(x + width/2, lasso_coefs, width, label=f'Lasso (α={best_alpha:.2f})',
                   color='#e74c3c', alpha=0.85, edgecolor='white')
    ax.set_xticks(x)
    ax.set_xticklabels(feat_names, rotation=45, ha='right')
    ax.set_title('系数绝对值对比（特征选择）')
    ax.set_ylabel('|w| (Absolute Coefficient)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    save_path = get_results_path('lasso_regression_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    lasso_regression()
