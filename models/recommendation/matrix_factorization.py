"""
矩阵分解 (Matrix Factorization)
================================
基于隐式用户/物品因子模型的推荐算法

核心思想：
  将用户-物品评分矩阵 R ≈ P @ Q^T
    - P: (n_users, k) 用户隐因子矩阵
    - Q: (n_items, k)  物品隐因子矩阵
    - k: 隐因子维度

实现方法：
  1. 基础 MF（SGD 优化，带 L2 正则化）
  2. SVD 分解（显式矩阵完整情况）
  3. ALS (Alternating Least Squares) 用于隐式反馈
  4. 评估：RMSE、Precision@K、召回率
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from collections import defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 矩阵分解实现 ─────────────────────────

class MatrixFactorization:
    """
    基础矩阵分解：P ∈ R^{n_users×k}, Q ∈ R^{n_items×k}
    目标：min_{P,Q} sum_{(u,i)} (R_{ui} - p_u^T q_i)^2 + λ(||P||^2 + ||Q||^2)
    使用 SGD 优化
    """
    def __init__(self, n_factors=10, lr=0.01, reg=0.02, max_iter=100):
        self.n_factors = n_factors
        self.lr = lr
        self.reg = reg
        self.max_iter = max_iter

    def fit(self, R, R_mask=None):
        """
        R: (n_users, n_items) 评分矩阵，0 表示缺失
        R_mask: 布尔矩阵，标记有效评分位置
        """
        if R_mask is None:
            R_mask = (R > 0)
        self.R = R
        self.R_mask = R_mask
        n_users, n_items = R.shape
        # 随机初始化
        rng = np.random.default_rng(42)
        self.P = rng.normal(0, 0.1, (n_users, self.n_factors))
        self.Q = rng.normal(0, 0.1, (n_items, self.n_factors))

        loss_history = []
        # SGD 迭代
        for it in range(self.max_iter):
            loss = 0.0
            # 随机采样有效评分
            users, items = np.where(R_mask)
            perm = np.random.permutation(len(users))
            for idx in perm:
                u, i = users[perm[idx]], items[perm[idx]]
                r_ui = R[u, i]
                pred = self.P[u] @ self.Q[i]
                err = r_ui - pred
                # 梯度更新
                self.P[u] += self.lr * (err * self.Q[i] - self.reg * self.P[u])
                self.Q[i] += self.lr * (err * self.P[u] - self.reg * self.Q[i])
                loss += err ** 2

            loss += self.reg * (np.linalg.norm(self.P)**2 + np.linalg.norm(self.Q)**2)
            loss_history.append(loss)
            if it % 20 == 0:
                print(f"    Iter {it}: Loss = {loss:.4f}")

        self.loss_history = loss_history
        return self

    def predict(self, user_id, item_id):
        if user_id < 0 or user_id >= self.P.shape[0]:
            return 0.0
        if item_id < 0 or item_id >= self.Q.shape[0]:
            return 0.0
        return self.P[user_id] @ self.Q[item_id]

    def recommend(self, user_id, n_recommend=5, exclude_rated=True):
        n_items = self.Q.shape[0]
        scores = np.array([self.predict(user_id, i) for i in range(n_items)])
        if exclude_rated:
            scores[~self.R_mask[user_id]] = -1e9
        top_items = np.argsort(scores)[-n_recommend:][::-1]
        return [(item, scores[item]) for item in top_items]


class ALSMF:
    """
    隐式反馈的 ALS (Alternating Least Squares)
    针对隐式反馈（点击/浏览）优化，使用置信度模型
    """
    def __init__(self, n_factors=10, reg=0.1, alpha=40, max_iter=20):
        self.n_factors = n_factors
        self.reg = reg
        self.alpha = alpha  # 置信度参数
        self.max_iter = max_iter

    def _prepare_confidence(self, R):
        """构建置信度矩阵 C_{ui} = 1 + α * R_{ui}"""
        return 1 + self.alpha * R

    def fit(self, R):
        """
        R: 隐式反馈矩阵（二值/计数）
        """
        self.R = R
        C = self._prepare_confidence(R)
        n_users, n_items = R.shape
        rng = np.random.default_rng(42)
        self.X = rng.normal(0, 0.01, (n_users, self.n_factors))
        self.Y = rng.normal(0, 0.01, (n_items, self.n_factors))

        loss_history = []
        I = np.eye(self.n_factors)

        for it in range(self.max_iter):
            # 更新用户因子 X
            for u in range(n_users):
                cu = np.diag(C[u])
                yu = np.where(C[u] > 1, 1, 0)
                Yt = self.Y.T
                XtX = Yt @ cu @ self.Y + self.reg * I
                Xty = Yt @ cu @ yu
                self.X[u] = np.linalg.solve(XtX, Xty)

            # 更新物品因子 Y
            for i in range(n_items):
                ci = np.diag(C[:, i])
                xi = np.where(C[:, i] > 1, 1, 0)
                Xt = self.X.T
                YtY = Xt @ ci @ self.X + self.reg * I
                Ytx = Xt @ ci @ xi
                self.Y[i] = np.linalg.solve(YtY, Ytx)

            loss = np.sum(C * (self.X @ self.Y.T - R)**2) + self.reg * (np.linalg.norm(self.X)**2 + np.linalg.norm(self.Y)**2)
            loss_history.append(loss)
            if it % 5 == 0:
                print(f"    ALS Iter {it}: Loss = {loss:.4f}")

        self.loss_history = loss_history
        return self

    def predict(self, user_id, item_id):
        if user_id < 0 or user_id >= self.X.shape[0]:
            return 0.0
        if item_id < 0 or item_id >= self.Y.shape[0]:
            return 0.0
        return self.X[user_id] @ self.Y[item_id]

    def recommend(self, user_id, n_recommend=5, exclude_rated=True):
        n_items = self.Y.shape[0]
        scores = np.array([self.predict(user_id, i) for i in range(n_items)])
        if exclude_rated:
            scores[self.R[user_id] > 0] = -1e9
        top_items = np.argsort(scores)[-n_recommend:][::-1]
        return [(item, scores[item]) for item in top_items]


# ─────────────────────── 评估指标 ─────────────────────────────

def evaluate(mf_model, R_train, R_test, k_recs=5, rating_threshold=4.0):
    """评估推荐系统：RMSE、Precision@K、Recall@K"""
    # RMSE
    test_users, test_items = np.where(R_test > 0)
    sq_err = []
    for u, i in zip(test_users, test_items):
        pred = mf_model.predict(u, i)
        sq_err.append((pred - R_test[u, i])**2)
    rmse = np.sqrt(np.mean(sq_err)) if sq_err else 0.0

    # Precision@K, Recall@K
    n_users = R_test.shape[0]
    precision_sum = recall_sum = 0
    hit_users = 0
    for u in range(n_users):
        true_items = set(np.where(R_test[u] >= rating_threshold)[0])
        if len(true_items) == 0:
            continue
        recs = mf_model.recommend(u, n_recommend=k_recs, exclude_rated=True)
        rec_items = set([item for item, _ in recs])
        hits = len(rec_items & true_items)
        precision = hits / k_recs
        recall = hits / len(true_items)
        precision_sum += precision
        recall_sum += recall
        if hits > 0:
            hit_users += 1

    hit_rate = hit_users / n_users
    precision = precision_sum / n_users
    recall = recall_sum / n_users
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "RMSE":       rmse,
        "Precision@K": precision,
        "Recall@K":    recall,
        "F1@K":        f1,
        "HitRate":      hit_rate
    }


# ─────────────────────── 主程序 ──────────────────────────────────

def matrix_factorization():
    print("矩阵分解推荐系统运行中...\n")

    # ── 生成 MovieLens 风格模拟数据 ────────────────────────────────
    print("1. 生成模拟用户-电影评分矩阵...")
    np.random.seed(42)
    N_USERS = 100
    N_ITEMS = 50
    RANK = 5
    SPARSITY = 0.85

    U_true = np.random.randn(N_USERS, RANK)
    V_true = np.random.randn(N_ITEMS, RANK)
    R_true = U_true @ V_true.T
    R_true = (R_true - R_true.min()) / (R_true.ptp() + 1e-8) * 4 + 1

    # 采样观察
    mask = np.random.random((N_USERS, N_ITEMS)) > SPARSITY
    R_obs = np.where(mask, R_true, 0)

    # 划分训练/测试
    R_train = R_obs.copy()
    test_mask = np.random.random((N_USERS, N_ITEMS)) > 0.9
    R_test = np.where(test_mask & mask, R_obs, 0)

    print(f"   {N_USERS} 用户 x {N_ITEMS} 电影")
    print(f"   训练集评分数: {np.count_nonzero(R_train)}")
    print(f"   测试集评分数: {np.count_nonzero(R_test)}")

    # ── 运行矩阵分解 ──────────────────────────────────────────────
    print("\n2. 训练矩阵分解模型...")

    mf = MatrixFactorization(n_factors=10, lr=0.005, reg=0.02, max_iter=80)
    print("   MF (SGD):")
    mf.fit(R_train)
    metrics_mf = evaluate(mf, R_train, R_test, k_recs=5)

    als = ALSMF(n_factors=10, reg=0.1, alpha=40, max_iter=15)
    print("\n   ALSMF (隐式反馈):")
    als.fit(R_train.astype(int))  # 隐式反馈用整数
    metrics_als = evaluate(als, R_train, R_test, k_recs=5)

    # 打印结果
    print("\n=== MF (SGD) 性能 ===")
    for k, v in metrics_mf.items():
        print(f"  {k}: {v:.4f}")
    print("\n=== ALSMF 性能 ===")
    for k, v in metrics_als.items():
        print(f"  {k}: {v:.4f}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n3. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：训练损失曲线 ──────────────────────────────────────
    ax = axes[0, 0]
    ax.plot(mf.loss_history, color="#e74c3c", linewidth=2, label="MF (SGD)")
    ax.plot(als.loss_history, color="#3498db", linewidth=2, label="ALSMF")
    ax.set_title("训练损失曲线", color="white", pad=8)
    ax.set_xlabel("迭代步数", color="#aaa"); ax.set_ylabel("Loss", color="#aaa")
    ax.legend(fontsize=9, facecolor="#0d0d1a", labelcolor="white")
    ax.grid(alpha=0.2, color="#555")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图2：真实 vs 预测评分对比（MF） ────────────────────────
    ax = axes[0, 1]
    users, items = np.where(R_test > 0)[:50]  # 前50个测试样本
    true_vals = R_true[users, items]
    pred_vals = np.array([mf.predict(u, i) for u, i in zip(users, items)])
    ax.scatter(true_vals, pred_vals, c="#3498db", alpha=0.7, s=50, edgecolors="white", linewidths=0.5)
    lim = [min(true_vals.min(), pred_vals.min()), max(true_vals.max(), pred_vals.max())]
    ax.plot(lim, lim, "--", color="#e74c3c", alpha=0.7, linewidth=1.5)
    ax.set_title("MF 预测 vs 真实评分", color="white", pad=8)
    ax.set_xlabel("真实评分", color="#aaa"); ax.set_ylabel("预测评分", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图3：用户/物品隐因子热力图（MF） ──────────────────────────
    ax = axes[0, 2]
    im_u = ax.imshow(mf.P[:20, :], aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_title("用户隐因子矩阵 P (前20用户)", color="white", pad=8)
    ax.set_xlabel("因子维度", color="#aaa"); ax.set_ylabel("用户ID", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im_u, ax=ax, fraction=0.02, pad=0.02).ax.tick_params(colors="gray")

    # ── 子图4：RMSE 对比 ────────────────────────────────────────
    ax = axes[1, 0]
    names = ["MF (SGD)", "ALSMF"]
    rmses = [metrics_mf["RMSE"], metrics_als["RMSE"]]
    pal = ["#e74c3c", "#3498db"]
    bars = ax.bar(names, rmses, color=pal, alpha=0.85)
    ax.set_title("RMSE 对比", color="white", pad=8)
    ax.set_ylabel("RMSE", color="#aaa")
    ax.set_ylim(0, max(rmses) * 1.2)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, rmses):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.02,
                f"{v:.3f}", ha="center", va="bottom", color="white", fontsize=10)

    # ── 子图5：Precision@K / Recall@K 对比 ─────────────────────────
    ax = axes[1, 1]
    metrics = ["Precision@K", "Recall@K", "F1@K"]
    mf_vals   = [metrics_mf[m] for m in metrics]
    als_vals = [metrics_als[m] for m in metrics]
    x = np.arange(len(metrics))
    width = 0.35
    ax.bar(x - width/2, mf_vals, width, color="#e74c3c", label="MF (SGD)", alpha=0.85)
    ax.bar(x + width/2, als_vals, width, color="#3498db", label="ALSMF", alpha=0.85)
    ax.set_title("推荐质量对比", color="white", pad=8)
    ax.set_ylabel("Score", color="#aaa")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, color="white")
    ax.set_ylim(0, max(max(mf_vals), max(als_vals)) * 1.2)
    ax.legend(fontsize=9, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图6：推荐结果示例（用户0） ────────────────────────────
    ax = axes[1, 2]
    recs_mf = mf.recommend(0, n_recommend=5)
    recs_als = als.recommend(0, n_recommend=5)
    mf_ids, mf_scores = zip(*recs_mf)
    als_ids, als_scores = zip(*recs_als)
    ax.barh(range(5), mf_scores, color="#e74c3c", alpha=0.7, label="MF")
    ax.barh(range(5), als_scores, color="#3498db", alpha=0.7, label="ALSMF", left=list(mf_scores))
    ax.set_yticks(range(5))
    ax.set_yticklabels([f"Item_{i}" for i in mf_ids], color="white")
    ax.set_title("用户0 推荐 Top-5 (堆叠)", color="white", pad=8)
    ax.set_xlabel("预测评分", color="#aaa")
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    plt.suptitle("矩阵分解推荐系统 (MF & ALSMF)", color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("matrix_factorization.png"))

    print("\n[DONE] 矩阵分解推荐完成!")


if __name__ == "__main__":
    matrix_factorization()
