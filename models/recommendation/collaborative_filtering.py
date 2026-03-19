"""
协同过滤 (Collaborative Filtering)
===================================
基于用户行为相似性的推荐算法

实现方法：
  1. 基于用户 (User-based): 找相似用户，推荐其喜欢的内容
  2. 基于物品 (Item-based): 找相似物品，推荐与用户历史相似的物品
  3. 相似度度量：皮尔逊相关系数、余弦相似度、Jaccard 相似度
  4. 评估：准确率/召回率/F1、覆盖度、新颖度
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise_distances
from collections import defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 相似度函数 ─────────────────────────────

def cosine_similarity(matrix):
    """余弦相似度矩阵"""
    norm = np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-8
    return matrix @ matrix.T / (norm @ norm.T)

def pearson_similarity(matrix):
    """皮尔逊相关系数（中心化后余弦）"""
    # 中心化：减去各自均值
    matrix_c = matrix - matrix.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(matrix_c, axis=1, keepdims=True) + 1e-8
    return matrix_c @ matrix_c.T / (norm @ norm.T)

def jaccard_similarity(matrix):
    """Jaccard 相似度（针对显式二元评分，集合重叠）"""
    # 阈值化评分：>=3 视为喜欢，否则忽略
    binary = (matrix >= 3).astype(int)
    intersect = binary @ binary.T
    union      = binary.sum(axis=1, keepdims=True) + binary.sum(axis=1).reshape(-1, 1) - intersect
    return intersect / (union + 1e-8)


# ─────────────────────── 协同过滤实现 ─────────────────────────

class UserBasedCF:
    """基于用户的协同过滤"""
    def __init__(self, k_neighbors=5, sim_method="cosine"):
        self.k_neighbors = k_neighbors
        self.sim_method = sim_method

    def fit(self, R):
        """R: (n_users, n_items) 评分矩阵，0 表示未评分"""
        self.R = R.copy()
        n_users, n_items = R.shape
        # 计算相似度
        if self.sim_method == "cosine":
            self.S = cosine_similarity(R)
        elif self.sim_method == "pearson":
            self.S = pearson_similarity(R)
        elif self.sim_method == "jaccard":
            self.S = jaccard_similarity(R)
        else:
            raise ValueError(f"Unknown sim_method: {self.sim_method}")
        # 对角线设为-1（排除自身）
        np.fill_diagonal(self.S, -1)
        self.n_users, self.n_items = n_users, n_items

    def predict(self, user_id, item_id):
        """预测 user_id 对 item_id 的评分"""
        if item_id < 0 or item_id >= self.n_items:
            return self.R[user_id].mean()
        # 找该用户评过分的邻居
        rated_items_mask = (self.R[:, item_id] > 0)
        neighbor_scores = self.S[user_id, :]
        # 取 top-k 邻居（正相似度）
        top_k_idx = np.argsort(neighbor_scores[rated_items_mask])[-self.k_neighbors:][::-1]
        top_k_sim = neighbor_scores[rated_items_mask][top_k_idx]
        top_k_ratings = self.R[rated_items_mask][top_k_idx, item_id]
        if len(top_k_ratings) == 0:
            return self.R[user_id].mean()
        # 加权预测
        pred = np.sum(top_k_sim * top_k_ratings) / (np.abs(top_k_sim).sum() + 1e-8)
        return pred

    def recommend(self, user_id, n_recommend=5, exclude_rated=True):
        """为用户推荐 top-n 物品"""
        scores = np.array([self.predict(user_id, i) for i in range(self.n_items)])
        if exclude_rated:
            # 排除已评分物品
            scores[self.R[user_id] > 0] = -1
        top_items = np.argsort(scores)[-n_recommend:][::-1]
        return [(item, scores[item]) for item in top_items]


class ItemBasedCF:
    """基于物品的协同过滤"""
    def __init__(self, k_neighbors=5, sim_method="cosine"):
        self.k_neighbors = k_neighbors
        self.sim_method = sim_method

    def fit(self, R):
        self.R = R.copy()
        # 计算物品相似度矩阵（转置后用用户相似度）
        if self.sim_method == "cosine":
            self.S = cosine_similarity(R.T)
        elif self.sim_method == "pearson":
            self.S = pearson_similarity(R.T)
        elif self.sim_method == "jaccard":
            self.S = jaccard_similarity(R.T)
        else:
            raise ValueError(f"Unknown sim_method: {self.sim_method}")
        np.fill_diagonal(self.S, -1)
        self.n_users, self.n_items = R.shape

    def predict(self, user_id, item_id):
        """预测：该用户评过分且与 item_id 相似的物品的加权评分"""
        if item_id < 0 or item_id >= self.n_items:
            return self.R[user_id].mean()
        # 该用户评过的物品
        rated_items = np.where(self.R[user_id] > 0)[0]
        item_sim = self.S[item_id, :]
        # 取 top-k 相似物品
        top_k_idx = rated_items[np.argsort(item_sim[rated_items])[-self.k_neighbors:][::-1]]
        top_k_sim = item_sim[top_k_idx]
        top_k_ratings = self.R[user_id, top_k_idx]
        if len(top_k_ratings) == 0:
            return self.R[user_id].mean()
        pred = np.sum(top_k_sim * top_k_ratings) / (np.abs(top_k_sim).sum() + 1e-8)
        return pred

    def recommend(self, user_id, n_recommend=5, exclude_rated=True):
        scores = np.array([self.predict(user_id, i) for i in range(self.n_items)])
        if exclude_rated:
            scores[self.R[user_id] > 0] = -1
        top_items = np.argsort(scores)[-n_recommend:][::-1]
        return [(item, scores[item]) for item in top_items]


# ─────────────────────── 评估指标 ─────────────────────────────

def evaluate(cf_model, R_train, R_test, k_recs=5):
    """评估推荐系统：Precision@K, Recall@K, Coverage"""
    n_users, n_items = R_test.shape
    precision_sum = recall_sum = 0
    hit_users = 0
    recommended_items = set()

    for u in range(n_users):
        # 获取该用户在测试集的真实高评分物品（>=4）
        true_items = set(np.where(R_test[u] >= 4)[0])
        if len(true_items) == 0:
            continue

        recs = cf_model.recommend(u, n_recommend=k_recs, exclude_rated=True)
        rec_items = set([item for item, _ in recs])
        recommended_items.update(rec_items)

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
    coverage = len(recommended_items) / n_items
    f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return {
        "Precision@K": precision,
        "Recall@K":    recall,
        "F1@K":        f1,
        "HitRate":      hit_rate,
        "Coverage":      coverage
    }


# ─────────────────────── 主程序 ──────────────────────────────────

def collaborative_filtering():
    print("协同过滤推荐系统运行中...\n")

    # ── 生成 MovieLens 风格模拟数据 ────────────────────────────────
    print("1. 生成模拟用户-电影评分矩阵...")
    np.random.seed(42)
    N_USERS = 100
    N_ITEMS = 50
    SPARSITY = 0.85  # 85% 评分缺失

    # 真实偏好矩阵（低秩）
    rank = 5
    U_true = np.random.randn(N_USERS, rank)
    V_true = np.random.randn(N_ITEMS, rank)
    R_true = U_true @ V_true.T
    R_true = (R_true - R_true.min()) / (R_true.ptp() + 1e-8) * 4 + 1  # 归一化到[1,5]

    # 采样观察矩阵
    mask = np.random.random((N_USERS, N_ITEMS)) > SPARSITY
    R_obs = np.where(mask, R_true, 0)

    # 划分训练/测试
    R_train = R_obs.copy()
    test_mask = np.random.random((N_USERS, N_ITEMS)) > 0.9
    R_test = np.where(test_mask & mask, R_obs, 0)

    print(f"   {N_USERS} 用户 x {N_ITEMS} 电影")
    print(f"   训练集评分数: {np.count_nonzero(R_train)}")
    print(f"   测试集评分数: {np.count_nonzero(R_test)}")

    # ── 运行协同过滤 ─────────────────────────────────────────────
    print("\n2. 训练 User-Based 和 Item-Based CF...")

    results = {}
    for method_name, cf_class in [("User-Based", UserBasedCF), ("Item-Based", ItemBasedCF)]:
        for sim in ["cosine", "pearson"]:
            print(f"   {method_name} ({sim})...")
            cf = cf_class(k_neighbors=5, sim_method=sim)
            cf.fit(R_train)
            metrics = evaluate(cf, R_train, R_test, k_recs=5)
            key = f"{method_name}_{sim}"
            results[key] = metrics
            print(f"     Prec@5={metrics['Precision@K']:.3f}  "
                  f"Rec@5={metrics['Recall@K']:.3f}  "
                  f"F1={metrics['F1@K']:.3f}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n3. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：原始偏好矩阵热力图（低秩） ────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(R_true[:20, :], cmap="RdYlGn", interpolation="nearest")
    ax.set_title("真实偏好矩阵（前20用户）", color="white", pad=8)
    ax.set_xlabel("电影ID", color="#aaa"); ax.set_ylabel("用户ID", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im, ax=ax, fraction=0.02).ax.tick_params(colors="gray")

    # ── 子图2：观察矩阵（稀疏） ─────────────────────────────────
    ax = axes[0, 1]
    im = ax.imshow(R_train[:20, :], cmap="RdYlGn", interpolation="nearest")
    ax.set_title("观察矩阵（稀疏，未评分=0）", color="white", pad=8)
    ax.set_xlabel("电影ID", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im, ax=ax, fraction=0.02).ax.tick_params(colors="gray")

    # ── 子图3：User-Based 推荐结果示例（用户0） ────────────────────
    ax = axes[0, 2]
    cf_user_cos = UserBasedCF(k_neighbors=5, sim_method="cosine")
    cf_user_cos.fit(R_train)
    recs = cf_user_cos.recommend(0, n_recommend=5)
    item_ids = [i for i, _ in recs]
    scores = [s for _, s in recs]

    # 该用户历史评分
    user_history = np.where(R_train[0] > 0)[0]
    history_scores = R_train[0, user_history]

    ax.scatter(user_history, history_scores, c="#3498db", s=80,
               alpha=0.8, label="历史评分")
    ax.scatter(item_ids, scores, c="#e74c3c", s=120,
               marker="*", alpha=0.9, label="推荐物品")
    ax.set_title(f"User-Based CF（用户0）推荐", color="white", pad=8)
    ax.set_xlabel("电影ID", color="#aaa"); ax.set_ylabel("评分", color="#aaa")
    ax.set_ylim(0.5, 5.5)
    ax.legend(fontsize=8, facecolor="#0d0d1a", labelcolor="white")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：Precision@K 对比 ────────────────────────────────────
    ax = axes[1, 0]
    names = list(results.keys())
    vals_prec = [results[n]["Precision@K"] for n in names]
    pal = plt.cm.viridis(np.linspace(0.3, 0.9, len(names)))
    bars = ax.bar(names, vals_prec, color=pal, alpha=0.85)
    ax.set_title("Precision@5 对比", color="white", pad=8)
    ax.set_ylabel("Precision", color="#aaa")
    ax.set_ylim(0, 0.4)
    ax.tick_params(colors="gray", axis="x", rotation=15)
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, vals_prec):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", color="white", fontsize=8)

    # ── 子图5：Recall@K 对比 ────────────────────────────────────────
    ax = axes[1, 1]
    vals_rec = [results[n]["Recall@K"] for n in names]
    bars = ax.bar(names, vals_rec, color=pal, alpha=0.85)
    ax.set_title("Recall@5 对比", color="white", pad=8)
    ax.set_ylabel("Recall", color="#aaa")
    ax.set_ylim(0, 0.4)
    ax.tick_params(colors="gray", axis="x", rotation=15)
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, vals_rec):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01,
                f"{v:.3f}", ha="center", color="white", fontsize=8)

    # ── 子图6：F1@K + Coverage 对比表 ────────────────────────────────
    ax = axes[1, 2]
    ax.axis("off")
    table_data = [
        ["方法", "F1@5", "Coverage"],
        *[[name, f"{results[name]['F1@K']:.3f}", f"{results[name]['Coverage']:.3f}"]
          for name in names]
    ]
    tbl = ax.table(cellText=table_data[1:], colLabels=table_data[0],
                   cellLoc="center", loc="center", colWidths=[0.32, 0.25, 0.25])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8)
    for (r, c), cell in tbl.get_celld().items():
        cell.set_facecolor("#1a3a5c" if r == 0 else "#16213e")
        cell.set_text_props(color="white"); cell.set_edgecolor("#334")
    ax.set_title("综合性能对比", color="white", pad=10)

    plt.suptitle("协同过滤推荐系统 (User-Based & Item-Based)",
                 color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("collaborative_filtering.png"))

    print("\n[DONE] 协同过滤推荐完成!")


if __name__ == "__main__":
    collaborative_filtering()
