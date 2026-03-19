"""
基于内容的推荐 (Content-Based Recommendation)
=============================================
基于物品属性和用户历史偏好，推荐相似物品

核心思想：
  1. 提取物品内容特征（标签、文本描述等）
  2. 用用户历史评分的物品特征构建用户画像（向量）
  3. 计算用户画像与候选物品的相似度，排序推荐

实现内容：
  1. 物品-特征矩阵 TF-IDF
  2. 用户画像：加权平均历史物品特征
  3. 相似度：余弦相似度
  4. 多样性控制：MMR (Maximal Marginal Relevance)
  5. 评估：Precision/Recall/覆盖率/多样性
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils import get_results_path, save_and_close


# ─────────────────────── 基于内容的推荐器 ────────────────────────

class ContentBasedRecommender:
    """
    基于内容的推荐系统

    参数：
      feature_matrix: (n_items, n_features) 物品-特征矩阵（如 TF-IDF）
    """
    def __init__(self, feature_matrix, item_ids):
        self.feature_matrix = feature_matrix
        self.item_ids = item_ids
        self.n_items, self.n_features = feature_matrix.shape
        self.user_profiles = {}

    def build_user_profile(self, R, user_id):
        """
        构建用户画像：加权平均用户历史评分物品的特征

        user_profile = sum(rating * item_feature) / sum(|rating|)
        """
        if user_id not in self.user_profiles:
            # 获取用户评过分的物品
            rated_items = np.where(R[user_id] > 0)[0]
            ratings = R[user_id, rated_items]
            # 加权平均特征
            weighted_features = np.zeros(self.n_features)
            total_weight = 0.0
            for item, r in zip(rated_items, ratings):
                weighted_features += r * self.feature_matrix[item]
                total_weight += abs(r)
            if total_weight > 0:
                self.user_profiles[user_id] = weighted_features / total_weight
            else:
                # 新用户用物品平均特征
                self.user_profiles[user_id] = self.feature_matrix.mean(axis=0)
        return self.user_profiles[user_id]

    def recommend(self, R, user_id, n_recommend=10, exclude_rated=True,
                 diversity_lambda=0.3, diversity_topk=30):
        """
        推荐物品（可选 MMR 多样性优化）

        MMR: argmax (λ * sim(item, user) - (1-λ) * max_sim_with_selected)
        """
        user_profile = self.build_user_profile(R, user_id)

        # 计算所有物品与用户的相似度
        similarities = cosine_similarity(
            user_profile.reshape(1, -1),
            self.feature_matrix
        )[0]

        if exclude_rated:
            # 排除用户评过分的物品
            rated_items = set(np.where(R[user_id] > 0)[0])
            for i in rated_items:
                similarities[i] = -1e9

        # 获取 top-k 候选
        top_k_idx = np.argsort(similarities)[-diversity_topk:][::-1]

        # MMR 迭代选择
        selected = []
        candidate_idx = list(top_k_idx)

        for _ in range(min(n_recommend, len(candidate_idx))):
            best_score = -np.inf
            best_item = None
            for item in candidate_idx:
                # 与用户的相似度
                relevance = similarities[item]
                # 与已选物品的最大相似度（用于多样性）
                if selected:
                    item_feature = self.feature_matrix[item].reshape(1, -1)
                    selected_features = self.feature_matrix[selected]
                    max_sim = cosine_similarity(item_feature, selected_features).max()
                else:
                    max_sim = 0.0
                # MMR 评分
                mmr_score = diversity_lambda * relevance - (1 - diversity_lambda) * max_sim
                if mmr_score > best_score:
                    best_score = mmr_score
                    best_item = item
            selected.append(best_item)
            candidate_idx.remove(best_item)

        # 返回推荐列表（物品ID + 相似度）
        return [(self.item_ids[i], similarities[i]) for i in selected]

    def evaluate(self, R_train, R_test, k_recs=10, rating_threshold=4.0):
        """评估：Precision@K, Recall@K, Diversity"""
        n_users = R_train.shape[0]
        precision_sum = recall_sum = 0
        diversity_sum = 0
        hit_users = 0
        recommended_items_set = set()

        for u in range(n_users):
            true_items = set(np.where(R_test[u] >= rating_threshold)[0])
            if len(true_items) == 0:
                continue
            recs = self.recommend(R_train, u, n_recommend=k_recs, exclude_rated=True)
            rec_items = [self.item_ids.tolist().index(item) if isinstance(item, str) else item
                        for item, _ in recs]
            rec_set = set(rec_items)
            recommended_items_set.update(rec_set)

            hits = len(rec_set & true_items)
            precision = hits / k_recs
            recall = hits / len(true_items)
            precision_sum += precision
            recall_sum += recall
            if hits > 0:
                hit_users += 1

            # 多样性：推荐物品间平均余弦相似度
            if len(rec_items) > 1:
                rec_features = self.feature_matrix[rec_items]
                sim_matrix = cosine_similarity(rec_features)
                np.fill_diagonal(sim_matrix, 0)
                diversity = 1 - sim_matrix.sum() / (len(rec_items) * (len(rec_items) - 1))
                diversity_sum += diversity

        hit_rate = hit_users / n_users
        precision = precision_sum / n_users
        recall = recall_sum / n_users
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        coverage = len(recommended_items_set) / self.n_items
        diversity = diversity_sum / n_users

        return {
            "Precision@K": precision,
            "Recall@K":    recall,
            "F1@K":        f1,
            "HitRate":      hit_rate,
            "Coverage":      coverage,
            "Diversity":    diversity
        }


# ─────────────────────── 生成模拟数据 ─────────────────────────

def generate_content_data(n_users=150, n_items=60, n_tags=20, seed=42):
    """
    生成带标签的物品内容和用户-物品评分矩阵

    物品标签：二进制向量 (n_items, n_tags)
    用户评分：基于用户偏好 + 物品标签匹配度
    """
    np.random.seed(seed)
    # 物品标签（每个物品 3-6 个标签）
    items = [f"Item_{i}" for i in range(n_items)]
    item_tags = np.zeros((n_items, n_tags), dtype=int)
    for i in range(n_items):
        n_tags_item = np.random.randint(3, 7)
        tag_indices = np.random.choice(n_tags, n_tags_item, replace=False)
        item_tags[i, tag_indices] = 1

    # 用户偏好（每个用户偏好 2-4 个标签）
    user_pref = np.zeros((n_users, n_tags), dtype=float)
    for u in range(n_users):
        pref_tags = np.random.choice(n_tags, np.random.randint(2, 5), replace=False)
        user_pref[u, pref_tags] = np.random.uniform(0.5, 1.0, size=len(pref_tags))

    # 生成评分：基于偏好匹配度 + 噪声
    R = np.zeros((n_users, n_items))
    for u in range(n_users):
        for i in range(n_items):
            # 匹配度：用户偏好与物品标签的点积
            match = np.dot(user_pref[u], item_tags[i])
            # 加上随机噪声
            rating = 1 + 4 * match + np.random.normal(0, 0.5)
            rating = np.clip(rating, 1, 5)
            R[u, i] = rating

    # 稀疏化（保留 20% 评分）
    mask = np.random.random((n_users, n_items)) > 0.8
    R = np.where(mask, R, 0)

    # 划分训练/测试
    train_mask = np.random.random((n_users, n_items)) > 0.9
    R_train = np.where(train_mask, R, 0)
    R_test  = np.where(~train_mask, R, 0)

    return items, item_tags, R_train, R_test, user_pref


def content_based():
    print("基于内容的推荐系统运行中...\n")

    # ── 生成数据 ──────────────────────────────────────────────────
    print("1. 生成带标签的物品内容数据...")
    items, item_tags, R_train, R_test, user_pref = generate_content_data(
        n_users=150, n_items=60, n_tags=20
    )
    n_users, n_items, n_tags = R_train.shape[0], len(items), item_tags.shape[1]
    print(f"   {n_users} 用户 x {n_items} 物品 x {n_tags} 标签")
    print(f"   训练集评分数: {np.count_nonzero(R_train)}")
    print(f"   测试集评分数: {np.count_nonzero(R_test)}")

    # ── 训练推荐器 ──────────────────────────────────────────────
    print("\n2. 训练基于内容的推荐器...")

    cb = ContentBasedRecommender(item_tags, np.arange(n_items))
    metrics = cb.evaluate(R_train, R_test, k_recs=10)

    print("=== 评估结果 ===")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    # ── 可视化 ────────────────────────────────────────────────────
    print("\n3. 生成可视化...")
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.patch.set_facecolor("#1a1a2e")
    for ax in axes.flat:
        ax.set_facecolor("#16213e")

    # ── 子图1：物品-标签矩阵热力图 ─────────────────────────────────
    ax = axes[0, 0]
    im = ax.imshow(item_tags.T, aspect="auto", cmap="YlOrRd", interpolation="nearest")
    ax.set_title("物品-标签矩阵", color="white", pad=8)
    ax.set_xlabel("物品ID", color="#aaa"); ax.set_ylabel("标签ID", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im, ax=ax, fraction=0.02).ax.tick_params(colors="gray")

    # ── 子图2：用户偏好热力图（前20用户） ─────────────────────────
    ax = axes[0, 1]
    im = ax.imshow(user_pref[:20, :], aspect="auto", cmap="PuRd", interpolation="nearest")
    ax.set_title("用户标签偏好（前20用户）", color="white", pad=8)
    ax.set_xlabel("标签ID", color="#aaa"); ax.set_ylabel("用户ID", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    plt.colorbar(im, ax=ax, fraction=0.02).ax.tick_params(colors="gray")

    # ── 子图3：用户画像示例（用户0） ─────────────────────────────
    ax = axes[0, 2]
    user_profile = cb.build_user_profile(R_train, 0)
    ax.bar(range(n_tags), user_profile, color="#3498db", alpha=0.85)
    ax.set_title("用户画像向量（用户0）", color="white", pad=8)
    ax.set_xlabel("标签ID", color="#aaa"); ax.set_ylabel("偏好权重", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图4：推荐结果示例（用户0） ─────────────────────────────
    ax = axes[1, 0]
    recs = cb.recommend(R_train, 0, n_recommend=10)
    rec_ids = [r[0] for r in recs]
    rec_scores = [r[1] for r in recs]
    ax.barh(range(10), rec_scores, color=plt.cm.viridis(np.linspace(0.3, 0.9, 10)), alpha=0.85)
    ax.set_yticks(range(10))
    ax.set_yticklabels([f"Item_{i}" for i in rec_ids], color="white", fontsize=8)
    ax.set_title("用户0 推荐 Top-10", color="white", pad=8)
    ax.set_xlabel("与用户画像相似度", color="#aaa")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    # ── 子图5：评估指标对比 ───────────────────────────────────────
    ax = axes[1, 1]
    metric_names = ["Precision@K", "Recall@K", "F1@K", "HitRate", "Coverage", "Diversity"]
    metric_vals = [metrics[m] for m in metric_names]
    pal = plt.cm.RdYlGn(np.linspace(0.2, 0.9, len(metric_names)))
    bars = ax.barh(metric_names, metric_vals, color=pal, alpha=0.85)
    ax.set_title("评估指标概览", color="white", pad=8)
    ax.set_xlabel("Score", color="#aaa")
    ax.set_xlim(0, max(metric_vals) * 1.15)
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")
    for bar, v in zip(bars, metric_vals):
        ax.text(v + 0.005, bar.get_y() + bar.get_height()/2,
                f"{v:.3f}", va="center", color="white", fontsize=8)

    # ── 子图6：不同 diversity_lambda 对多样性的影响 ───────────────────
    ax = axes[1, 2]
    lambdas = np.linspace(0, 1, 11)
    diversities = []
    precisions = []
    for lam in lambdas:
        cb_test = ContentBasedRecommender(item_tags, np.arange(n_items))
        cb_test.evaluate(R_train, R_test, k_recs=10)  # build profiles
        # 手动计算多样性
        div_scores = []
        for u in range(min(20, R_train.shape[0])):
            recs = cb_test.recommend(R_train, u, n_recommend=10, diversity_lambda=lam)
            rec_items = [cb_test.item_ids.tolist().index(r[0]) for r in recs]
            if len(rec_items) > 1:
                rec_features = item_tags[rec_items]
                sim = cosine_similarity(rec_features).sum() - np.trace(np.eye(len(rec_items)))
                diversity = 1 - sim / (len(rec_items) * (len(rec_items) - 1))
                div_scores.append(diversity)
        diversities.append(np.mean(div_scores) if div_scores else 0)

    ax.plot(lambdas, diversities, marker="o", color="#e74c3c", linewidth=2, markersize=6)
    ax.set_title("多样性 vs λ (MMR)", color="white", pad=8)
    ax.set_xlabel("多样性权重 λ", color="#aaa"); ax.set_ylabel("推荐多样性", color="#aaa")
    ax.grid(alpha=0.2, color="#555")
    ax.tick_params(colors="gray")
    for sp in ax.spines.values(): sp.set_color("#444")

    plt.suptitle("基于内容的推荐系统 (TF-IDF + 用户画像 + MMR)",
                 color="white", fontsize=14, y=1.01, fontweight="bold")
    plt.tight_layout()
    save_and_close(fig, get_results_path("content_based.png"))

    print("\n[DONE] 基于内容的推荐完成!")


if __name__ == "__main__":
    content_based()
