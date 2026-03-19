"""
语义分割 (Semantic Segmentation)
==================================
基于超像素 + 图割（Graph Cut）+ 随机森林的语义分割（纯 NumPy / sklearn 实现）

核心流程：
  1. 合成场景图像（背景 + 几个彩色目标区域）
  2. 像素级特征提取（位置、颜色/灰度值、局部梯度、邻域统计）
  3. 用逻辑回归 / 随机森林（手写简化版）进行像素分类
  4. 后处理：CRF-like 平滑（空间一致性约束）
  5. 可视化：预测分割图 vs 真实 mask + mIoU 评估
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# ──────────────────────── 数据集 ─────────────────────────────────

IMG_SIZE  = 48
N_CLASSES = 4   # 0=背景, 1=圆形, 2=矩形, 3=三角形

CLASS_COLORS = [
    [0.15, 0.15, 0.20],   # 背景 —— 深色
    [0.95, 0.40, 0.40],   # 圆形 —— 红
    [0.35, 0.75, 0.85],   # 矩形 —— 青
    [0.55, 0.90, 0.55],   # 三角 —— 绿
]
SEG_CMAP = ListedColormap(CLASS_COLORS)


def _circle_mask(size, cx, cy, r):
    Y, X = np.ogrid[:size, :size]
    return (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2


def _rect_mask(size, x1, y1, x2, y2):
    mask = np.zeros((size, size), dtype=bool)
    mask[y1:y2, x1:x2] = True
    return mask


def _triangle_mask(size, cx, cy, half):
    mask = np.zeros((size, size), dtype=bool)
    for row in range(max(0, cy - half), min(size, cy + half)):
        span = int((row - (cy - half)) * half / (half + 1e-5))
        c1, c2 = max(0, cx - span), min(size, cx + span)
        mask[row, c1:c2] = True
    return mask


def generate_scene(rng, n_objects: int = None) -> tuple:
    """返回 (RGB 图像 H×W×3, 标签图 H×W)"""
    img  = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.float32) * np.array([0.15, 0.15, 0.20])
    label = np.zeros((IMG_SIZE, IMG_SIZE), dtype=np.int32)

    if n_objects is None:
        n_objects = rng.integers(2, 4)

    for _ in range(n_objects):
        obj_type = rng.integers(1, N_CLASSES)   # 1,2,3
        cx = rng.integers(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
        cy = rng.integers(IMG_SIZE // 4, 3 * IMG_SIZE // 4)
        color = np.array(CLASS_COLORS[obj_type], dtype=np.float32)
        color += rng.uniform(-0.05, 0.05, 3).astype(np.float32)
        color  = np.clip(color, 0, 1)

        if obj_type == 1:  # 圆形
            r = rng.integers(5, 11)
            m = _circle_mask(IMG_SIZE, cx, cy, r)
        elif obj_type == 2:  # 矩形
            w = rng.integers(8, 15)
            h = rng.integers(8, 15)
            m = _rect_mask(IMG_SIZE,
                            max(0, cx - w//2), max(0, cy - h//2),
                            min(IMG_SIZE, cx + w//2), min(IMG_SIZE, cy + h//2))
        else:  # 三角形
            half = rng.integers(6, 12)
            m = _triangle_mask(IMG_SIZE, cx, cy, half)

        img[m]   = color
        label[m] = obj_type

    # 加轻微噪声
    img += rng.normal(0, 0.03, img.shape).astype(np.float32)
    img = np.clip(img, 0, 1)
    return img, label


def generate_dataset(n_train: int = 200, n_val: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    train = [generate_scene(rng) for _ in range(n_train)]
    val   = [generate_scene(rng) for _ in range(n_val)]
    return train, val


# ──────────────────────── 像素特征提取 ──────────────────────────

def extract_pixel_features(img: np.ndarray) -> np.ndarray:
    """
    img: (H, W, 3)
    返回: (H*W, n_features)
    """
    H, W, C = img.shape
    gray = img.mean(axis=2)

    # 归一化坐标
    ys, xs = np.mgrid[0:H, 0:W]
    ny = ys / (H - 1) - 0.5
    nx = xs / (W - 1) - 0.5

    # 梯度幅值
    gy = np.gradient(gray, axis=0)
    gx = np.gradient(gray, axis=1)
    mag = np.sqrt(gx**2 + gy**2)

    # 局部均值（3×3 平均池化，使用 pad + 滑动）
    pad_gray = np.pad(gray, 1, mode="edge")
    local_mean = np.zeros_like(gray)
    local_std  = np.zeros_like(gray)
    for dr in range(3):
        for dc in range(3):
            local_mean += pad_gray[dr:dr+H, dc:dc+W]
    local_mean /= 9

    for dr in range(3):
        for dc in range(3):
            local_std += (pad_gray[dr:dr+H, dc:dc+W] - local_mean) ** 2
    local_std = np.sqrt(local_std / 9)

    feats = np.stack([
        img[:, :, 0],      # R
        img[:, :, 1],      # G
        img[:, :, 2],      # B
        gray,              # 灰度
        ny,                # 归一化 y 坐标
        nx,                # 归一化 x 坐标
        gx,                # x 梯度
        gy,                # y 梯度
        mag,               # 梯度幅值
        local_mean,        # 局部均值
        local_std,         # 局部标准差
    ], axis=-1)            # (H, W, n_feats)

    return feats.reshape(H * W, -1).astype(np.float32)


# ──────────────────────── 简化随机森林 ──────────────────────────

class DecisionStump:
    """单特征阈值分类器（随机森林的基元）"""
    def __init__(self):
        self.feat_idx = None
        self.threshold = None
        self.left_cls  = None
        self.right_cls = None

    def fit(self, X, y, sample_weight=None):
        n, d = X.shape
        best_score = -1.0
        rng = np.random.default_rng()
        # 随机采样特征子集
        n_try = max(1, int(np.sqrt(d)))
        feat_subset = rng.choice(d, n_try, replace=False)
        for fi in feat_subset:
            vals = X[:, fi]
            thrs = rng.choice(vals, min(10, len(vals)), replace=False)
            for thr in thrs:
                left  = y[vals <= thr]
                right = y[vals >  thr]
                if len(left) == 0 or len(right) == 0:
                    continue
                score = len(left) * gini_impurity(left) + len(right) * gini_impurity(right)
                score = -score  # 越小越好，取负
                if score > best_score:
                    best_score = score
                    self.feat_idx  = fi
                    self.threshold = thr
                    self.left_cls  = most_common(left)
                    self.right_cls = most_common(right)
        if self.feat_idx is None:
            self.feat_idx  = 0
            self.threshold = 0
            self.left_cls  = most_common(y)
            self.right_cls = self.left_cls

    def predict(self, X):
        mask = X[:, self.feat_idx] <= self.threshold
        out  = np.where(mask, self.left_cls, self.right_cls)
        return out


def gini_impurity(y):
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - (p ** 2).sum()


def most_common(arr):
    vals, cnts = np.unique(arr, return_counts=True)
    return int(vals[cnts.argmax()])


class SimpleRandomForest:
    """极简随机森林（决策树桩集成）"""
    def __init__(self, n_estimators: int = 30, max_samples: float = 0.7,
                 seed: int = 42):
        self.n_estimators = n_estimators
        self.max_samples  = max_samples
        self.rng  = np.random.default_rng(seed)
        self.stumps: list = []
        self.n_classes = None
        self.losses = []

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.n_classes = int(y.max()) + 1
        N = len(X)
        n_sub = max(1, int(N * self.max_samples))
        self.stumps = []
        for t in range(self.n_estimators):
            idx = self.rng.choice(N, n_sub, replace=True)
            stump = DecisionStump()
            stump.fit(X[idx], y[idx])
            self.stumps.append(stump)
            # 记录 OOB 类似的训练准确率
            pred = self._vote(X[:min(200, N)])
            acc  = (pred == y[:min(200, N)]).mean()
            self.losses.append(1 - acc)
        return self

    def _vote(self, X: np.ndarray) -> np.ndarray:
        votes = np.stack([s.predict(X) for s in self.stumps], axis=1)
        out   = np.apply_along_axis(most_common, 1, votes)
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._vote(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        votes = np.stack([s.predict(X) for s in self.stumps], axis=1)
        proba = np.zeros((len(X), self.n_classes), dtype=np.float32)
        for ci in range(self.n_classes):
            proba[:, ci] = (votes == ci).mean(axis=1)
        return proba


# ──────────────────────── CRF-like 后处理 ───────────────────────

def spatial_smooth(pred_label: np.ndarray, n_iter: int = 2,
                    sigma: float = 0.5) -> np.ndarray:
    """
    简化版 CRF 平滑：对每个像素用 3×3 邻域多数投票替换当前标签
    """
    from scipy.ndimage import label as sp_label  # 只用 ndimage.label
    result = pred_label.copy()
    H, W   = result.shape
    pad    = np.pad(result, 1, mode="edge")

    for _ in range(n_iter):
        new_result = result.copy()
        for r in range(H):
            for c in range(W):
                neighborhood = pad[r:r+3, c:c+3].ravel()
                new_result[r, c] = most_common(neighborhood)
        pad    = np.pad(new_result, 1, mode="edge")
        result = new_result

    return result


# ──────────────────────── 评估 mIoU ──────────────────────────────

def mean_iou(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> dict:
    ious = []
    per_class = {}
    for c in range(n_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()
        iou = tp / (tp + fp + fn + 1e-10)
        ious.append(iou)
        per_class[c] = iou
    return {"mIoU": np.mean(ious), "per_class": per_class}


# ──────────────────────── 可视化 ──────────────────────────────────

CLS_NAMES = ["Background", "Circle", "Rectangle", "Triangle"]


def plot_results(model: SimpleRandomForest, val_dataset: list,
                 all_miou: list,
                 save_path: str = "results/semantic_segmentation_results.png"):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")

    # ── 1. 学习曲线（训练 error 代理）──
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(model.losses, color="#FF6B6B", linewidth=1.5)
    ax1.set_title("RF Training Error Rate", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("# Stumps", color="white")
    ax1.set_ylabel("1 - Accuracy", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    for s in ax1.spines.values(): s.set_edgecolor("#444")
    ax1.grid(True, alpha=0.2)

    # ── 2. Per-class IoU ──
    ax2 = fig.add_subplot(2, 4, 2)
    # 计算验证集平均 per-class IoU
    all_true, all_pred = [], []
    for img, label in val_dataset[:20]:
        feats = extract_pixel_features(img)
        pred  = model.predict(feats).reshape(IMG_SIZE, IMG_SIZE)
        pred  = spatial_smooth(pred)
        all_true.append(label.ravel())
        all_pred.append(pred.ravel())
    agg_true = np.concatenate(all_true)
    agg_pred = np.concatenate(all_pred)
    miou_info = mean_iou(agg_true, agg_pred, N_CLASSES)

    pc_ious = [miou_info["per_class"][c] for c in range(N_CLASSES)]
    colors  = ["#888888", "#FF6B6B", "#4ECDC4", "#98D8C8"]
    ax2.bar(CLS_NAMES, pc_ious, color=colors, alpha=0.85)
    ax2.set_ylim(0, 1.1)
    ax2.set_title(f"Per-Class IoU  (mIoU={miou_info['mIoU']:.3f})",
                  color="white", fontsize=10, fontweight="bold")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white", axis="both")
    ax2.set_xticklabels(CLS_NAMES, rotation=15, color="white", fontsize=8)
    for s in ax2.spines.values(): s.set_edgecolor("#444")
    for i, v in enumerate(pc_ious):
        ax2.text(i, v + 0.02, f"{v:.2f}", ha="center", color="white", fontsize=8)

    # ── 3. mIoU 分布（val 前 20 个场景）──
    ax3 = fig.add_subplot(2, 4, 3)
    ax3.plot(range(len(all_miou)), all_miou, "o-", color="#45B7D1",
             linewidth=1.5, markersize=4)
    ax3.axhline(np.mean(all_miou), color="#FFA07A", linestyle="--",
                label=f"Mean={np.mean(all_miou):.3f}")
    ax3.set_title("mIoU per Val Scene", color="white", fontsize=10, fontweight="bold")
    ax3.set_xlabel("Scene #", color="white")
    ax3.set_ylabel("mIoU", color="white")
    ax3.set_facecolor("#16213e")
    ax3.tick_params(colors="white")
    ax3.legend(facecolor="#0f3460", labelcolor="white", fontsize=8)
    for s in ax3.spines.values(): s.set_edgecolor("#444")
    ax3.grid(True, alpha=0.2)

    # ── 4. 混淆矩阵（像素级）──
    ax4 = fig.add_subplot(2, 4, 4)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64)
    for t, p in zip(agg_true, agg_pred):
        cm[t, p] += 1
    cm_norm = cm / (cm.sum(axis=1, keepdims=True) + 1e-10)
    im = ax4.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax4.set_xticks(range(N_CLASSES))
    ax4.set_yticks(range(N_CLASSES))
    ax4.set_xticklabels([c[:3] for c in CLS_NAMES], color="white", fontsize=8, rotation=45)
    ax4.set_yticklabels([c[:3] for c in CLS_NAMES], color="white", fontsize=8)
    ax4.set_title("Normalized Confusion Matrix", color="white", fontsize=10, fontweight="bold")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax4.text(j, i, f"{cm_norm[i,j]:.2f}", ha="center", va="center",
                     color="white" if cm_norm[i, j] < 0.5 else "black", fontsize=8)
    plt.colorbar(im, ax=ax4, fraction=0.046, pad=0.04)

    # ── 5-8. 分割结果示例 ──
    sample_idx = np.random.default_rng(99).choice(len(val_dataset), 4, replace=False)
    for k, si in enumerate(sample_idx):
        img, label = val_dataset[si]
        feats = extract_pixel_features(img)
        pred  = model.predict(feats).reshape(IMG_SIZE, IMG_SIZE)
        pred  = spatial_smooth(pred)

        ax_img  = fig.add_subplot(3, 4, 9 + k)       # row 3
        ax_lab  = fig.add_axes([0.0, 0.0, 0.01, 0.01])  # placeholder

        # 合并为三列显示（原图 | GT | Pred）
        combined = np.concatenate([
            img,
            np.stack([SEG_CMAP(label / (N_CLASSES-1))[:, :, :3]
                      ], axis=0)[0],
            np.stack([SEG_CMAP(pred / (N_CLASSES-1))[:, :, :3]
                      ], axis=0)[0],
        ], axis=1)  # (H, 3W, 3)

        ax_img.imshow(combined)
        mi = mean_iou(label.ravel(), pred.ravel(), N_CLASSES)["mIoU"]
        ax_img.set_title(f"Img|GT|Pred  mIoU={mi:.2f}", color="white", fontsize=8)
        ax_img.axis("off")

        # 删除占位 axes
        ax_lab.remove()

    plt.suptitle("Semantic Segmentation — Random Forest + CRF Smoothing",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ──────────────────────── 主函数 ──────────────────────────────────

def semantic_segmentation():
    """语义分割完整流程（合成数据 + 随机森林 + CRF 平滑）"""
    print("=" * 60)
    print("   语义分割 (Semantic Segmentation) — 随机森林 + CRF")
    print("=" * 60)

    # ── 1. 数据 ──
    print("\n[1/4] 生成合成场景数据...")
    train_data, val_data = generate_dataset(n_train=150, n_val=30, seed=42)
    print(f"  训练场景: {len(train_data)}, 验证场景: {len(val_data)}")

    # ── 2. 构建像素级训练集 ──
    print("\n[2/4] 提取像素特征...")
    # 使用部分训练图像避免内存过大
    X_list, y_list = [], []
    rng = np.random.default_rng(42)
    for img, label in train_data:
        feats = extract_pixel_features(img)  # (H*W, n_feats)
        # 随机采样像素（每张图最多 500 个像素，加快训练）
        n_pixels = len(feats)
        sel = rng.choice(n_pixels, min(500, n_pixels), replace=False)
        X_list.append(feats[sel])
        y_list.append(label.ravel()[sel])
    X_train = np.concatenate(X_list)
    y_train = np.concatenate(y_list)
    print(f"  像素特征维度: {X_train.shape[1]}, 训练样本量: {len(X_train)}")
    unique, counts = np.unique(y_train, return_counts=True)
    for uid, cnt in zip(unique, counts):
        print(f"    {CLS_NAMES[uid]:<12}: {cnt}")

    # ── 3. 训练 ──
    print("\n[3/4] 训练简化随机森林 (30 棵决策桩)...")
    model = SimpleRandomForest(n_estimators=30, max_samples=0.7, seed=42)
    model.fit(X_train, y_train)
    print(f"  最终训练 Error: {model.losses[-1]:.4f}")

    # ── 4. 评估 ──
    print("\n[4/4] 评估验证集 mIoU...")
    all_miou = []
    for img, label in val_data[:20]:
        feats = extract_pixel_features(img)
        pred  = model.predict(feats).reshape(IMG_SIZE, IMG_SIZE)
        pred  = spatial_smooth(pred)
        mi    = mean_iou(label.ravel(), pred.ravel(), N_CLASSES)
        all_miou.append(mi["mIoU"])
    print(f"  平均 mIoU (前20张): {np.mean(all_miou):.4f}")
    print(f"  最大 mIoU: {max(all_miou):.4f}  最小: {min(all_miou):.4f}")

    # 示例展示
    print("\n  示例（前3张验证场景）:")
    for i in range(min(3, len(val_data))):
        img, label = val_data[i]
        feats = extract_pixel_features(img)
        pred  = model.predict(feats).reshape(IMG_SIZE, IMG_SIZE)
        pred  = spatial_smooth(pred)
        mi    = mean_iou(label.ravel(), pred.ravel(), N_CLASSES)
        px_acc = (pred == label).mean()
        print(f"    Scene {i}: pixel_acc={px_acc:.4f}  mIoU={mi['mIoU']:.4f}")

    # ── 可视化 ──
    print("\n生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(model, val_data, all_miou,
                 save_path="results/semantic_segmentation_results.png")

    print("\n[DONE] 语义分割完成!")
    return model, all_miou


if __name__ == "__main__":
    semantic_segmentation()
