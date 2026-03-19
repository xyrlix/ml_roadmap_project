"""
目标检测 (Object Detection)
============================
基于滑动窗口 + 手工特征 + 机器学习分类器实现目标检测（无需深度学习框架）

核心算法：
  1. 合成场景图像（每张包含 0-3 个随机几何"物体"：矩形/圆形/三角形）
  2. 多尺度滑动窗口提取候选区域
  3. 为每个候选区域提取 HOG-like 特征
  4. 用 Logistic 分类器判断候选框是否包含目标，以及目标类别
  5. NMS（非极大值抑制）过滤重叠框
  6. 计算 IoU / Precision@IoU0.5 评估检测效果
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from itertools import product

# ─────────────────────── 合成场景数据 ────────────────────────────

CLASSES = ["rectangle", "circle", "triangle"]
N_CLASSES = len(CLASSES)
BG_CLASS  = N_CLASSES       # 背景类 id

IMG_SIZE  = 64
PATCH_SIZE = 16


def draw_rectangle(img, rng, cx, cy, w, h, val=0.85):
    x1, y1 = max(0, cx - w // 2), max(0, cy - h // 2)
    x2, y2 = min(IMG_SIZE - 1, cx + w // 2), min(IMG_SIZE - 1, cy + h // 2)
    img[y1:y2, x1:x2] = val + rng.uniform(-0.1, 0.1)
    return (x1, y1, x2, y2, 0)  # bbox + class_id


def draw_circle(img, rng, cx, cy, r, val=0.75):
    Y, X = np.ogrid[:IMG_SIZE, :IMG_SIZE]
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
    img[mask] = val + rng.uniform(-0.1, 0.1)
    x1 = max(0, cx - r); y1 = max(0, cy - r)
    x2 = min(IMG_SIZE - 1, cx + r); y2 = min(IMG_SIZE - 1, cy + r)
    return (x1, y1, x2, y2, 1)


def draw_triangle(img, rng, cx, cy, size, val=0.65):
    """简单的等腰三角形（光栅化）"""
    half = size // 2
    for row in range(cy - half, min(cy + half, IMG_SIZE)):
        span = int((row - (cy - half)) * size / (2 * half + 1e-5))
        c1 = max(0, cx - span)
        c2 = min(IMG_SIZE - 1, cx + span)
        if 0 <= row < IMG_SIZE:
            img[row, c1:c2] = val + rng.uniform(-0.1, 0.1)
    x1 = max(0, cx - half); y1 = max(0, cy - half)
    x2 = min(IMG_SIZE - 1, cx + half); y2 = min(IMG_SIZE - 1, cy + half)
    return (x1, y1, x2, y2, 2)


def generate_scene(rng, n_objects: int = None) -> tuple:
    """生成一张含随机目标的合成图像，返回 (image, gt_boxes)"""
    img = rng.uniform(0.05, 0.15, (IMG_SIZE, IMG_SIZE)).astype(np.float32)
    if n_objects is None:
        n_objects = rng.integers(1, 4)  # 1~3 个目标
    gt_boxes = []
    for _ in range(n_objects):
        cx = rng.integers(PATCH_SIZE, IMG_SIZE - PATCH_SIZE)
        cy = rng.integers(PATCH_SIZE, IMG_SIZE - PATCH_SIZE)
        cls = rng.integers(N_CLASSES)
        if cls == 0:
            w = rng.integers(8, 18)
            h = rng.integers(8, 18)
            box = draw_rectangle(img, rng, cx, cy, w, h)
        elif cls == 1:
            r = rng.integers(5, 10)
            box = draw_circle(img, rng, cx, cy, r)
        else:
            size = rng.integers(10, 18)
            box = draw_triangle(img, rng, cx, cy, size)
        gt_boxes.append(box)
    img = np.clip(img, 0, 1)
    return img, gt_boxes


def generate_dataset(n_train: int = 200, n_val: int = 50, seed: int = 42):
    rng = np.random.default_rng(seed)
    train = [generate_scene(rng) for _ in range(n_train)]
    val   = [generate_scene(rng) for _ in range(n_val)]
    return train, val


# ─────────────────────── 特征提取 (HOG-like) ────────────────────

def hog_like_features(patch: np.ndarray, cell_size: int = 4) -> np.ndarray:
    """
    简化版 HOG：计算每个 cell 内的梯度方向直方图（8 个方向 bin）
    返回拼接的特征向量
    """
    h, w = patch.shape
    gx = np.gradient(patch, axis=1)
    gy = np.gradient(patch, axis=0)
    mag   = np.sqrt(gx ** 2 + gy ** 2)
    angle = (np.arctan2(gy, gx) * 180 / np.pi) % 180  # 0-180 度，无向
    n_bins = 8

    feats = []
    for r in range(0, h, cell_size):
        for c in range(0, w, cell_size):
            cell_mag   = mag[r:r+cell_size, c:c+cell_size].ravel()
            cell_angle = angle[r:r+cell_size, c:c+cell_size].ravel()
            hist, _ = np.histogram(cell_angle, bins=n_bins, range=(0, 180),
                                   weights=cell_mag)
            hist = hist / (hist.sum() + 1e-6)
            feats.append(hist)

    # 加入统计特征
    feats.append([patch.mean(), patch.std(), patch.max(), patch.min()])
    return np.concatenate(feats).astype(np.float32)


def extract_patch(img: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> np.ndarray:
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.zeros((PATCH_SIZE, PATCH_SIZE), dtype=np.float32)
    # 简单 resize：最近邻插值到 PATCH_SIZE
    rows = np.linspace(0, patch.shape[0] - 1, PATCH_SIZE).astype(int)
    cols = np.linspace(0, patch.shape[1] - 1, PATCH_SIZE).astype(int)
    return patch[np.ix_(rows, cols)]


# ─────────────────────── IoU 工具 ─────────────────────────────

def compute_iou(boxA, boxB) -> float:
    ax1, ay1, ax2, ay2 = boxA
    bx1, by1, bx2, by2 = boxB
    ix1 = max(ax1, bx1); iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2); iy2 = min(ay2, by2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    areaA = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    areaB = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = areaA + areaB - inter
    return inter / (union + 1e-10)


def nms(boxes, scores, iou_thr: float = 0.35) -> list:
    """非极大值抑制，返回保留的下标"""
    if len(boxes) == 0:
        return []
    order = np.argsort(scores)[::-1]
    keep  = []
    while len(order) > 0:
        i = order[0]
        keep.append(i)
        rest = order[1:]
        ious = np.array([compute_iou(boxes[i][:4], boxes[j][:4]) for j in rest])
        order = rest[ious < iou_thr]
    return keep


# ─────────────────────── 简单分类器（Softmax SGD）───────────────

class LinearClassifier:
    def __init__(self, n_features: int, n_classes: int, lr: float = 0.05,
                 n_epochs: int = 20, l2: float = 1e-4, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.W = rng.normal(0, 0.01, (n_features, n_classes)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2 = l2
        self.losses = []

    def _softmax(self, z):
        z = z - z.max(axis=1, keepdims=True)
        e = np.exp(z)
        return e / (e.sum(axis=1, keepdims=True) + 1e-10)

    def fit(self, X: np.ndarray, y: np.ndarray, batch_size: int = 128):
        N = len(X)
        rng = np.random.default_rng(0)
        for epoch in range(self.n_epochs):
            idx = rng.permutation(N)
            total_loss = 0.0
            for s in range(0, N, batch_size):
                bi = idx[s: s + batch_size]
                Xb, yb = X[bi], y[bi]
                probs = self._softmax(Xb @ self.W + self.b)
                loss  = -np.log(probs[np.arange(len(yb)), yb] + 1e-10).mean()
                total_loss += loss * len(yb)
                dp = probs.copy()
                dp[np.arange(len(yb)), yb] -= 1
                dp /= len(yb)
                self.W -= self.lr * (Xb.T @ dp + self.l2 * self.W)
                self.b -= self.lr * dp.sum(axis=0)
            self.losses.append(total_loss / N)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._softmax(X @ self.W + self.b)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.predict_proba(X).argmax(axis=1)


# ─────────────────────── 训练数据构建 ────────────────────────────

def build_training_data(dataset: list, iou_pos_thr: float = 0.5,
                         iou_neg_thr: float = 0.2, max_neg_ratio: int = 3):
    """
    从数据集中提取正/负样本：
      - 与任意 GT box IoU >= pos_thr → 正样本（使用对应 GT 类别）
      - 与所有 GT box IoU <  neg_thr → 负样本（背景类）
    """
    scales = [PATCH_SIZE, PATCH_SIZE + 8, PATCH_SIZE + 16]
    steps  = [8, 8, 12]

    all_X, all_y = [], []
    for img, gt_boxes in dataset:
        for scale, step in zip(scales, steps):
            for y1, x1 in product(range(0, IMG_SIZE - scale, step),
                                   range(0, IMG_SIZE - scale, step)):
                x2, y2 = x1 + scale, y1 + scale
                patch = extract_patch(img, x1, y1, x2, y2)
                feat  = hog_like_features(patch)

                best_iou, best_cls = 0.0, BG_CLASS
                for gt in gt_boxes:
                    iou = compute_iou((x1, y1, x2, y2), gt[:4])
                    if iou > best_iou:
                        best_iou = iou
                        best_cls = gt[4]

                if best_iou >= iou_pos_thr:
                    all_X.append(feat)
                    all_y.append(best_cls)
                elif best_iou < iou_neg_thr:
                    all_X.append(feat)
                    all_y.append(BG_CLASS)

    X = np.array(all_X, dtype=np.float32)
    y = np.array(all_y, dtype=np.int32)

    # 平衡负样本（背景类往往多很多）
    pos_idx = np.where(y != BG_CLASS)[0]
    neg_idx = np.where(y == BG_CLASS)[0]
    np.random.default_rng(42).shuffle(neg_idx)
    neg_idx = neg_idx[:max_neg_ratio * max(1, len(pos_idx))]
    all_idx = np.concatenate([pos_idx, neg_idx])
    np.random.default_rng(42).shuffle(all_idx)
    return X[all_idx], y[all_idx]


# ─────────────────────── 推断（滑动窗口 + NMS）────────────────────

def detect(img: np.ndarray, clf: LinearClassifier,
           conf_thr: float = 0.55, iou_thr: float = 0.35) -> list:
    """返回检测框列表 [(x1,y1,x2,y2,class_id,score)]"""
    scales = [PATCH_SIZE, PATCH_SIZE + 8, PATCH_SIZE + 16]
    steps  = [6, 8, 10]
    candidates = []

    for scale, step in zip(scales, steps):
        for y1, x1 in product(range(0, IMG_SIZE - scale, step),
                               range(0, IMG_SIZE - scale, step)):
            x2, y2 = x1 + scale, y1 + scale
            patch  = extract_patch(img, x1, y1, x2, y2)
            feat   = hog_like_features(patch).reshape(1, -1)
            probs  = clf.predict_proba(feat)[0]
            cls_id = probs.argmax()
            score  = probs[cls_id]
            if cls_id != BG_CLASS and score >= conf_thr:
                candidates.append((x1, y1, x2, y2, int(cls_id), float(score)))

    if not candidates:
        return []
    boxes_arr  = candidates
    scores_arr = [c[5] for c in candidates]
    keep = nms(boxes_arr, scores_arr, iou_thr)
    return [candidates[k] for k in keep]


# ─────────────────────── 评估 ─────────────────────────────────

def eval_detection(dataset: list, clf: LinearClassifier,
                   iou_thr: float = 0.5) -> dict:
    tp, fp, fn_total = 0, 0, 0
    per_class_tp = {c: 0 for c in range(N_CLASSES)}
    per_class_gt = {c: 0 for c in range(N_CLASSES)}

    for img, gt_boxes in dataset:
        dets = detect(img, clf)
        matched_gt = set()
        for d in dets:
            best_iou, best_g = 0.0, -1
            for gi, gt in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                iou = compute_iou(d[:4], gt[:4])
                if iou > best_iou:
                    best_iou, best_g = iou, gi
            if best_iou >= iou_thr and best_g >= 0:
                matched_gt.add(best_g)
                tp += 1
                if d[4] == gt_boxes[best_g][4]:
                    per_class_tp[d[4]] += 1
            else:
                fp += 1
        fn_total += len(gt_boxes) - len(matched_gt)
        for gt in gt_boxes:
            per_class_gt[gt[4]] += 1

    precision = tp / (tp + fp + 1e-10)
    recall    = tp / (tp + fn_total + 1e-10)
    f1        = 2 * precision * recall / (precision + recall + 1e-10)
    return {"precision": precision, "recall": recall, "f1": f1,
            "tp": tp, "fp": fp, "fn": fn_total,
            "per_class_tp": per_class_tp, "per_class_gt": per_class_gt}


# ─────────────────────── 可视化 ─────────────────────────────────

BOX_COLORS = ["#FF6B6B", "#4ECDC4", "#45B7D1"]


def plot_results(clf: LinearClassifier, val_dataset: list, metrics: dict,
                 save_path: str = "results/object_detection_results.png"):
    fig = plt.figure(figsize=(18, 12))
    fig.patch.set_facecolor("#1a1a2e")

    # ── 1. 训练损失 ──
    ax1 = fig.add_subplot(2, 4, 1)
    ax1.plot(clf.losses, color="#FF6B6B", linewidth=2)
    ax1.set_title("Classifier Training Loss", color="white", fontsize=10, fontweight="bold")
    ax1.set_xlabel("Epoch", color="white")
    ax1.set_ylabel("Loss", color="white")
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white")
    for s in ax1.spines.values(): s.set_edgecolor("#444")
    ax1.grid(True, alpha=0.2)

    # ── 2. P/R/F1 ──
    ax2 = fig.add_subplot(2, 4, 2)
    metric_names = ["Precision", "Recall", "F1"]
    vals = [metrics["precision"], metrics["recall"], metrics["f1"]]
    bars = ax2.bar(metric_names, vals, color=["#FF6B6B", "#4ECDC4", "#45B7D1"], alpha=0.85)
    ax2.set_ylim(0, 1.1)
    ax2.set_title(f"Detection @ IoU=0.5", color="white", fontsize=10, fontweight="bold")
    ax2.set_facecolor("#16213e")
    ax2.tick_params(colors="white")
    for s in ax2.spines.values(): s.set_edgecolor("#444")
    for bar, v in zip(bars, vals):
        ax2.text(bar.get_x() + bar.get_width()/2, v + 0.02, f"{v:.3f}",
                 ha="center", color="white", fontsize=9)

    # ── 3. Per-class detection rate ──
    ax3 = fig.add_subplot(2, 4, 3)
    cls_rates = [metrics["per_class_tp"][c] / max(1, metrics["per_class_gt"][c])
                 for c in range(N_CLASSES)]
    ax3.bar(CLASSES, cls_rates, color=BOX_COLORS, alpha=0.85)
    ax3.set_ylim(0, 1.1)
    ax3.set_title("Per-Class Detection Rate", color="white", fontsize=10, fontweight="bold")
    ax3.set_facecolor("#16213e")
    ax3.tick_params(colors="white")
    for s in ax3.spines.values(): s.set_edgecolor("#444")
    for i, (v, cls) in enumerate(zip(cls_rates, CLASSES)):
        ax3.text(i, v + 0.02, f"{v:.2f}", ha="center", color="white", fontsize=9)

    # ── 4. TP/FP/FN ──
    ax4 = fig.add_subplot(2, 4, 4)
    tfn = [metrics["tp"], metrics["fp"], metrics["fn"]]
    ax4.bar(["TP", "FP", "FN"], tfn, color=["#4ECDC4", "#FF6B6B", "#FFA07A"], alpha=0.85)
    ax4.set_title("TP / FP / FN Count", color="white", fontsize=10, fontweight="bold")
    ax4.set_facecolor("#16213e")
    ax4.tick_params(colors="white")
    for s in ax4.spines.values(): s.set_edgecolor("#444")

    # ── 5-8. 检测结果示例 ──
    n_show = 4
    sample_idx = np.random.default_rng(42).choice(len(val_dataset), n_show, replace=False)
    for k, si in enumerate(sample_idx):
        ax = fig.add_subplot(2, 4, 5 + k)
        img, gt_boxes = val_dataset[si]
        dets = detect(img, clf)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        # GT boxes（绿色虚线）
        for gt in gt_boxes:
            x1, y1, x2, y2, cls_id = gt
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                      linewidth=1.5, edgecolor="lime",
                                      facecolor="none", linestyle="--")
            ax.add_patch(rect)
            ax.text(x1, y1 - 2, CLASSES[cls_id][0].upper(), color="lime", fontsize=7)
        # 检测框（彩色实线）
        for det in dets:
            x1, y1, x2, y2, cls_id, score = det
            color = BOX_COLORS[cls_id]
            rect  = patches.Rectangle((x1, y1), x2-x1, y2-y1,
                                       linewidth=1.5, edgecolor=color, facecolor="none")
            ax.add_patch(rect)
            ax.text(x1, y2 + 4, f"{CLASSES[cls_id][0].upper()}{score:.2f}",
                    color=color, fontsize=6)
        ax.set_title(f"Scene {si}", color="white", fontsize=9)
        ax.axis("off")
        ax.set_facecolor("#16213e")

    fig.text(0.5, 0.01,
             "Green dashed = GT boxes  |  Colored solid = Detections  "
             "|  R=Rectangle  C=Circle  T=Triangle",
             ha="center", color="#aaa", fontsize=8)
    plt.suptitle("Object Detection — Sliding Window + HOG + NMS",
                 color="white", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  图表已保存至: {save_path}")


# ─────────────────────── 主函数 ─────────────────────────────────

def object_detection():
    """目标检测完整流程（合成数据 + 滑动窗口 + NMS）"""
    print("=" * 60)
    print("   目标检测 (Object Detection) — 滑动窗口 + HOG + NMS")
    print("=" * 60)

    # ── 1. 生成数据 ──
    print("\n[1/5] 生成合成场景数据集...")
    train_data, val_data = generate_dataset(n_train=200, n_val=50, seed=42)
    total_gt = sum(len(gt) for _, gt in train_data)
    print(f"  训练场景: {len(train_data)}, GT 框总数: {total_gt}")
    print(f"  验证场景: {len(val_data)}")

    # ── 2. 构建训练样本 ──
    print("\n[2/5] 提取候选窗口特征（HOG-like）...")
    X_train, y_train = build_training_data(train_data)
    unique, counts = np.unique(y_train, return_counts=True)
    cls_names_bg = CLASSES + ["Background"]
    print(f"  训练样本: {len(X_train)}, 特征维度: {X_train.shape[1]}")
    for uid, cnt in zip(unique, counts):
        print(f"    {cls_names_bg[uid]:<12}: {cnt}")

    # ── 3. 训练分类器 ──
    print("\n[3/5] 训练线性分类器 (Softmax)...")
    clf = LinearClassifier(n_features=X_train.shape[1],
                           n_classes=N_CLASSES + 1,   # +1 for background
                           lr=0.1, n_epochs=30, l2=1e-4, seed=42)
    clf.fit(X_train, y_train)
    print(f"  最终 Loss: {clf.losses[-1]:.4f}")

    # ── 4. 检测评估 ──
    print("\n[4/5] 评估检测效果 (IoU @ 0.5)...")
    metrics = eval_detection(val_data, clf, iou_thr=0.5)
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall   : {metrics['recall']:.4f}")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}  FN={metrics['fn']}")

    # ── 5. 可视化 ──
    print("\n[5/5] 生成可视化图表...")
    import os
    os.makedirs("results", exist_ok=True)
    plot_results(clf, val_data, metrics,
                 save_path="results/object_detection_results.png")

    print("\n[DONE] 目标检测完成!")
    return clf, metrics


if __name__ == "__main__":
    object_detection()
