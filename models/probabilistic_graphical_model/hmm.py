# HMM 隐马尔可夫模型
# 观测序列由不可见的状态序列生成，状态之间满足马尔可夫性质

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from hmmlearn import hmm
from utils import get_results_path, save_and_close


def generate_hmm_data(n_samples=300, seed=42):
    """
    生成 HMM 数据（天气模型）：
      隐藏状态: 晴天(0), 雨天(1), 阴天(2)
      观测:    活动: 散步(0), 购物(1), 健身(2)
    """
    np.random.seed(seed)
    # 状态转移矩阵
    A = np.array([
        [0.6, 0.2, 0.2],  # 晴天 -> 晴/雨/阴
        [0.3, 0.4, 0.3],
        [0.2, 0.3, 0.5],
    ])
    # 发射矩阵（观测概率）
    B = np.array([
        [0.5, 0.2, 0.3],  # 晴天 -> 散步/购物/健身
        [0.2, 0.6, 0.2],
        [0.3, 0.3, 0.4],
    ])
    # 初始分布
    pi = np.array([0.5, 0.3, 0.2])

    states = []
    obs    = []
    s = np.random.choice([0,1,2], p=pi)
    for _ in range(n_samples):
        s = np.random.choice([0,1,2], p=A[s])
        o = np.random.choice([0,1,2], p=B[s])
        states.append(s)
        obs.append(o)
    return np.array(states), np.array(obs)


def hmm():
    """HMM 隐马尔可夫模型实现（天气-活动序列）"""
    print("HMM 隐马尔可夫模型运行中...\n")

    # 1. 数据准备
    print("1. 生成 HMM 数据...")
    n_samples = 500
    true_states, obs_seq = generate_hmm_data(n_samples)
    print(f"   样本数: {n_samples}")
    print("   隐藏状态映射: 0=晴天, 1=雨天, 2=阴天")
    print("   观测映射: 0=散步, 1=购物, 2=健身")

    # 2. 训练 HMM (GaussianHMM：观测建模为高斯分布）
    print("2. 训练 HMM (GaussianHMM)...")
    # 为观测添加一点高斯噪声
    obs_gaussian = obs_seq[:, np.newaxis].astype(float) + \
                   np.random.normal(0, 0.1, (n_samples, 1))

    model = hmm.GaussianHMM(
        n_components=3,
        covariance_type='diag',
        n_iter=100,
        random_state=42,
        init_params='stmc'  # 初始状态/转移/均值/协方差
    )
    model.fit(obs_gaussian)
    print(f"   收敛: {model.monitor_.converged}")

    # 3. 解码（维特比算法）
    print("3. 维特比解码（估计隐藏状态）...")
    log_prob, pred_states = model.decode(obs_gaussian)
    accuracy = np.mean(pred_states == true_states)
    print(f"   状态估计准确率: {accuracy:.4f}")

    # 4. 不同隐状态数对比
    print("4. 不同 n_components 对比...")
    comp_range = [2, 3, 4, 5]
    ll_scores = []
    for n_c in comp_range:
        m = hmm.GaussianHMM(n_components=n_c, covariance_type='diag',
                           n_iter=50, random_state=42)
        try:
            m.fit(obs_gaussian)
            ll_scores.append(m.score(obs_gaussian))
            print(f"   n_components={n_c}  log_likelihood={ll_scores[-1]:.1f}")
        except Exception as e:
            print(f"   n_components={n_c}  跳过: {e}")
            ll_scores.append(np.nan)

    # 5. 可视化
    print("5. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    state_names = ['晴天', '雨天', '阴天']
    obs_names   = ['散步', '购物', '健身']

    # ── 子图1：真实 vs 预测状态序列（前 50 步）──
    ax = axes[0]
    n_show = 50
    ax.step(range(n_show), true_states[:n_show], label='真实状态',
            color='steelblue', where='mid')
    ax.step(range(n_show), pred_states[:n_show], label='HMM 预测',
            color='tomato', linestyle='--', where='mid')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(state_names)
    ax.set_title(f'前 {n_show} 步：真实 vs 预测状态（准确率={accuracy:.3f}）')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Hidden State')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图2：观测序列 ──
    ax = axes[1]
    ax.scatter(range(n_show), obs_seq[:n_show], c=obs_seq[:n_show],
                cmap='coolwarm', s=60, alpha=0.8, edgecolors='k')
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(obs_names)
    ax.set_title(f'观测序列（前 {n_show} 步）')
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Observation')
    ax.grid(True, alpha=0.3, axis='y')

    # ── 子图3：n_components 对数似然 ──
    ax = axes[2]
    ax.plot(comp_range, ll_scores, 'bo-', linewidth=1.8)
    ax.axvline(x=3, color='green', linestyle='--', label='真实值=3')
    ax.set_title('n_components 对数似然曲线')
    ax.set_xlabel('Number of Hidden States')
    ax.set_ylabel('Log Likelihood')
    ax.legend()
    ax.grid(True, alpha=0.3)

    save_path = get_results_path('hmm_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    hmm()
