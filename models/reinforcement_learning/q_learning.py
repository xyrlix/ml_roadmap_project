# Q-Learning 强化学习模型
# 基于值迭代的无模型强化学习，在 GridWorld 迷宫中求最优策略

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from utils import get_results_path, save_and_close


# ──────────────────────────────────────────────
# GridWorld 环境
# ──────────────────────────────────────────────
class GridWorld:
    """
    5×5 GridWorld:
      S = 起点(0,0)  G = 终点(4,4)
      陷阱格：(1,1), (2,3), (3,1)
    动作: 0=上, 1=下, 2=左, 3=右
    """
    ROWS, COLS = 5, 5
    ACTIONS = 4   # 上下左右
    TRAPS   = {(1, 1), (2, 3), (3, 1)}
    START   = (0, 0)
    GOAL    = (4, 4)

    def __init__(self):
        self.state = self.START

    def reset(self):
        self.state = self.START
        return self._encode(self.state)

    def _encode(self, s):
        return s[0] * self.COLS + s[1]

    def step(self, action):
        r, c = self.state
        if action == 0:   r -= 1
        elif action == 1: r += 1
        elif action == 2: c -= 1
        elif action == 3: c += 1
        # 边界处理（撞墙原地）
        r = max(0, min(r, self.ROWS - 1))
        c = max(0, min(c, self.COLS - 1))
        self.state = (r, c)

        if self.state == self.GOAL:
            return self._encode(self.state), +10.0, True
        elif self.state in self.TRAPS:
            return self._encode(self.state), -5.0, True
        else:
            return self._encode(self.state), -0.1, False


def q_learning():
    """Q-Learning 强化学习实现（GridWorld 迷宫）"""
    print("Q-Learning 强化学习模型运行中...\n")

    env = GridWorld()
    N_STATES  = env.ROWS * env.COLS
    N_ACTIONS = env.ACTIONS

    # 超参数
    EPISODES   = 2000
    ALPHA      = 0.1     # 学习率
    GAMMA      = 0.95    # 折扣因子
    EPS_START  = 1.0     # ε-greedy 初始值
    EPS_MIN    = 0.05
    EPS_DECAY  = 0.998

    # 初始化 Q 表
    Q = np.zeros((N_STATES, N_ACTIONS))

    # 1. 训练
    print("1. 训练 Q-Learning 智能体...")
    episode_rewards = []
    episode_lengths = []
    epsilon = EPS_START

    for ep in range(EPISODES):
        state = env.reset()
        total_r = 0
        steps   = 0
        done    = False
        while not done and steps < 100:
            # ε-greedy 策略
            if np.random.random() < epsilon:
                action = np.random.randint(N_ACTIONS)
            else:
                action = np.argmax(Q[state])

            next_state, reward, done = env.step(action)
            # Q 值更新
            td_target = reward + GAMMA * np.max(Q[next_state]) * (1 - int(done))
            Q[state, action] += ALPHA * (td_target - Q[state, action])

            state   = next_state
            total_r += reward
            steps   += 1

        epsilon = max(EPS_MIN, epsilon * EPS_DECAY)
        episode_rewards.append(total_r)
        episode_lengths.append(steps)

        if (ep + 1) % 500 == 0:
            avg_r = np.mean(episode_rewards[-100:])
            print(f"   Episode {ep+1:4d}  avg_reward(100)={avg_r:.2f}  "
                  f"epsilon={epsilon:.3f}")

    # 2. 提取最优路径
    print("2. 提取最优策略路径...")
    env_test = GridWorld()
    state = env_test.reset()
    path = [env_test.START]
    done = False
    for _ in range(50):
        action = np.argmax(Q[state])
        state, _, done = env_test.step(action)
        r, c = env_test.state
        path.append((r, c))
        if done:
            break
    print(f"   最优路径长度: {len(path)} 步  {'成功到达终点' if env_test.state == env_test.GOAL else '未到终点'}")

    # 3. 可视化
    print("3. 可视化结果...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── 子图1：Q 值最大动作热力图（GridWorld）──
    ax = axes[0]
    V = np.max(Q, axis=1).reshape(env.ROWS, env.COLS)
    im = ax.imshow(V, cmap='YlOrRd', vmin=V.min(), vmax=V.max())
    plt.colorbar(im, ax=ax)

    # 绘制箭头（最优动作）
    action_arrows = {0: (-0.3, 0), 1: (0.3, 0), 2: (0, -0.3), 3: (0, 0.3)}
    for r in range(env.ROWS):
        for c in range(env.COLS):
            s = r * env.COLS + c
            best_a = np.argmax(Q[s])
            dr, dc = action_arrows[best_a]
            ax.annotate('', xy=(c + dc, r + dr), xytext=(c, r),
                        arrowprops=dict(arrowstyle='->', color='black', lw=1.5))

    # 标记特殊格
    for (tr, tc) in env.TRAPS:
        ax.add_patch(plt.Rectangle((tc-0.5, tr-0.5), 1, 1,
                                   fill=True, color='gray', alpha=0.5))
    gr, gc = env.GOAL
    ax.add_patch(plt.Rectangle((gc-0.5, gr-0.5), 1, 1,
                                fill=True, color='lime', alpha=0.5))
    ax.text(gc, gr, 'G', ha='center', va='center', fontsize=14, fontweight='bold')

    # 画最优路径
    if len(path) > 1:
        pr, pc = zip(*path)
        ax.plot(pc, pr, 'b-o', linewidth=2, markersize=6, zorder=5, label='最优路径')
    ax.set_title('Q-Learning GridWorld\n最优策略箭头图')
    ax.set_xticks(range(env.COLS))
    ax.set_yticks(range(env.ROWS))
    ax.legend(fontsize=8)

    # ── 子图2：学习曲线（奖励滑动平均）──
    ax = axes[1]
    window = 50
    smooth_r = np.convolve(episode_rewards,
                            np.ones(window)/window, mode='valid')
    ax.plot(episode_rewards, alpha=0.2, color='steelblue', label='Episode Reward')
    ax.plot(np.arange(window-1, EPISODES), smooth_r,
            color='steelblue', linewidth=2, label=f'{window}-ep 滑动均值')
    ax.set_title('Q-Learning 学习曲线')
    ax.set_xlabel('Episode')
    ax.set_ylabel('Total Reward')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── 子图3：Q 值收敛热力图 ──
    ax = axes[2]
    Q_display = Q.copy()
    im2 = ax.imshow(Q_display, cmap='RdBu', aspect='auto')
    plt.colorbar(im2, ax=ax)
    ax.set_title('Q 值矩阵 (State × Action)')
    ax.set_xlabel('Action (0=上,1=下,2=左,3=右)')
    ax.set_ylabel('State (row×COLS + col)')

    save_path = get_results_path('q_learning_result.png')
    save_and_close(save_path)
    print(f"   图表已保存: {save_path}")


if __name__ == "__main__":
    q_learning()
