

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

# 动作空间 [上, 下, 左, 右]
actions = ['up', 'down', 'left', 'right', "still"]

# 动作对应的坐标变化
action_effects = {
    'up': (-1, 0),
    'down': (1, 0),
    'left': (0, -1),
    'right': (0, 1),
    "still": (0, 0)
}

def get_states(grid_size):
    return [(i, j) for i in range(grid_size) for j in range(grid_size)]


def get_transition_probs(s, a, grid_size, success_prob=0.8):
    """计算状态转移概率，包含随机性"""
    transitions = {}
    original_effect = action_effects[a]
    # 预期动作的转移
    intended_next = (
        (s[0] + original_effect[0]),
        (s[1] + original_effect[1])
    )
    transitions[intended_next] = success_prob
    
    # 其他动作的随机性（总概率 1 - success_prob，均匀分配）
    for other_a in action_effects:
        if other_a != a:
            effect = action_effects[other_a]
            next_state = (
                (s[0] + effect[0]),
                (s[1] + effect[1])
            )
            prob = (1 - success_prob) / (len(action_effects) - 1)
            transitions[next_state] = transitions.get(next_state, 0) + prob
    
    return transitions

# 构建 p_r 和 p_s_prime
def build_models(grid_size, terminals, forbiddens, success_prob=0.8):
    states = get_states(grid_size)
    p_r = {s: {a: {} for a in actions} for s in states}
    p_s_prime = {s: {a: {} for a in actions} for s in states}
    
    for s in states:
        if s in terminals:
            # 终止状态无动作，固定奖励
            for a in actions:
                if a == 'still':
                    continue
                p_r[s][a] = {0: 1.0}
                p_s_prime[s][a] = {s: 1.0}  # 终止状态不转移
            continue
            
        for a in actions:
            transitions = get_transition_probs(s, a, grid_size, success_prob)
            p_s_prime[s][a] = transitions
            
            # 根据转移后的状态 s' 计算奖励分布
            reward_dict = {}
            for s_prime, prob in transitions.items():
                # 根据 s' 是否是终止或禁区确定奖励。示例逻辑：
                if s_prime in terminals:
                    r = 10   # 到达终止状态奖励
                elif s_prime in forbiddens:
                    r = -10  # 掉入危险区惩罚
                elif s_prime[0] < 0 or s_prime[0] >= grid_size or s_prime[1] < 0 or s_prime[1] >= grid_size:
                    r = -100  # 越界惩罚
                else:
                    r = 0   # 普通步长惩罚
                
                # 将奖励概率累加
                if r in reward_dict:
                    reward_dict[r] += prob
                else:
                    reward_dict[r] = prob
            
            p_r[s][a] = reward_dict
    
    return states, p_r, p_s_prime

def plot_values(v_dict, title, save_path=None):
    grid_size = int(np.sqrt(len(v_dict)))
    grid = np.zeros((grid_size, grid_size))
    for (i, j), val in v_dict.items():
        grid[i][j] = val
    plt.figure(figsize=(6,6))
    plt.imshow(grid, cmap='viridis', origin='upper')
    plt.colorbar()
    plt.title(title)
    plt.axis('off')
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, f"{grid[i, j]:.1f}", ha='center', va='center', color='white')

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    plt.close()

def plot_policy(policy_dict):
    grid_size = int(np.sqrt(len(policy_dict)))
    action_symbols = {
        'up': '↑', 'down': '↓', 'left': '←', 'right': '→',
        'still': '⬤'
    }
    grid = np.empty((grid_size, grid_size), dtype=str)
    for (i, j), a in policy_dict.items():
        grid[i][j] = action_symbols[a]
    plt.figure(figsize=(6,6))
    plt.imshow(np.zeros((grid_size, grid_size)), cmap='Blues')
    plt.title("Optimal Policy")
    for i in range(grid_size):
        for j in range(grid_size):
            plt.text(j, i, grid[i, j], ha='center', va='center', fontsize=15)
    plt.axis('off')
    plt.show()