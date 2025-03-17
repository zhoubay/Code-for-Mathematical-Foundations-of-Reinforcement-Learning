

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb


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

TERMINAL_SCORE = 10
DANGEROUS_SCORE = -10
BOUNDARY_SCORE = -1
EVERY_SCORE = 0

def bounce_back(s, grid_size):
    if s[0] < 0:
        return (0, s[1])
    elif s[0] >= grid_size:
        return (grid_size-1, s[1])
    if s[1] < 0:
        return (s[0], 0)
    elif s[1] >= grid_size:
        return (s[0], grid_size-1)
    return s

def get_states(grid_size):
    return [(i, j) for i in range(grid_size) for j in range(grid_size)]


def get_transition_probs(s, a, grid_size, success_prob=0.8):
    """计算状态转移概率，包含随机性"""
    transitions = {}
    original_effect = action_effects[a]
    # 预期动作的转移
    intended_next = (
        max(0, min(grid_size-1, s[0] + original_effect[0])),
        max(0, min(grid_size-1, s[1] + original_effect[1]))
    )
    transitions[intended_next] = success_prob
    
    # 其他动作的随机性（总概率 1 - success_prob，均匀分配）
    for other_a in action_effects:
        if other_a != a:
            effect = action_effects[other_a]
            next_state = (
                max(0, min(grid_size-1, s[0] + effect[0])),
                max(0, min(grid_size-1, s[1] + effect[1]))
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
        # if s in terminals:
        #     # 终止状态无动作，固定奖励
        #     for a in actions:
        #         if a == 'still':
        #             continue
        #         p_r[s][a] = {0: 1.0}
        #         p_s_prime[s][a] = {s: 1.0}  # 终止状态不转移
        #     continue
            
        for a in actions:
            transitions = get_transition_probs(s, a, grid_size, success_prob)
            p_s_prime[s][a] = transitions
            expected_next_state = (
                s[0] + action_effects[a][0],
                s[1] + action_effects[a][1]
            )
            
            # 根据转移后的状态 (s, a) 计算奖励分布，因为 p (r | s, a) = ∑_{s'} (p (r | s', s, a) * p (s' | s, a))
            # p (s' | s, a) 为 transitions
            # p (r | s', s, a) 为reward的概率，一般情况下，如果s'确定的话，那么就意味着r也确定了。在特殊情况下，即(s, a)越界，那就不能根据s'计算reward
            # 并且假设p (r | s', s, a)为deterministic的
            reward_dict = {}
            for s_prime, prob in transitions.items():
                # 根据 s' 是否是终止或禁区确定奖励。
                if s_prime in terminals:
                    r = TERMINAL_SCORE   # 到达终止状态奖励
                elif s_prime in forbiddens:
                    r = DANGEROUS_SCORE  # 掉入危险区惩罚
                # 如果s_prime 与 expected_next_state不相同，意味着出界
                elif s_prime != expected_next_state:
                    r = BOUNDARY_SCORE  # 越界惩罚
                else:
                    r = EVERY_SCORE   # 普通步长惩罚
                
                # 将奖励概率累加
                if r in reward_dict:
                    reward_dict[r] += prob
                else:
                    reward_dict[r] = prob
            
            p_r[s][a] = reward_dict
    
    return states, p_r, p_s_prime
def plot_values_and_policy(value_dict, policy_dict, 
                           forbidden_cells=None, target_cells=None,
                           forbidden_color='orange', target_color='limegreen', 
                           bg_color='white', title=None, save_path=None, 
                           action_symbols=None, value_format="{:.1f}", 
                           fontsize=12, figsize=(12, 6)):
    """并排绘制数值表和策略表，通用背景色设置
    
    Args:
        value_dict: {(i,j): float} 数值字典
        policy_dict: {(i,j): str} 策略动作字典
        forbidden_cells: list 禁区坐标列表
        target_cells: list 目标区坐标列表
        forbidden_color: 禁区背景色
        target_color: 目标区背景色
        bg_color: 常规背景色
    """
    # 初始化区域设置
    forbidden_cells = forbidden_cells or []
    target_cells = target_cells or []
    
    # 确定网格尺寸
    grid_size = int(np.sqrt(len(value_dict)))
    value_grid = np.zeros((grid_size, grid_size))
    policy_grid = np.empty((grid_size, grid_size), dtype='object')
    
    # 填充数据
    for (i, j), val in value_dict.items():
        value_grid[i, j] = val
    for (i, j), action in policy_dict.items():
        policy_grid[i, j] = action
    
    # 创建画布和子图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # 通用绘图函数
    def plot_grid(ax, data, is_value=True):
        """核心绘图函数，包含背景色和文本逻辑"""
        # 生成背景色矩阵
        bg_colors = np.full((grid_size, grid_size, 3), to_rgb(bg_color))
        for (i, j) in forbidden_cells:
            bg_colors[i, j] = to_rgb(forbidden_color)
        for (i, j) in target_cells:
            bg_colors[i, j] = to_rgb(target_color)
        
        # 绘制背景色
        ax.imshow(bg_colors, origin='upper')
        
        # 添加单元格数据
        for i in range(grid_size):
            for j in range(grid_size):
                bg_rgb = bg_colors[i, j]
                brightness = np.dot(bg_rgb, [0.299, 0.587, 0.114])
                text_color = 'black' if brightness > 0.5 else 'white'
                
                if is_value:
                    # 数值文本
                    ax.text(j, i, value_format.format(data[i,j]), 
                           ha='center', va='center', color=text_color, fontsize=fontsize)
                else:
                    # 策略符号
                    action_dict = data[i, j]
                    # 选出概率最高的action
                    action = max(action_dict, key=action_dict.get)
                    symbol = action_symbols.get(action, action)
                    ax.text(j, i, symbol, 
                           ha='center', va='center', color=text_color, fontsize=fontsize)
    
    # 绘制左图（数值表）
    plot_grid(ax1, value_grid, is_value=True)
    ax1.set_title("State Values")
    ax1.axis('off')
    
    # 绘制右图（策略表）
    if action_symbols is None:
        action_symbols = {'up':'↑', 'down':'↓', 'left':'←', 'right':'→', 'still':'⬤'}
    plot_grid(ax2, policy_grid, is_value=False)
    ax2.set_title("Policy")
    ax2.axis('off')
    
    # 全局标题
    if title:
        plt.suptitle(title, y=0.95)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
    else:
        plt.show()
    plt.close()
