import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.grids import (
    build_models,
    actions,
    plot_values_and_policy,
    plot_values_and_policy_gif,
)

# 定义网格参数
grid_size = 5

target_areas = [(3, 2)]  # 终止和危险状态
forbidden_areas = [(1, 1), (1, 2), (2, 2), (3, 1), (3, 3), (4, 1)]  # 禁止状态

states, p_r, p_s_prime = build_models(
    grid_size, target_areas, forbidden_areas, success_prob=1
)
v_initial = {s: 0 for s in states}
p_initial = {s: {"still": 1.0} for s in states}


def value_iteration(
    p_r,
    p_s_prime,
    v_init,
    states,
    target_areas,
    actions,
    gamma,
    threshold=1e-5,
    max_iter=100,
    save_history=False,
):
    v = v_init.copy()
    p = None
    if save_history:
        v_history = []
        p_history = []
    for k in range(max_iter):
        delta = 0
        value_new = {}
        policy_new = {}
        for s in states:
            policy_new[s] = {}
            q_list = []
            for a in actions:
                # ------------------------------------------------------------------------------------
                # 计算q_k(s, a)的值。
                # q_k(s, a) = expected reward + gamma * expected value of next state
                # 建议使用列表推导来实现下面的代码
                # 参考信息：https://docs.python.org/zh-cn/3.13/tutorial/datastructures.html#list-comprehensions
                # 可以通过调取p_r的值来计算expected_r, 然后通过列表推导计算expected_v，最后计算q_k(s, a)
                # 将(q, a) 存入q_list，然后取最大值，更新policy和value
                # Expected code: ~4 lines
                # ------------------------------------------------------------------------------------
                expected_r = sum(prob * r for r, prob in p_r[s][a].items())
                expected_v = sum(
                    prob * v[s_prime]
                    for s_prime, prob in p_s_prime[s][a].items()
                )
                q = expected_r + gamma * expected_v
                q_list.append((q, a))
                # ------------------------------------------------------------------------------------
                # End of code snippet
                # ------------------------------------------------------------------------------------
            max_q, argmax_q_a = max(q_list, key=lambda x: x[0])
            # policy update
            policy_new[s][argmax_q_a] = 1.0
            # value update
            value_new[s] = max_q
        # norm between old and new value function
        delta = (sum((v[s] - value_new[s]) ** 2 for s in states)) ** 0.5
        if save_history:
            v_history.append(value_new.copy())
            p_history.append(policy_new.copy())
        v = value_new.copy()
        p = policy_new.copy()
        if delta < threshold:
            break
    return_dict = {
        "v": v,
        "p": p,
    }
    if save_history:
        return_dict["v_history"] = v_history
        return_dict["p_history"] = p_history
    return return_dict


# 运行算法
print("Running Value Iteration...")
gamma = 0.9
return_dict = value_iteration(
    p_r,
    p_s_prime,
    v_initial,
    states,
    target_areas,
    actions,
    gamma,
    save_history=True,
)
v_optimal = return_dict["v"]
policy_optimal = return_dict["p"]
v_history = return_dict["v_history"]
p_history = return_dict["p_history"]

print("Start plotting...")
plot_values_and_policy_gif(
    v_history,
    p_history,
    forbidden_areas,
    target_areas,
    gif_save_path=os.path.join(
        os.path.dirname(__file__), "figs/Value_iteration.gif"
    ),
    verbose=True,
)
plot_values_and_policy(
    value_dict=v_history[-1],
    policy_dict=p_history[-1],
    forbidden_cells=forbidden_areas,
    target_cells=target_areas,
    title="State Value and Policy at Final Iteration",
    save_path=os.path.join(
        os.path.dirname(__file__), "figs/Final_value_iteration.png"
    ),
)
