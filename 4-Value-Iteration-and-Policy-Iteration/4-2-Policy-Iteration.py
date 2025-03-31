import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.grids import (
    GridWorld,
    plot_values_and_policy,
    plot_values_and_policy_gif,
)

from utils.test_utils import grid_load_json, check_value_and_policy


def policy_evaluation(
    env,
    policy,
    gamma,
    threshold=1e-5,
):
    """
    策略评估：计算当前策略下的状态价值函数
    """
    v = {s: 0.0 for s in env.get_states()}
    while True:
        delta = 0
        value_new = {}
        for s in env.get_states():
            total = 0
            # 遍历所有动作（根据策略的分布）
            for a in policy[s]:
                if policy[s][a] == 0:
                    continue  # 概率为0的动作可直接跳过
                # ------------------------------------------------------------------------------------
                # 通过计算每一个action的总和来计算v_new(s)
                # v_new(s) = ∑_a policy(s, a) * (∑_r p ( r | s, a) * r + gamma * ∑_{s'} p (s' | s, a) * v(s'))
                # 使用env.get_reward_probs和env.get_transition_probs来计算expected_r和expected_v
                # 参考信息:  utils/grids.py::GridWorld::get_reward_probs
                #           utils/grids.py::GridWorld::get_transition_probs
                # 建议使用列表推导来实现下面的代码
                # 参考信息：https://docs.python.org/zh-cn/3.13/tutorial/datastructures.html#list-comprehensions
                # 先通过p_r计算expected_r，再通过p_s_prime计算expected_next_v，累积加权到total
                # 最后在循环外更新value_new[s]
                # Expected code: ~3 lines
                # ------------------------------------------------------------------------------------
                expected_r = sum(
                    prob * r for r, prob in env.get_reward_probs(s, a).items()
                )
                expected_next_v = sum(
                    prob * v[s_prime]
                    for s_prime, prob in env.get_transition_probs(s, a).items()
                )
                total += policy[s][a] * (expected_r + gamma * expected_next_v)
                # ------------------------------------------------------------------------------------
                # End of code snippet
                # ------------------------------------------------------------------------------------
            value_new[s] = total
        delta = (
            sum((v[s] - value_new[s]) ** 2 for s in env.get_states())
        ) ** 0.5
        # 判断是否收敛
        if delta < threshold:
            break
        v = value_new.copy()
    return v


def policy_improvement(env, v, gamma):
    """
    策略改进：根据当前值函数生成新策略
    """
    new_policy = {}
    for s in env.get_states():
        new_policy[s] = {}
        q_list = []
        # 遍历所有可能的动作
        for a in env.get_actions(s):
            # ------------------------------------------------------------------------------------
            # 计算q ( s, a )
            # q ( s, a ) = expected_r ( s, a ) + gamma * expected_next_v ( s, a )
            # 使用env.get_reward_probs和env.get_transition_probs来计算expected_r和expected_v
            # 参考信息:  utils/grids.py::GridWorld::get_reward_probs
            #           utils/grids.py::GridWorld::get_transition_probs
            # 建议使用列表推导来实现下面的代码
            # 参考信息：https://docs.python.org/zh-cn/3.13/tutorial/datastructures.html#list-comprehensions
            # 先通过p_r计算expected_r，再通过p_s_prime计算expected_next_v，累积加权到q
            # 将(q, a) 存入q_list
            # Expected code: ~4 lines
            # ------------------------------------------------------------------------------------
            expected_r = sum(
                prob * r for r, prob in env.get_reward_probs(s, a).items()
            )
            expected_next_v = sum(
                prob * v[s_prime]
                for s_prime, prob in env.get_transition_probs(s, a).items()
            )
            q = expected_r + gamma * expected_next_v
            q_list.append((q, a))
            # ------------------------------------------------------------------------------------
            # End of code snippet
            # ------------------------------------------------------------------------------------
        argmax_q_a = max(q_list, key=lambda x: x[0])[1]
        # 生成确定性策略（当前最优动作的概率为1）
        new_policy[s][argmax_q_a] = 1.0
    return new_policy


def policy_iteration(
    env,
    gamma,
    initial_policy=None,
    threshold=1e-5,
    save_history=False,
):
    # 初始化随机策略（均匀分布）
    if initial_policy is None:
        initial_policy = {s: {"still": 1.0} for s in env.get_states()}

    if save_history:
        v_history = []
        p_history = []

    policy_k = initial_policy.copy()
    v_policy_k_minus_1 = None
    while True:
        # 1. 策略评估
        v_policy_k = policy_evaluation(env, policy_k, gamma)
        # 2. 策略改进
        policy_k_plus_1 = policy_improvement(env, v_policy_k, gamma)
        # 检查策略是否稳定
        if v_policy_k_minus_1 is not None:
            delta_v = (
                sum(
                    (v_policy_k[s] - v_policy_k_minus_1[s]) ** 2
                    for s in env.get_states()
                )
                ** 0.5
            )
        else:
            delta_v = float("inf")
        if delta_v < threshold:
            break
        v_policy_k_minus_1 = v_policy_k.copy()
        policy_k = policy_k_plus_1.copy()
        if save_history:
            v_history.append(v_policy_k)
            p_history.append(policy_k)

    return_dict = {
        "v": v_policy_k,
        "p": policy_k,
    }
    if save_history:
        return_dict["v_history"] = v_history
        return_dict["p_history"] = p_history
    return return_dict


example_root = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "examples")
)
grid_example_list = [
    os.path.join(example_root, "grids_5.json"),
]
for grid_example in grid_example_list:
    grid_dict = grid_load_json(grid_example)
    grid_size = grid_dict["grid_size"]
    target_areas = grid_dict["target_areas"]
    forbidden_areas = grid_dict["forbidden_areas"]
    expected_optimal_v = grid_dict["optimal_value"]
    expected_optimal_p = grid_dict["optimal_policy"]

    env = GridWorld(grid_size, target_areas, forbidden_areas, success_prob=1.0)
    v_initial = {s: 0 for s in env.get_states()}
    p_initial = {s: {"still": 1.0} for s in env.get_states()}

    # 运行算法
    print("Running Value Iteration...")
    gamma = 0.9
    return_dict = policy_iteration(
        env=env,
        gamma=gamma,
        save_history=True,
    )
    v_optimal = return_dict["v"]
    policy_optimal = return_dict["p"]
    v_history = return_dict.get("v_history", None)
    p_history = return_dict.get("p_history", None)

    # 检查v_optimal和expected_optimal_v是否一致
    check_value_and_policy(
        generated_value=v_optimal,
        expected_value=expected_optimal_v,
        check_type="value",
    )
    # 检查policy_optimal和expected_optimal_p是否一致
    check_value_and_policy(
        generated_policy=policy_optimal,
        expected_policy=expected_optimal_p,
        check_type="policy",
    )
    print(f"Pass Value Iteration Test!")

    print("Start plotting...")
    if v_history is not None and p_history is not None:
        plot_values_and_policy_gif(
            v_history,
            p_history,
            forbidden_areas,
            target_areas,
            gif_save_path=os.path.join(
                os.path.dirname(__file__), "figs/Policy_Iteration.gif"
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
            os.path.dirname(__file__), "figs/Final_policy_iteration.png"
        ),
    )
