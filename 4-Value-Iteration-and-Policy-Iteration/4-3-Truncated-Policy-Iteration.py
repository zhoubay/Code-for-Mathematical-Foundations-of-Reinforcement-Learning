import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.grids import (
    GridWorld,
    plot_values_and_policy,
    plot_values_and_policy_gif,
)

from utils.test_utils import grid_load_json, check_value_and_policy


def truncated_policy_evaluation(env, policy, gamma, initial_v, j_truncate):
    """
    部分策略评估：固定迭代次数而非完全收敛

    Args:
        initial_v: 初始价值函数 {s: float}
        j_truncate (int): 最大迭代步数
    """
    v = initial_v.copy()
    for _ in range(j_truncate):
        new_v = {}
        for s in env.get_states():
            total = 0.0
            for a in policy[s]:
                if policy[s][a] == 0:
                    continue
                # ------------------------------------------------------------------------------------
                # 通过计算每一个action的总和来计算v_new(s)
                # v_new(s) = ∑_a policy(s, a) * (∑_r p ( r | s, a) * r + gamma * ∑_{s'} p (s' | s, a) * v(s'))
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
            new_v[s] = total
        v = new_v.copy()
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


def truncated_policy_iteration(
    env,
    gamma,
    j_truncate=3,
    initial_policy=None,
    threshold=1e-5,
    save_history=False,
):
    """
    切割策略迭代主算法

    Args:
        j_truncate: 策略评估阶段的固定迭代次数
    """
    # 初始化随机策略（均匀分布）
    if initial_policy is None:
        initial_policy = {s: {"still": 1.0} for s in env.get_states()}

    if save_history:
        v_history = []
        p_history = []

    # 初始价值函数为0
    policy_k = initial_policy.copy()
    v_k = {s: 0.0 for s in env.get_states()}
    prev_v_k = None

    while True:
        # 1. 策略评估（固定j_truncate步）
        v_k = truncated_policy_evaluation(
            env, policy_k, gamma, v_k, j_truncate
        )

        # 2. 策略改进（同标准策略迭代）
        policy_next = policy_improvement(env, v_k, gamma)

        # 记录历史
        if save_history:
            v_history.append(v_k.copy())
            p_history.append(policy_next.copy())

        # 判断价值函数是否收敛
        if prev_v_k is not None:
            delta = (
                sum((v_k[s] - prev_v_k[s]) ** 2 for s in env.get_states())
            ) ** 0.5
            if delta < threshold:
                break
        prev_v_k = v_k.copy()
        policy_k = policy_next.copy()

    return_dict = {"v": v_k, "p": policy_k}
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
    return_dict = truncated_policy_iteration(
        env=env,
        gamma=gamma,
        j_truncate=5,  # 控制评估步数（核心参数）
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
                os.path.dirname(__file__),
                "figs/Truncated_Policy_Iteration.gif",
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
            os.path.dirname(__file__),
            "figs/Final_truncated_policy_iteration.png",
        ),
    )
