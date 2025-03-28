import json


def grid_dump_json(
    grid_size,
    target_areas,
    forbidden_areas,
    v_optimal,
    policy_optimal,
    dump_file=None,
):
    if dump_file is None:
        raise ValueError("dump_file should be specified")

    # turn the tuple key into str
    _v_optimal = {str(k): v for k, v in v_optimal.items()}
    _policy_optimal = {str(k): v for k, v in policy_optimal.items()}

    with open(dump_file, "w") as f:
        json.dump(
            {
                "grid_size": grid_size,
                "target_areas": target_areas,
                "forbidden_areas": forbidden_areas,
                "optimal_value": _v_optimal,
                "optimal_policy": _policy_optimal,
            },
            f,
            indent=4,
        )


def grid_load_json(dump_file):
    with open(dump_file, "r") as f:
        data = json.load(f)
    # turn the str key into tuple
    _v_optimal = {eval(k): v for k, v in data["optimal_value"].items()}
    _policy_optimal = {eval(k): v for k, v in data["optimal_policy"].items()}
    target_areas = [tuple(t_a) for t_a in data["target_areas"]]
    forbidden_areas = [tuple(f_a) for f_a in data["forbidden_areas"]]
    return {
        "grid_size": data["grid_size"],
        "target_areas": target_areas,
        "forbidden_areas": forbidden_areas,
        "optimal_value": _v_optimal,
        "optimal_policy": _policy_optimal,
    }


def check_value_and_policy(
    generated_value=None,
    generated_policy=None,
    expected_value=None,
    expected_policy=None,
    check_type: str = "value",
    threshold=1e-5,
):
    """
    Check if the generated value and policy are equal to the expected value and policy within a threshold.

    Args:
        generated_value (dict): generated value function
        generated_policy (dict): generated policy function
        expected_value (dict): expected value function
        expected_policy (dict): expected policy function
        check_type (str): "value" or "policy"
        threshold (float): threshold for checking equality

    Returns:
        None
    """
    allowed_check_types = ["value", "policy"]
    if check_type == "value":
        assert (
            generated_value is not None and expected_value is not None
        ), f"generated_value and expected_value should be specified for {check_type} check."
        # check they have same state
        # assert set(generated_value.keys()) == set(expected_value.keys()), f"generated_value and expected_value should have same state."
        # check every value
        for state in expected_value:
            assert (
                abs(generated_value[state] - expected_value[state]) < threshold
            ), f"generated_value[{state}] and expected_value[{state}] should be equal within {threshold}."
    elif check_type == "policy":
        assert (
            generated_policy is not None and expected_policy is not None
        ), f"generated_policy and expected_policy should be specified for {check_type} check."
        # check every policy
        for state in expected_policy:
            for action in expected_policy[state]:
                assert (
                    abs(
                        generated_policy[state][action]
                        - expected_policy[state][action]
                    )
                    < threshold
                ), f"generated_policy[{state}][{action}] and expected_policy[{state}][{action}] should be equal within {threshold}."
    else:
        raise ValueError(f"check_type should be one of {allowed_check_types}.")
