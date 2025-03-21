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
