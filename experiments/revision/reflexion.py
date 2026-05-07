from engine import parse_common_args, run_method, run_reflexion_case


if __name__ == "__main__":
    run_method(parse_common_args("reflexion"), run_reflexion_case)

