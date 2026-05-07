from engine import parse_common_args, run_cd_case, run_method


if __name__ == "__main__":
    run_method(parse_common_args("constraint_decoding"), run_cd_case)

