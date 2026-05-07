from engine import parse_common_args, run_auto_lr_case, run_method


if __name__ == "__main__":
    run_method(parse_common_args("autoLR"), run_auto_lr_case)

