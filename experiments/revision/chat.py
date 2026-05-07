from engine import parse_common_args, run_actor_case, run_method


def runner(ctx, case):
    return run_actor_case(ctx, case, mode="chat")


if __name__ == "__main__":
    run_method(parse_common_args("chat"), runner)

