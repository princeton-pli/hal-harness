def log_start(msg: str) -> None:
    print(f"====={msg}=====")


def log(msg: str = "") -> None:
    print(msg)


def log_end(msg: str | None = None) -> None:
    if msg:
        print(msg)
    print("===============\n")
