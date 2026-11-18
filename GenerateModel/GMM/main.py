from gmm import run_learn as gmm
from compare import run_learn as compare
from plot import run_learn as plot

RUN_LEARNS = [gmm, compare, plot]


def main():
    is_running = True
    while is_running:
        print("[main] Select a learn to run (1-3) or 0 to quit:")
        for i, learning in enumerate(RUN_LEARNS, start=1):
            print(f"    {i} - {learning.__doc__}")
        choice = input(">>> ").strip()
        if choice.lower() == '0':
            is_running = False
            continue
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= len(RUN_LEARNS):
                learning = RUN_LEARNS[choice_num - 1]
                print(f"=====  Start   {choice_num} - {learning.__doc__} =====")
                learning()
                print(f"===== Finished {choice_num} - {learning.__doc__} =====")
            else:
                print(f"[main] Error: number between 1 and {len(RUN_LEARNS)}.")
        except ValueError as e:
            print(f"[main] Error: invalid input. {e}")


if __name__ == "__main__":
    main()
