from learn1 import run_learn as learn1
from learn2 import run_learn as learn2
from learn3 import run_learn as learn3
from learn4 import run_learn as learn4
from learn5 import run_learn as learn5
from learn6 import run_learn as learn6
from learn7 import run_learn as learn7
from learn8 import run_learn as learn8
from learn9 import run_learn as learn9
from learn10 import run_learn as learn10

LEARN4_MODEL_PATH = r".\Checkpoints\learn4_model.pth"
LEARN4_JUST_PARMS = True
MNIST_DATASET_PATH = r".\Datasets"
RUN_LEARNS = [
    learn1, learn2, learn3, learn4, learn5,
    learn6, learn7, learn8, learn9, learn10
]

def main():
    is_running = True
    while is_running:
        print("[main] Select a learn to run (1-10) or 0 to quit:")
        choice = input(">>> ").strip()
        if choice.lower() == '0':
            is_running = False
            continue
        try:
            choice_num = int(choice)
            if 1 <= choice_num <= 16:
                learning = RUN_LEARNS[choice_num - 1]
                print(f"=====  Start   {choice_num} - {learning.__doc__} =====")    
                learning(
                    learn4_model_path=LEARN4_MODEL_PATH,
                    learn4_just_parms=LEARN4_JUST_PARMS,
                    mnist_dataset_path=MNIST_DATASET_PATH
                )
                print(f"===== Finished {choice_num} - {learning.__doc__} =====")
            else:
                print("[main] Error: number between 1 and 10.")
        except ValueError as e:
            print(f"[main] Error: invalid input. {e}")
            for i, learning in enumerate(RUN_LEARNS, start=1):
                print(f"    {i} - {learning.__doc__}")

if __name__ == "__main__":
    main()