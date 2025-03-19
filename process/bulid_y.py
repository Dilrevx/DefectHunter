import os
import json
import numpy as np

num_test = []
num_train = []
num_val = []
project_name = "FFmpeg"
task = "test"


def main(project_name: str, task: str):
    global num_test
    num_test = []
    with open(f"data/raw/{project_name}/{task}.jsonl", "r", encoding="utf8") as f:
        c = f.readlines()
        for i in c:
            text = json.loads(i)["target"]
            num_test.append(int(text))
    arr_y = np.array(num_test)
    print(arr_y.shape)
    np.save(f"data/dataset/{task}_y.npy", arr_y)


if __name__ == "__main__":
    for task in ["train", "test", "valid"]:
        if os.path.exists(f"data/dataset/{task}_y.npy"):
            continue
        main(project_name, task)
