import json
from typing import List

import numpy as np
import torch
from time import time
from unixcoder import UniXcoder
from multiprocessing import Pool, cpu_count

"""
Preprocess tokenize. Store in ./data/tokens_ids/{project_name}_{task}.npy
"""

# 全局配置
PROJECT_NAME = "FFmpeg"
MODEL_NAME = "microsoft/unixcoder-base"
DEVICE = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
OUTPUT_DIR = "data/tokens_ids"

model = UniXcoder(MODEL_NAME).to(DEVICE)


def main(model: UniXcoder, task: str) -> List[np.ndarray]:
    """
    task: str, "train", "test", "valid"
    """
    array_list = []
    with open(f"data/raw/{PROJECT_NAME}/{task}.jsonl", "r") as f:
        func_list = f.readlines()

    for i in range(0, len(func_list)):
        func = json.loads(func_list[i].replace("\n", ""))["func"].replace("\n", "")
        tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
        array_list.append(tokens_ids)

    return array_list


def init_process():
    global process_model
    process_model = UniXcoder(MODEL_NAME).to(DEVICE)
    process_model.eval()


def process_element(raw_data: str):
    data = json.loads(raw_data.replace("\n", ""))["func"].replace("\n", "")
    tokens_ids = process_model.tokenize([data], max_length=512, mode="<encoder-only>")
    return tokens_ids


def main_multiprocessing(_, __):
    """
    task: str, "train", "test", "valid"
    """
    input_path = f"data/raw/{PROJECT_NAME}/{task}.jsonl"

    # 读取全部数据
    with open(input_path, "r") as f:
        raw_data = f.readlines()

    # 创建进程池（根据CPU核心数调整）
    num_workers = min(8, cpu_count() - 2)
    print(f"Processing {len(raw_data)} samples with {num_workers} workers...")

    # 不用分割批次，
    torch.multiprocessing.set_start_method("spawn")
    with Pool(num_workers, initializer=init_process) as pool:
        print("Start processing...")
        embeddings = pool.map(
            process_element,
            raw_data,
        )

    return embeddings


if __name__ == "__main__":
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for task in ["train", "test", "valid"]:
        array_list = main_multiprocessing(model, task)
        np.save(f"{OUTPUT_DIR}/{PROJECT_NAME}_{task}.npy", np.stack(array_list))
        print(f"Saved {PROJECT_NAME}_{task}.npy")
