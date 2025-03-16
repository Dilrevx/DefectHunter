import json

import numpy as np
import torch
from time import time
from unixcoder import UniXcoder

# model = UniXcoder("unx")
model = UniXcoder("microsoft/unixcoder-base")
device = torch.device("cuda")
model.to(device)

for task in ["train", "test", "valid"]:
    array_list = []
    project_name = "FFmpeg"
    t0 = time()

    with open(f"data/raw/{project_name}/{task}.jsonl", "r") as f:
        func_list = f.readlines()

    print(f"Loaded {len(func_list)} functions in {time() - t0:.2f} seconds.")

    for i in range(0, len(func_list)):
        func = json.loads(func_list[i].replace("\n", ""))["func"].replace("\n", "")
        tokens_ids = model.tokenize([func], max_length=512, mode="<encoder-only>")
        source_ids = torch.tensor(tokens_ids).to(device)
        print(f"tokenized & moved to device in {time() - t0:.2f} seconds.")
        tokens_embeddings, max_func_embedding = model(source_ids)
        print(f"encoded in {time() - t0:.2f} seconds.")
        max_func_embedding = max_func_embedding.cpu().detach().numpy()
        print(max_func_embedding.shape)
        print(
            f"finished {i + 1}/{len(func_list)} functions in {time() - t0:.2f} seconds."
        )
        array_list.append(max_func_embedding)
        print(len(array_list))

    np.save(f"data/dataset/{task}_emb.npy", np.stack(array_list))
