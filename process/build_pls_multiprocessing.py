import json
import numpy as np
import torch
from multiprocessing import Pool, cpu_count
from functools import partial
from unixcoder import UniXcoder

# 全局配置
PROJECT_NAME = "FFmpeg"
MODEL_NAME = "microsoft/unixcoder-base"
BATCH_SIZE = 32  # 根据内存调整
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 进程初始化函数（每个子进程独立加载模型）
def init_process():
    global process_model
    process_model = UniXcoder(MODEL_NAME).to(DEVICE)
    process_model.eval()


# 批处理函数（每个进程独立处理一批数据）
def process_batch(batch, device_str: str):
    torch.set_num_threads(1)  # 避免MKL占用过多资源
    device = torch.device(device_str)

    embeddings = []
    for func_str in batch:
        try:
            data = json.loads(func_str.strip())
            func = data["func"].replace("\n", "")

            # Tokenization
            tokens_ids = process_model.tokenize(
                [func], max_length=512, mode="<encoder-only>"
            )

            # 模型推理
            with torch.no_grad():
                source_ids = torch.tensor(tokens_ids).to(device)
                _, func_embedding = process_model(source_ids)
                embeddings.append(func_embedding.cpu().numpy())
        except Exception as e:
            print(f"Error processing sample: {str(e)}")
            embeddings.append(np.zeros((1, 768)))  # 填充默认值

    return np.stack(embeddings)


def main():
    for task in ["train", "test", "valid"]:
        input_path = f"data/raw/{PROJECT_NAME}/{task}.jsonl"
        output_path = f"data/dataset/{task}_emb.npy"

        # 读取全部数据
        with open(input_path, "r") as f:
            all_data = f.readlines()

        # 创建进程池（根据CPU核心数调整）
        num_workers = min(8, cpu_count() - 2)
        print(f"Processing {len(all_data)} samples with {num_workers} workers...")

        # 分割批次
        batches = [
            all_data[i : i + BATCH_SIZE] for i in range(0, len(all_data), BATCH_SIZE)
        ]

        # 多进程处理
        with Pool(processes=num_workers, initializer=init_process, initargs=()) as pool:
            processor = partial(process_batch, device_str=DEVICE.type)
            results = pool.imap(processor, batches, chunksize=1)

            # 收集结果
            final_embeddings = []
            for i, batch_result in enumerate(results):
                final_embeddings.append(batch_result)
                if (i + 1) % 10 == 0:
                    print(
                        f"Processed {min((i+1)*BATCH_SIZE, len(all_data))}/{len(all_data)} samples"
                    )

            np.save(output_path, np.vstack(final_embeddings))
            print(f"Saved {output_path} with shape {np.vstack(final_embeddings).shape}")


if __name__ == "__main__":
    # 设置多进程启动方式（Linux建议使用fork，Windows用spawn）
    torch.multiprocessing.set_start_method("spawn" if DEVICE.type == "cuda" else "fork")
    main()
