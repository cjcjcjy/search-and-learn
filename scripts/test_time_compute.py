# !/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import sys

from datasets import Dataset

sys.path.append("/home/jcyang/search/src")
import torch
from vllm import LLM

from sal.config import Config
from sal.models.reward_models import load_prm
from sal.search import beam_search, best_of_n, dvts
from sal.utils.data import get_dataset, save_dataset
from sal.utils.parser import H4ArgumentParser
from sal.utils.score import score

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


APPROACHES = {
    "beam_search": beam_search,
    "dvts": dvts,
    "best_of_n": best_of_n,
}


def main():
    parser = H4ArgumentParser(Config)
    config = parser.parse()
    print(config)
    approach_fn = APPROACHES[config.approach]

    num_gpus = torch.cuda.device_count()
    """llm = LLM(
    model=config.model_path,
    gpu_memory_utilization=config.gpu_memory_utilization,
    enable_prefix_caching=True,
    seed=config.seed,
    tensor_parallel_size=num_gpus,
    )
    

    dataset = get_dataset(config)
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        # fn_kwargs={"config": config, "llm": llm, "prm": prm},
        fn_kwargs={"config": config, "llm": llm},
        desc="Running search",
        load_from_cache_file=False,
    ) """


    # 读取 JSON Lines 文件并转换为 Python 列表
    import json
    dataa = []
    co = 0
    with open(
        f"/home/jcyang/search/data/{config.dataset_file}/completions.jsonl", "r", encoding="utf-8"
    ) as f:
        for line in f:
            if co < 800:
                co+=1
                continue
            # if co == 800:
            #     break
            co += 1
            data = json.loads(line)
            data = {
                key: value for key, value in data.items() if not key.startswith("pred_")
            }
            dataa.append(data)
    # 将字典列表转换为 Dataset 对象
    dataset = Dataset.from_list(dataa)
    prm = load_prm(config)
    dataset = dataset.map(
        approach_fn,
        batched=True,
        batch_size=config.search_batch_size,
        fn_kwargs={"config": config, "prm": prm},
        desc="Running search",
        load_from_cache_file=False,
    )

    dataset = score(dataset, config)

    # save_dataset(dataset, config)
    dataset.to_json(
        f"data/{config.dataset_file}/{config.prm_file}/best_of_n_completions1.jsonl",
        lines=True,
    )


if __name__ == "__main__":
    main()
