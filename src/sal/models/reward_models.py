#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

import logging

from peft import PeftModel

logging.basicConfig(level=logging.INFO, filename="test.log", filemode="a")

logger = logging.getLogger("test")
logger.setLevel(level=logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s"
)

file_handler = logging.FileHandler("test.log")
file_handler.setLevel(level=logging.INFO)
file_handler.setFormatter(formatter)


logger.addHandler(file_handler)


import sys
from itertools import accumulate

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from sal.config import Config

sys.path.append(r"/home/jcyang/skywork-o1-prm-inference")
from model_utils.io_utils import (
    derive_step_rewards,
    prepare_batch_input_for_model,
    prepare_input,
)
from model_utils.prm_model import PRM_MODEL

CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902


def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores


class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError


class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "/data/disk2/jjxiao/peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(output), (
                f"{len(output_score)} != {len(output)}"
            )

        return output_scores


class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            f"/data/disk2/jjxiao/RLHFlow/{self.search_config.prm_path}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"/data/disk2/jjxiao/RLHFlow/{self.search_config.prm_path}",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=2,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to("cuda:0")
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to("cpu")
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)
        torch.cuda.empty_cache()

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores





import numpy as np
import torch
from peft import PeftModel
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class Math_psa(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        self.good_token = '+'
        self.bad_token = '-'

        self.tokenizer = AutoTokenizer.from_pretrained("/data/disk2/jjxiao/Qwen2.5-Math-7B-Instruct", add_eos_token=False, padding_side='left')
        self.tokenizer.pad_token_id = 151655 # "<|image_pad|>"

        self.candidate_tokens = self.tokenizer.encode(f" {self.good_token} {self.bad_token}") # [488, 481]

        self.model = AutoModelForCausalLM.from_pretrained("/data/disk2/jjxiao/Qwen2.5-Math-7B-Instruct", 
                                                          device_map="auto", 
                                                          torch_dtype=torch.bfloat16,
                                                          **model_kwargs,
                                                        #   attn_implementation="flash_attention_2",
                                                          ).eval()
        # adapter_config = PeftConfig.from_pretrained(cp_path)
        self.model = PeftModel.from_pretrained(self.model, "/data/disk2/jjxiao/checkpoint-2127/")

        return self.model, self.tokenizer
    

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=4,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": " \n\n\n\n\n", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": " \n\n\n\n\n", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            # ).to(self.model.device)
            ).to("cuda:0")
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to("cpu")
            # ).to("cuda:1")
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        (inputs2_batch[i, 1:]).to("cpu",torch.int64) == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)
        torch.cuda.empty_cache()

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


class llemma_7b_prm_prm800k(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_name = f"/data/disk2/jjxiao/ScalableMath/{self.search_config.prm_path}-level-1to3-hf/"
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")

        tokenizer = AutoTokenizer.from_pretrained("EleutherAI/llemma_7b")

        return model, tokenizer
    
    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = False,
        batch_size=4,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:

                qa = question + ' ' + ans
                qa = qa.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('\n\n##', '\n##').replace('\n##', '\n\n##')
                qa = qa + '\n\n'
                qa = qa.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
                scoring_tokens = self.tokenizer.encode("\n\n", add_special_tokens=False)[1:]
                # eos_token = tokenizer.eos_token_id
                input_ids = self.tokenizer.encode(qa)
                candidate_positions = []

                for start_idx in range(len(input_ids)):
                    # if tuple(input_ids[start_idx:start_idx+len(begin_solution_tokens)]) == tuple(begin_solution_tokens):
                    #     begin_solution_flag = True

                    if tuple(input_ids[start_idx:start_idx+len(scoring_tokens)]) == tuple(scoring_tokens):
                        candidate_positions.append(start_idx)

                    # if input_ids[start_idx] == eos_token:
                    #     candidate_positions.append(start_idx)
                    #     break

                # maybe delete the first and the second to last candidate_positions
                # because they are "\n\n" after "# Solution" and after "# Answer"
                # del candidate_positions[0]
                # del candidate_positions[-2]

                input_tensor = torch.tensor([input_ids]).to(self.model.device)
                candidate_positions = torch.tensor(candidate_positions)

                with torch.no_grad():
                    logits = self.model(input_tensor).logits
                    scores =logits.mean(dim=-1)
                    step_scores = scores[0][candidate_positions]
                    step_probs = torch.sigmoid(step_scores)

                step_probs = step_probs.detach().to(torch.float).cpu().numpy()
                print(step_probs)

                all_step_scores.append(step_probs)
            all_scores.append(all_step_scores)
        return all_scores


class PURE_PRM_7B(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        
        from transformers import AutoModelForTokenClassification, AutoTokenizer
        model_name = "/data/disk2/jjxiao/PURE-PRM-7B/"

        tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            trust_remote_code=True,
        )
        model = AutoModelForTokenClassification.from_pretrained(
            model_name, 
            device_map="auto", 
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        ).eval()

        return model, tokenizer

    def make_step_rewards(self, logits, token_masks):
                all_scores_res = []
                for sample, token_mask in zip(logits, token_masks):
                    # sample: (seq_len, num_labels)
                    probs = sample[token_mask].softmax(dim=-1)  # (num_steps, 2)
                    process_reward = probs[:, 1] - probs[:, 0]  # (num_steps,)
                    # weighted sum to approx. min, highly recommend when BoN eval and Fine-tuning LLM
                    # weight = torch.softmax(
                    #     -process_reward / 0.1, 
                    #     dim=-1,
                    # )
                    # process_reward = weight * process_reward
                    all_scores_res.append(process_reward.cpu().tolist())
                return all_scores_res

    def score(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            print(question)
            all_step_scores = []
            for ans in answers:

                qa = ans
                qa = qa.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('\n\n##', '\n##').replace('\n##', '\n\n##')
                
                
                steps = qa.split("\n\n")

                step_separator = "\n\n"
                step_separator_token = self.tokenizer(
                    step_separator, 
                    add_special_tokens=False, 
                    return_tensors='pt',
                )['input_ids'].to("cpu")
                input_ids = self.tokenizer(
                    question, 
                    add_special_tokens=False, 
                    return_tensors='pt',
                )['input_ids'].to("cpu")

                score_ids = []
                for step in steps:
                    step_ids = self.tokenizer(
                        step, 
                        add_special_tokens=False, 
                        return_tensors='pt',
                    )['input_ids'].to("cpu")
                    input_ids = torch.cat(
                        [input_ids, step_ids, step_separator_token], 
                        dim=-1,
                    ).to("cpu")
                    score_ids.append(input_ids.size(-1) - 1)

                input_ids = input_ids.to(self.model.device)
                input_ids = input_ids.int()
                token_masks = torch.zeros_like(input_ids, dtype=torch.bool)
                token_masks[0, score_ids] = True
                assert torch.all(input_ids[token_masks].to("cpu") == step_separator_token)

                logits = self.model(input_ids).logits
                step_reward = self.make_step_rewards(logits, token_masks)
                print(step_reward)  # [[0.796875, 0.185546875, -0.0625, 0.078125]]


                all_step_scores.append(step_reward[0])
                del step_reward, input_ids
                torch.cuda.empty_cache()
            all_scores.append(all_step_scores)
        return all_scores
    
class Qwen(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            f"/data/disk2/jjxiao/Qwen/{self.search_config.prm_path}"
        )
        model = AutoModelForCausalLM.from_pretrained(
            f"/data/disk2/jjxiao/Qwen/{self.search_config.prm_path}",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=4,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "<extra_0>", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "<extra_0>", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            # ).to(self.model.device)
            ).to("cuda:0")
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to("cpu")
            # ).to("cuda:1")
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        (inputs2_batch[i, 1:]).to("cpu",torch.int64) == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)
            torch.cuda.empty_cache()

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores

class Skywork_o1_Open_PRM(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        prm_model_path = f"/data/disk2/jjxiao/Skywork/{self.search_config.prm_path}"

        tokenizer = AutoTokenizer.from_pretrained(
            prm_model_path, trust_remote_code=True
        )
        model = PRM_MODEL.from_pretrained(prm_model_path, device_map="auto").eval()

        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ):
        reshaped_output_scores = []
        import torch
        from torch.amp import autocast  # 使用新的 autocast API

        for question, answers in zip(questions, outputs, strict=True):
            torch.cuda.empty_cache()  # 清理 GPU 缓存
            scores = []
            i = 0
            for answer in answers:
                processed_data = [
                    prepare_input(
                        question, answer, tokenizer=self.tokenizer, step_token="\n\n"
                    )
                ]
                input_ids, steps, reward_flags = zip(*processed_data)

                input_ids, attention_mask, reward_flags = prepare_batch_input_for_model(
                    input_ids, reward_flags, self.tokenizer.pad_token_id
                )
                input_ids = input_ids.to("cuda:0")
                attention_mask = attention_mask.to("cuda:0")

                with autocast(
                    device_type="cuda", dtype=torch.float16
                ):  # 使用新的 autocast API
                    with torch.no_grad():  # 如果不需梯度，使用 no_grad
                        _, _, rewards = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_probs=True,
                        )

                step_rewards = derive_step_rewards(rewards, reward_flags)
                scores.append(step_rewards[0])
            reshaped_output_scores.append(scores)
        return reshaped_output_scores


class Llama_PRM800K(PRM):
    # skywork
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_name = "/data/disk2/jjxiao/UW-Madison-Lee-Lab/Llama-PRM800K"

        tokenizer = AutoTokenizer.from_pretrained(model_name) 
        tokenizer.pad_token = tokenizer.eos_token  
        tokenizer.padding_side = 'left' 
        tokenizer.truncation_side = 'left'
        model = AutoModelForCausalLM.from_pretrained(model_name, device_map = "auto")

        return model, tokenizer

    def score(self, questions: list[str], outputs: list[list[str]]):
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:

                qa = ans
                qa = qa.replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n').replace('\n\n##', '\n##').replace('\n##', '\n\n##')
                solution = qa.split("\n\n")
                input_text = question + ' \n\n' + ' \n\n\n\n'.join(solution) + ' \n\n\n\n' # solution steps are separated by ' \n\n\n\n'
                input_id = torch.tensor([self.tokenizer.encode(input_text)]).to(self.model.device)
                candidate_tokens = [12, 10]
                with torch.no_grad():
                    logits = self.model(input_id).logits[:,:,candidate_tokens]
                    scores = logits.softmax(dim=-1)[:,:,1] 
                    step_probs = scores[input_id == 23535]                    

                step_probs = step_probs.detach().to(torch.float).cpu().numpy()
                if len(step_probs) == 0:
                    step_probs =[0.0]
                print(step_probs)
                all_step_scores.append(step_probs)
                torch.cuda.empty_cache()
            all_scores.append(all_step_scores)
        return all_scores

def load_prm(config: Config) -> PRM:
    if config.prm_path == "Llama_PRM800K":
        return Llama_PRM800K(config)
    
    if config.prm_path in ["Llama3.1-8B-PRM-Deepseek-Data", "Llama3.1-8B-PRM-Mistral-Data"]:
        return RLHFFlow(config)
    
    if config.prm_path in ["llemma_7b_prm_prm800k", "llemma_7b_oprm_prm800k", "llemma_7b_prm_metamath"]:
        return llemma_7b_prm_prm800k(config)
    
    if config.prm_path == "math-psa":
        return Math_psa(config)
    
    if config.prm_path == "math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "PURE_PRM_7B":
        return PURE_PRM_7B(config) 

    if config.prm_path in ["Qwen2.5-Math-7B-PRM800K", "Qwen2.5-Math-PRM-7B"]:
        return Qwen(config)

    if config.prm_path in ["Skywork-o1-Open-PRM-Qwen-2.5-7B", "Skywork-o1-Open-PRM-Qwen-2.5-1.5B"]:
        return Skywork_o1_Open_PRM(config)
    


    raise NotImplementedError(f"PRM {config.prm_path} not implemented")