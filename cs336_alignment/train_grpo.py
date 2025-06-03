import torch
from vllm import LLM, SamplingParams
import json

from cs336_alignment.math_baseline import evaluate_vllm


def get_starter_params():
    return {
        'n_grpo_steps': 200,
        'learning_rate': 1e-5,
        'advantage_eps': 1e-6,
        'rollout_batch_size': 256,
        'group_size': 8,
        'sampling_temperature': 1.0,
        'sampling_min_tokens': 4,
        'sampling_max_tokens': 1024,
        'epochs_per_rollout_batch': 1,
        'train_batch_size': 256,
        'gradient_accumulation_steps': 128,
        'gpu_memory_utilization': 0.85,
        'loss_type': 'reinforce_with_baseline',
        'use_std_normalization': True,
        # 'optimizer': torch.optim.AdamW(
        #     policy.parameters(),
        #     lr=learning_rate,
        #     weight_decay=0.0,
        #     betas=(0.9, 0.95),
        # ),
    }

def init_llm(params):
    sampling_params = SamplingParams(
        temperature=params['sampling_temperature'],
        top_p=1.0,
        min_tokens=params['sampling_min_tokens'],
        max_tokens=params['sampling_max_tokens'],
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model='/data/a5-alignment/models/Qwen2.5-Math-1.5B')

    return llm, sampling_params

def get_validation_data():
    with open('/data/a5-alignment/MATH/validation.jsonl', 'r') as f:
        prompt_data = [json.loads(json_line) for json_line in f]
    
    prompts = []
    answers = []

    for p in prompt_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        prompts.append(prompt_string)

        answers.append(p['answer'])

    return prompts, answers

if __name__ == '__main__':
    params = get_starter_params()

    llm, sampling_params = init_llm(params)
    prompts, answers = get_validation_data()