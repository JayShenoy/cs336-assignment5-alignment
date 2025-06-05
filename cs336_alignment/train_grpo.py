import torch
from vllm import LLM, SamplingParams
import json
import random

from cs336_alignment.math_baseline import evaluate_vllm
from cs336_alignment.vllm_helper import *
from cs336_alignment.sft_helper import tokenize_prompt_and_output, get_response_log_probs


with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()

def get_starter_params(policy, debug=False):
    params = {
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
    }

    if debug:
        params['n_grpo_steps'] = 1
        params['rollout_batch_size'] = 2
        params['train_batch_size'] = 2
        params['gradient_accumulation_steps'] = 1
        params['group_size'] = 2

    if not debug:
        params['optimizer'] = torch.optim.AdamW(
            policy.parameters(),
            lr=1e-5,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
    
    return params

def init_sampling_params(params):
    sampling_params = SamplingParams(
        temperature=params['sampling_temperature'],
        top_p=1.0,
        min_tokens=params['sampling_min_tokens'],
        max_tokens=params['sampling_max_tokens'],
        logprobs=0,
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    return sampling_params

def get_jsonl_data(fpath):
    with open(fpath, 'r') as f:
        prompt_data = [json.loads(json_line) for json_line in f]
    
    dataset = []

    for p in prompt_data:
        prompt_string = R1_ZERO_PROMPT.format(
            question=p['problem']
        )
        answer_string = p['answer']

        dataset.append({
            'prompt': prompt_string,
            'answer': p['answer'],
        })

    return dataset

def get_training_data():
    return get_jsonl_data('/data/a5-alignment/MATH/train.jsonl')

def sample_dataset(dataset, num_samples):
    sampled_data = random.sample(dataset, num_samples)

    ret = {
        'prompts': [],
        'answers': [],
    }

    for d in sampled_data:
        ret['prompts'].append(d['prompt'])
        ret['answers'].append(d['answer'])

    return ret

def train_policy(policy, tokenizer, vllm, sampling_params, training_data, training_params):
    assert training_params['train_batch_size'] % training_params['gradient_accumulation_steps'] == 0, (
        "train_batch_size must be divisible by gradient_accumulation_steps"
    )
    micro_train_batch_size = training_params['train_batch_size'] // training_params['gradient_accumulation_steps']

    assert training_params['rollout_batch_size'] % training_params['group_size'] == 0, (
        "rollout_batch_size must be divisible by group_size"
    )
    n_prompts_per_rollout_batch = training_params['rollout_batch_size'] // training_params['group_size']

    assert training_params['train_batch_size'] >= training_params['group_size'], (
        "train_batch_size must be greater than or equal to group_size"
    )
    n_microbatches_per_rollout_batch = training_params['rollout_batch_size'] // micro_train_batch_size

    device = policy.device

    for _ in range(training_params['n_grpo_steps']):
        sampled_training_data = sample_dataset(training_data, micro_train_batch_size)
        prompts = sampled_training_data['prompts']
        answers = sampled_training_data['answers']

        data_tokenized = tokenize_prompt_and_output(prompts, answers, tokenizer)
        input_ids = data_tokenized['input_ids'].to(device)
        labels = data_tokenized['labels'].to(device)
        policy_log_probs = get_response_log_probs(
            policy,
            input_ids,
            labels,
            return_token_entropy=False
        )
        policy_log_probs = policy_log_probs['log_probs']

        print(policy_log_probs.shape)
        print(policy_log_probs)

        # outputs = vllm.generate(prompts, sampling_params)

        # old_log_probs = []

        # for o in outputs:
        #     curr_old_log_probs = [list(d.values())[0] for d in o.outputs[0].logprobs]
        #     curr_old_log_probs = [p.logprob for p in curr_old_log_probs]
        #     old_log_probs.append(curr_old_log_probs)
        
        # old_log_probs = torch.tensor(old_log_probs).to(policy.device)
        # print(old_log_probs.shape)
        # print(old_log_probs)

if __name__ == '__main__':
    DEBUG = True

    policy, tokenizer = init_policy(debug=DEBUG)
    params = get_starter_params(policy, debug=DEBUG)
    vllm = init_vllm(
        '/data/a5-alignment/models/Qwen2.5-Math-1.5B',
        0,
        42,
        params['gpu_memory_utilization']
    )
    sampling_params = init_sampling_params(params)
    training_data = get_training_data()

    policy_trained = train_policy(policy, tokenizer, vllm, sampling_params, training_data, params)