import torch
from vllm import LLM, SamplingParams
import json
import random

from cs336_alignment.math_baseline import evaluate_vllm
from cs336_alignment.vllm_helper import *
from cs336_alignment.sft_helper import tokenize_prompt_and_output, get_response_log_probs
from cs336_alignment.drgrpo_grader import r1_zero_reward_fn
from cs336_alignment.grpo import *


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
        params['rollout_batch_size'] = 1
        params['train_batch_size'] = 1
        params['gradient_accumulation_steps'] = 1
        params['group_size'] = 1

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

        load_policy_into_vllm_instance(policy, vllm)

        vllm_rollouts = vllm.generate(prompts, sampling_params)

        rollout_input_text = []
        rollout_response_text = []

        for rollout in vllm_rollouts:
            for r in rollout.outputs:
                rollout_input_text.append(rollout.prompt)
                rollout_response_text.append(r.text)
        
        rollout_data_tokenized = tokenize_prompt_and_output(
            rollout_input_text,
            rollout_response_text,
            tokenizer
        )
        rollout_input_ids = rollout_data_tokenized['input_ids'].to(device)
        rollout_labels = rollout_data_tokenized['labels'].to(device)
        rollout_response_mask = rollout_data_tokenized['response_mask'].to(device)
        
        advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
            r1_zero_reward_fn,
            rollout_response_text,
            answers,
            training_params['group_size'],
            training_params['advantage_eps'],
            training_params['use_std_normalization'],
        )

        advantages = advantages.to(device)
        raw_rewards = raw_rewards.to(device)

        policy_log_probs_dict = get_response_log_probs(
            policy,
            rollout_input_ids,
            rollout_labels,
            return_token_entropy=True
        )
        policy_log_probs = policy_log_probs_dict['log_probs']
        policy_token_entropy = policy_log_probs_dict['token_entropy']

        old_log_probs = policy_log_probs # change this when doing off-policy updates

        grpo_microbatch_train_step(
            policy_log_probs,
            rollout_response_mask,
            training_params['gradient_accumulation_steps'],
            training_params['loss_type'],
            raw_rewards,
            advantages,
            old_log_probs,
            1.0,
        )

if __name__ == '__main__':
    DEBUG = True

    policy, tokenizer = init_policy(debug=DEBUG)
    params = get_starter_params(policy, debug=DEBUG)
    vllm = init_vllm(
        '/data/a5-alignment/models/Qwen2.5-Math-1.5B',
        'cuda:1',
        42,
        params['gpu_memory_utilization'],
        debug=DEBUG
    )
    sampling_params = init_sampling_params(params)
    training_data = get_training_data()

    policy_trained = train_policy(policy, tokenizer, vllm, sampling_params, training_data, params)