import torch
from vllm import LLM, SamplingParams
import json
import random
import wandb
import sys

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
        'gpu_memory_utilization': 0.2,
        'loss_type': 'reinforce_with_baseline',
        'use_std_normalization': True,
        'eval_sample_size': 1024,
        'eval_log_frequency': 5,
    }

    params['optimizer'] = torch.optim.AdamW(
        policy.parameters(),
        lr=params['learning_rate'],
        weight_decay=0.0,
        betas=(0.9, 0.95),
    )

    if debug:
        params['n_grpo_steps'] = 5
        params['rollout_batch_size'] = 1
        params['train_batch_size'] = 1
        params['gradient_accumulation_steps'] = 1
        params['group_size'] = 1
        params['eval_sample_size'] = 16
        params['eval_log_frequency'] = 1
    
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

def get_eval_data():
    return get_jsonl_data('/data/a5-alignment/MATH/validation.jsonl')

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

def duplicate_data(arr, group_size):
    '''
    Ex: duplicate_data([1, 2, 3], 2) => [1, 1, 2, 2, 3, 3]
    '''

    return [x for x in arr for _ in range(group_size)]

def train_policy(policy, tokenizer, vllm, sampling_params, training_data, training_params,
                experiment_name, eval_data):
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

    wandb_run = wandb.init(
        entity="jayshenoy-stanford-university",
        project="cs336_alignment",
        config=training_params,
        name=experiment_name,
    )

    # Setup wandb metrics
    wandb.define_metric("train_step")
    wandb.define_metric("eval_step")
    wandb.define_metric("train/*", step_metric="train_step")
    wandb.define_metric("eval/*", step_metric="eval_step")

    train_step = 0
    eval_step = 0

    for grpo_step_idx in range(training_params['n_grpo_steps']):
        load_policy_into_vllm_instance(policy, vllm)

        if grpo_step_idx % training_params['eval_log_frequency'] == 0:
            sampled_eval_data = sample_dataset(eval_data, training_params['eval_sample_size'])
            prompts_batch = sampled_eval_data['prompts']
            answers_batch = sampled_eval_data['answers']

            vllm_rollouts = vllm.generate(prompts_batch, sampling_params)

            rollout_input_text = []
            rollout_response_text = []

            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)
            
            _, _, reward_metadata = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_response_text,
                answers_batch,
                1,
                training_params['advantage_eps'],
                training_params['use_std_normalization'],
            )

            # Print a randomly sampled eval response
            eval_rand_idx = random.randrange(training_params['eval_sample_size'])
            print('Eval step:', eval_step)
            print('Prompt:')
            print(rollout_input_text[eval_rand_idx])
            print('Correct Answer:')
            print(answers_batch[eval_rand_idx])
            print('LLM Response:')
            print(rollout_response_text[eval_rand_idx])

            wandb_run.log({
                'eval_step': eval_step,
                'eval/reward_mean': reward_metadata['mean'],
            })

            eval_step += 1

        # One policy gradient step per train_batch_size of data
        for rollout_batch_idx in range(0, training_params['train_batch_size'], training_params['rollout_batch_size']):
            # Sample a batch of data, then select microbatches later
            sampled_training_data = sample_dataset(training_data, n_prompts_per_rollout_batch)
            prompts_batch = sampled_training_data['prompts']
            answers_batch = sampled_training_data['answers']

            prompts_batch = duplicate_data(prompts_batch, training_params['group_size'])
            answers_batch = duplicate_data(answers_batch, training_params['group_size'])

            vllm_rollouts = vllm.generate(prompts_batch, sampling_params)

            rollout_input_text = []
            rollout_response_text = []

            for rollout in vllm_rollouts:
                for r in rollout.outputs:
                    rollout_input_text.append(rollout.prompt)
                    rollout_response_text.append(r.text)
            
            advantages, raw_rewards, reward_metadata = compute_group_normalized_rewards(
                r1_zero_reward_fn,
                rollout_response_text,
                answers_batch,
                training_params['group_size'],
                training_params['advantage_eps'],
                training_params['use_std_normalization'],
            )

            wandb_run.log({
                'train_step': train_step,
                'train/reward_mean': reward_metadata['mean']
            })

            rollout_data_tokenized = tokenize_prompt_and_output(
                rollout_input_text,
                rollout_response_text,
                tokenizer
            )

            for _ in range(training_params['epochs_per_rollout_batch']):
                training_params['optimizer'].zero_grad()

                rollout_batch_loss = 0

                for microbatch_idx in range(n_microbatches_per_rollout_batch):
                    microbatch_slice = slice(
                        microbatch_idx * micro_train_batch_size,
                        (microbatch_idx + 1) * micro_train_batch_size
                    )

                    microbatch_input_ids = rollout_data_tokenized['input_ids'][microbatch_slice].to(device)
                    microbatch_labels = rollout_data_tokenized['labels'][microbatch_slice].to(device)
                    microbatch_response_mask = rollout_data_tokenized['response_mask'][microbatch_slice].to(device)

                    advantages_microbatch = advantages[microbatch_slice].to(device)
                    raw_rewards_microbatch = raw_rewards[microbatch_slice].to(device)

                    policy_log_probs_dict = get_response_log_probs(
                        policy,
                        microbatch_input_ids,
                        microbatch_labels,
                        return_token_entropy=True
                    )
                    policy_log_probs = policy_log_probs_dict['log_probs']
                    policy_token_entropy = policy_log_probs_dict['token_entropy']

                    old_log_probs = policy_log_probs # change this when doing off-policy updates

                    advantages_microbatch = advantages_microbatch.unsqueeze(-1)

                    loss, loss_metadata = grpo_microbatch_train_step(
                        policy_log_probs,
                        microbatch_response_mask,
                        training_params['gradient_accumulation_steps'],
                        training_params['loss_type'],
                        raw_rewards_microbatch,
                        advantages_microbatch,
                        old_log_probs,
                        1.0,
                    )

                    rollout_batch_loss += loss.item()
        
                training_params['optimizer'].step()

                rollout_batch_loss /= n_microbatches_per_rollout_batch
                wandb_run.log({
                    'train_step': train_step,
                    'train/loss': loss
                })
                train_step += 1
    
    wandb_run.finish()
    
    print('Training complete')

if __name__ == '__main__':
    DEBUG = int(sys.argv[1])

    print('Debug mode:', DEBUG)

    policy, tokenizer = init_policy(debug=DEBUG)
    params = get_starter_params(policy, debug=DEBUG)
    vllm = init_vllm(
        '/data/a5-alignment/models/Qwen2.5-Math-1.5B',
        'cuda:0',
        42,
        params['gpu_memory_utilization'],
        debug=DEBUG
    )
    sampling_params = init_sampling_params(params)
    training_data = get_training_data()
    eval_data = get_eval_data()

    if DEBUG:
        experiment_name = 'debug_5_grpo_steps'
    else:
        experiment_name = 'grpo_baseline'
    policy_trained = train_policy(policy, tokenizer, vllm, sampling_params,
                                training_data, params, experiment_name, eval_data)