import torch


def compute_group_normalized_rewards(
    reward_fn,
    rollout_responses,
    repeated_ground_truths,
    group_size,
    advantage_eps,
    normalize_by_std,
    ):

    raw_rewards = []

    for rollout_response, gt_response in zip(rollout_responses, repeated_ground_truths):
        curr_reward = reward_fn(rollout_response, gt_response)['reward']
        raw_rewards.append(curr_reward)
    
    # Compute mean reward for each group
    raw_rewards = torch.tensor(raw_rewards)
    rewards_per_group = raw_rewards.reshape((-1, group_size))
    mean_reward_per_group = torch.mean(rewards_per_group, dim=-1, keepdim=True)

    advantage = rewards_per_group - mean_reward_per_group

    if normalize_by_std:
        std_reward_per_group = torch.std(rewards_per_group, dim=-1, keepdim=True)

        advantage /= (std_reward_per_group + advantage_eps)
    
    advantage = advantage.flatten()

    return advantage, raw_rewards, {}