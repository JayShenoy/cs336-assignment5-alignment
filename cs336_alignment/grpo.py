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

    metadata = {
        'mean': torch.mean(raw_rewards),
        'std': torch.std(raw_rewards),
        'max': torch.max(raw_rewards),
        'min': torch.min(raw_rewards),
    }

    return advantage, raw_rewards, metadata

def compute_naive_policy_gradient_loss(
    raw_rewards_or_advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    ) -> torch.Tensor:

    return -raw_rewards_or_advantages * policy_log_probs

def compute_grpo_clip_loss(
    advantages: torch.Tensor,
    policy_log_probs: torch.Tensor,
    old_log_probs: torch.Tensor,
    cliprange: float,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:

    pi_ratio = torch.exp(policy_log_probs - old_log_probs)
    unclipped_term = advantages * pi_ratio

    clipped_term = torch.clip(pi_ratio, min=1 - cliprange, max=1 + cliprange)
    clipped_term *= advantages

    loss = -torch.minimum(unclipped_term, clipped_term)

    return loss, {}