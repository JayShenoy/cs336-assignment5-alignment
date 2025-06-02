import torch
import torch.nn.functional as F


def tokenize_prompt_and_output(prompt_strs, output_strs, tokenizer):
    prompt_tokens = tokenizer(prompt_strs)['input_ids']
    output_tokens = tokenizer(output_strs)['input_ids']

    batch_sz = len(prompt_tokens)

    prompt_and_output_lens = [len(p) + len(o) for p, o in zip(prompt_tokens, output_tokens)]
    padded_len = max(prompt_and_output_lens)

    input_ids = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    labels = torch.empty((batch_sz, padded_len - 1), dtype=torch.long)
    response_mask = torch.zeros((batch_sz, padded_len - 1), dtype=torch.bool)

    for i, (p_toks, o_toks) in enumerate(zip(prompt_tokens, output_tokens)):
        p_o_concat = torch.tensor(p_toks + o_toks)
        concat_len = len(p_o_concat)
        p_o_concat_padded = F.pad(p_o_concat, (0, padded_len - concat_len), 'constant', 151643)

        input_ids[i] = p_o_concat_padded[:-1]
        labels[i] = p_o_concat_padded[1:]

        o_start = len(p_toks) - 1
        o_end = concat_len - 1
        response_mask[i, o_start:o_end] = True
    
    return {
        'input_ids': input_ids,
        'labels': labels,
        'response_mask': response_mask,
    }

def compute_entropy(logits: torch.Tensor) -> torch.Tensor:
    p_numerator = torch.exp(logits)
    p_denom = torch.sum(p_numerator, dim=-1, keepdim=True)

    log_prob = logits - torch.logsumexp(logits, dim=-1, keepdim=True)

    summands = (p_numerator / p_denom) * log_prob

    return -torch.sum(summands, dim=-1)    