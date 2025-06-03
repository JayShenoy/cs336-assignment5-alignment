from vllm import LLM, SamplingParams
import json
from collections.abc import Callable
from typing import List
import pandas as pd

from cs336_alignment.drgrpo_grader import r1_zero_reward_fn


with open('cs336_alignment/prompts/r1_zero.prompt', 'r') as f:
    R1_ZERO_PROMPT = f.read()

def evaluate_vllm(
    vllm_model: LLM,
    reward_fn: Callable[[str, str], dict[str, float]],
    prompts: List[str],
    answers,
    eval_sampling_params: SamplingParams
    ) -> None:
    """
    Evaluate a language model on a list of prompts,
    compute evaluation metrics, and serialize results to disk.
    """
    outputs = vllm_model.generate(prompts, eval_sampling_params)

    results = []

    for output, answer in zip(outputs, answers):
        prompt = output.prompt
        generated_text = output.outputs[0].text

        reward = r1_zero_reward_fn(generated_text, answer)

        results.append({
            'prompt': prompt,
            'response': generated_text,
            'correct_answer': answer,
            'reward': reward,
        })
    
    return results

def eval_qwen_math():
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

    sampling_params = SamplingParams(
        temperature=1.0, top_p=1.0, max_tokens=1024
    )
    sampling_params.stop = ["</answer>"]
    sampling_params.include_stop_str_in_output = True

    llm = LLM(model='/data/a5-alignment/models/Qwen2.5-Math-1.5B')

    results = evaluate_vllm(llm, r1_zero_reward_fn, prompts, answers, sampling_params)

    with open('/home/c-jshenoy/cs336-assignment5-alignment/cs336_alignment/output/qwen_baseline_perf.json', 'w') as f:
        json.dump(results, f, indent=4)

def print_qwen_eval_results():
    with open('/home/c-jshenoy/cs336-assignment5-alignment/cs336_alignment/output/qwen_baseline_perf.json', 'r') as f:
        results = json.load(f)
    
    format_one_answer_one = 0
    format_one_answer_zero = 0
    format_zero_answer_zero = 0

    format_zero_10_examples = []
    format_one_answer_zero_10_examples = []
    
    # Tally each reward combination
    for r in results:
        format_reward = r['reward']['format_reward']
        answer_reward = r['reward']['answer_reward']

        if format_reward == 1 and answer_reward == 1:
            format_one_answer_one += 1
        elif format_reward == 1 and answer_reward == 0:
            format_one_answer_zero += 1

            if len(format_one_answer_zero_10_examples) < 10:
                format_one_answer_zero_10_examples.append(r)
        elif format_reward == 0 and answer_reward == 0:
            format_zero_answer_zero += 1

            if len(format_zero_10_examples) < 10:
                format_zero_10_examples.append(r)

    reward_labels = [
        'Format Reward = 1, Answer Reward = 1',
        'Format Reward = 1, Answer Reward = 0',
        'Format Reward = 0, Answer Reward = 0',
    ]

    tallies = [
        format_one_answer_one,
        format_one_answer_zero,
        format_zero_answer_zero,
    ]
    
    df = pd.DataFrame({
        'Reward': reward_labels,
        'Tally': tallies,
    })

    table_md = df.to_markdown(index=False)
    print(table_md)

    print('10 examples where format reward = 0:')
    print(json.dumps(format_zero_10_examples, indent=4))

    print('10 examples where format reward = 1, answer reward = 0:')
    print(json.dumps(format_one_answer_zero_10_examples, indent=4))

if __name__ == '__main__':
    # eval_qwen_math()
    print_qwen_eval_results()