from deita.selection.scorer import Llama_Scorer, RewardScorer
import json
from tqdm import tqdm
import torch

dataset = "cohere"

model_name_or_path = "weqweasdas/hh_rlhf_rm_open_llama_3b"
data_path = f"data-selection/data/processed/{dataset}/{dataset}_data.jsonl"
output_path = f"data-selection/data/processed/{dataset}/"
BATCH_SIZE = 4

scorer = RewardScorer(model_name_or_path, batch_size = BATCH_SIZE)

# example input
# input_text = "word to describe UI with helpful tooltips" # Example Input
# output_text = "User-friendly or intuitive UI" # Example Output
# quality_score = scorer.infer_quality(input_text, output_text)
scores = []

with open(data_path) as f:
    lines = f.readlines()
for i in tqdm(range(0, len(lines), BATCH_SIZE)):
    batches = lines[i:i+BATCH_SIZE]
    messages = [json.loads(line)["messages"] for line in batches]
    # filter the system messages
    messages = [[m for m in message if m["role"] != "system"] for message in messages]

    len_conversation = [len(message) // 2 for message in messages]
    index = [i for i, count in enumerate(len_conversation) for _ in range(count)]
    local_index = [i for _, count in enumerate(len_conversation) for i in range(count)]
    inputs = [messages[j][2*i]["content"] for i, j in zip(local_index, index)]
    outputs = [messages[j][2*i+1]["content"] for i, j in zip(local_index, index)]
    rewards = scorer.batch_infer(inputs, outputs)

    index, reward_scores = torch.tensor(index, dtype=torch.int64), torch.tensor(rewards, dtype=torch.float32)
    dummy = torch.zeros(BATCH_SIZE, dtype=torch.float32)

    # gather reduce scores based on index
    reward_scores = dummy.scatter_add(0, index, reward_scores).tolist()
    # dviding by the number of messages in the conversation
    reward_scores = [reward_score / len_conversation[i] for i, reward_score in enumerate(reward_scores)]

with open(output_path + "/reward_scores.jsonl", "w") as f:
    for score in scores:
        f.write(json.dumps(score) + "\n")

    # quality_score = scorer.infer_quality(input_text, output_text)

# output the scores
# print(quality_score)
        
# nohup python examples/scoring/score_vllm.py > examples/scoring/score_vllm.log 2>&1 &