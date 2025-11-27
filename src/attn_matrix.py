import torch
from tqdm.auto import tqdm
import torch.nn as nn
import argparse
import datasets
import pandas as pd
import model_handler


MODEL_CHOICES = ["aya", "mistral", "llama3"]


def start(input_args):
    device = input_args.device or "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_handler.load_model(input_args.model, device, input_args.half_precision, input_args.local)

    _ = get_attn_head_matrix(model, tokenizer, input_args.lan, model_name=input_args.model, data_num=input_args.data_num)


# E.g. get_wiki_data("hi_train-00001-of-00002.parquet", "hi")
# From https://huggingface.co/datasets/wikimedia/wikipedia
def get_wiki_data(file_path, lan):
    f = pd.read_parquet(file_path)
    data_list = []
    for text in f.get("text")[0:50000]:
        data_list.append(text)
    model_handler.save_json(f"../data/wikipedia/wikipedia_{lan}.json", data_list)


# Get attention head importance matrix using LAHIS and save the .pth file
def get_attn_head_matrix(model, tokenizer, lan, model_name="aya", data_num=1000):
    dataset = datasets.load_dataset("json", data_files=f"../data/wikipedia/wikipedia_{lan}.json", split="train")
    dataset = dataset.shuffle(seed=86).select(range(data_num))

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads

    head_mask = nn.Parameter(torch.ones(num_layers, num_heads, dtype=torch.bfloat16), requires_grad=True)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    total_head_importance = torch.zeros_like(head_mask)
    neg_grad_counts = torch.zeros_like(head_mask, dtype=torch.int)

    optimizer = torch.optim.AdamW([head_mask], lr=1e-2)
    for data_dict in tqdm(dataset):
        input_ids = tokenizer(data_dict.get("text"), return_tensors="pt", truncation=True, padding=True,
                              max_length=2048).input_ids.to(model.device)

        outputs = model(input_ids, labels=input_ids, head_mask=head_mask.to(model.device))
        loss = outputs.loss

        optimizer.zero_grad()
        loss.backward()

        total_head_importance += (head_mask.grad.abs() * head_mask * 1000).detach()
        neg_grad_counts += (head_mask.grad * head_mask < 0).int()

        optimizer.step()

    average_head_importance = total_head_importance / len(dataset)
    average_neg_grad_counts = neg_grad_counts.float() / len(dataset)
    final_matrix = average_head_importance * average_neg_grad_counts

    torch.save(final_matrix, f"../results/{model_name}/{model_name}_{lan}.pth")

    return final_matrix


# python3 attn_matrix.py --model aya -b --lan en --data-num 1000
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the modifier")

    parser.add_argument("--model", default="aya", choices=MODEL_CHOICES, help="model to apply the modifier")
    parser.add_argument("--device", help="training device")
    parser.add_argument("-b", "--half-precision", action="store_true", default=False,
                        help="set precision to torch.bfloat16")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="verbose")
    parser.add_argument("--local", type=bool, default=True, help="if load model from local dir")
    parser.add_argument("--lan", type=str, default="en", help="language")
    parser.add_argument("--data-num", type=int, default=1000, help="number of data")

    args = parser.parse_args()

    start(args)
