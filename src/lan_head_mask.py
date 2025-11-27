import argparse
import torch
import torch.nn as nn
import datasets
from tqdm.auto import tqdm
import model_handler


MODEL_CHOICES = ["aya", "mistral", "llama3"]


def start(input_args):
    device = input_args.device or "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_handler.load_model(input_args.model, device, input_args.half_precision, input_args.local)

    ft_with_attn_heads(model, tokenizer, input_args.lan, p=0.02, data_split="train[:390]",
                       data_num=200, test_type="test", model_name=input_args.model)

    return


def ft_with_attn_heads(model, tokenizer, lan, p=0.02, data_split="train", data_num=100, test_type="test", model_name="aya"):
    xquad_prompt = model_handler.load_json("../data/xquad/xquad_prompt.json")
    # xquad: context; question; answer
    dataset = datasets.load_dataset("json", data_files=f"../data/xquad/xquad_{lan}.json", split=data_split)
    dataset = dataset.shuffle(seed=68).select(range(data_num))

    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    n = int(num_layers * num_heads * p)

    head_mask = nn.Parameter(torch.ones(num_layers, num_heads, dtype=torch.bfloat16), requires_grad=True)

    importance_mask = torch.zeros(num_layers * num_heads, dtype=torch.bool)
    if test_type == "test":
        import_matrix = torch.load(f"../results/{model_name}/{model_name}_{lan}.pth")
        _, topk_indices = torch.topk(import_matrix.view(-1), k=n)
    elif test_type == "random":
        topk_indices = torch.randperm(head_mask.numel())[:n]
    else:
        print("Wrong test_type!")
    importance_mask[topk_indices] = True

    def mask_gradient(grad):
        flat_grad = grad.view(-1)
        flat_grad[~importance_mask.to(flat_grad.device)] = 0
        return flat_grad.view(num_layers, num_heads)

    head_mask.register_hook(mask_gradient)

    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW([head_mask], lr=1e-2)
    max_epoch = 2
    for epoch in range(max_epoch):
        total_loss = 0
        for data_dict in tqdm(dataset):
            context = data_dict.get("context", "")
            question = data_dict.get("question", "")
            answer = data_dict.get("answer", "").strip(".").strip()
            prompt = xquad_prompt.get(f"xquad_{lan}").replace("&&context&&", context, 1).replace("&&question&&",
                                                                                                 question, 1)

            prompt_input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True,
                                         max_length=2048).input_ids.to(model.device)
            input_ids = tokenizer(prompt + answer, return_tensors="pt", truncation=True, padding=True,
                                  max_length=2048).input_ids.to(model.device)
            labels = input_ids.clone()
            labels[:, :prompt_input_ids.shape[1]] = -100

            outputs = model(input_ids, labels=labels, head_mask=head_mask.to(model.device))

            loss = outputs.loss
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"epoch {epoch} - loss: {total_loss / len(dataset)}")

    torch.save(head_mask, f"../results/{model_name}/head_mask/{model_name}_xquad_{lan}_{test_type}.pth")

    return head_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the modifier")

    parser.add_argument("--model", default="aya", choices=MODEL_CHOICES, help="model to apply the modifier")
    parser.add_argument("--device", help="training device")
    parser.add_argument("-b", "--half-precision", action="store_true", default=False,
                        help="set precision to torch.bfloat16")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="verbose")
    parser.add_argument("--local", type=bool, default=True, help="if load model from local dir")
    parser.add_argument("--lan", type=str, default="en", help="language")

    args = parser.parse_args()

    start(args)
