import argparse
import torch
import matplotlib.pyplot as plt
from collections import Counter
import model_handler


def start(input_args):
    get_head_indices(input_args.lan_list, input_args.q, min_repetition=input_args.min_rep, model_name=input_args.model)


def get_head_indices(lan_list, q, min_repetition=4, model_name="aya"):
    sorted_init_head_list = []
    for lan in lan_list:
        loss_drop_matrix = torch.load(f"../results/{model_name}/{model_name}_{lan}.pth").to(dtype=float)
        flattened = loss_drop_matrix.view(-1)
        threshold = torch.quantile(flattened, q=q)
        percentile_indices = (flattened > threshold).nonzero(as_tuple=False).squeeze()
        values = flattened[percentile_indices]
        sorted_idx = torch.argsort(values, descending=True)
        sorted_percentile_indices = percentile_indices[sorted_idx]
        sorted_init_head_list.append(sorted_percentile_indices)

    all_indices = torch.cat(sorted_init_head_list).tolist()
    index_counts = Counter(all_indices)
    disqualified = [idx for idx, count in sorted(index_counts.items(), key=lambda x: -x[1]) if count >= min_repetition]
    # print("Repeated: ", disqualified, "\n")

    final_indices = []
    for i, topk_indices in enumerate(sorted_init_head_list):
        filtered_indices = [idx for idx in topk_indices.tolist() if idx not in disqualified]
        final_indices.append(filtered_indices)
    # print([len(item) for item in final_indices])

    # Save heads
    final_indices = {lan: list(item) for lan, item in zip(lan_list, final_indices)}
    print(final_indices)
    model_handler.save_json(f"../results/{model_name}/head_indices.json", final_indices)
    model_handler.save_json(f"../results/{model_name}/repeated_indices.json", disqualified)


# Show the attention head distribution of each language
def show_attn_head_distribution(model, lan_list, model_name):
    plt.figure(figsize=(6, 4))

    for i, lan in enumerate(lan_list):
        filtered_indices_n = model_handler.load_json(f"../results/{model_name}/head_indices.json").get(lan)[:20]

        head_mask = torch.ones(model.config.num_hidden_layers, model.config.num_attention_heads)
        head_mask.view(-1)[filtered_indices_n] = torch.tensor(0.0)

        layer_zero_counts = (head_mask == 0).sum(dim=1)
        total_zeros = layer_zero_counts.sum()
        zero_percentage = (layer_zero_counts / total_zeros * 100)

        x = range(head_mask.shape[0])
        plt.plot(x, zero_percentage.tolist(), '-', label=lan)

    plt.xlabel("Layer")
    plt.ylabel("Head % per layer")
    plt.title("Important Attention Heads Distribution by Language")
    plt.legend()
    plt.tight_layout()
    plt.show()


def get_head_mask(model, test_type, head_mask, head_lan, n, model_name="aya", layer=0):
    if test_type == "random":
        random_indices = torch.randperm(head_mask.numel())[:9]
        head_mask.view(-1)[random_indices] = torch.tensor(0.0).to(model.dtype)
    elif test_type == "full":
        # Language-general heads + language-specific heads
        loss_drop_matrix = torch.load(f"../results/{model_name}/{model_name}_{head_lan}.pth")
        _, topk_indices = torch.topk(loss_drop_matrix.view(-1), k=n)
        head_mask.view(-1)[topk_indices] = torch.tensor(0.0).to(model.dtype)
    elif test_type == "specific":
        # Only language-specific heads
        filtered_indices_n = model_handler.load_json(f"../results/{model_name}/head_indices.json").get(head_lan)[:n]
        head_mask.view(-1)[filtered_indices_n] = torch.tensor(0.0).to(model.dtype)
    elif test_type == "general":
        # Only language-general heads
        filtered_indices_n = model_handler.load_json(f"../results/{model_name}/repeated_indices.json")
        head_mask.view(-1)[filtered_indices_n] = torch.tensor(0.0).to(model.dtype)
    elif test_type == "layer":
        head_mask[layer] = torch.tensor(0.0).to(model.dtype)

    return head_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the modifier")

    parser.add_argument("--model", default="aya", choices=MODEL_CHOICES, help="model to apply the modifier")
    parser.add_argument("--device", help="training device")
    parser.add_argument("-b", "--half-precision", action="store_true", default=False,
                        help="set precision to torch.bfloat16")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="verbose")
    parser.add_argument("--local", type=bool, default=True, help="if load model from local dir")
    parser.add_argument("--lan-list", type=str, default="en", nargs='+', help="language list")
    parser.add_argument("--q", type=float, default=0.96, help="q")
    parser.add_argument("--min-rep", type=int, default=4, help="minimum number of repetition")

    args = parser.parse_args()

    start(args)
