from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nethook import Trace, TraceDict
import dotenv
import os
import json

# Model path
AYA_NAME = "CohereLabs/aya-23-8B"
MISTRAL_NAME = "mistralai/Mistral-7B-v0.1"
LLAMA3_NAME = "meta-llama/Llama-3.2-3B"

# TODO Input your local path of LLMs
AYA_NAME_LOCAL = "../../autodl-tmp/aya-23-8B"
MISTRAL_NAME_LOCAL = "../../autodl-tmp/Mistral-7B-v0.1"
LLAMA3_NAME_LOCAL = "../../autodl-tmp/Llama-3.2-3B"


# Get access token
dotenv.load_dotenv()
HF_TOKEN = os.getenv('HF_TOKEN')


def load_json(json_path):
    with open(json_path, "r", encoding='utf-8') as fp:
        data = json.load(fp)

    return data


def save_json(json_path, data):
    with open(json_path, "w", encoding='utf-8') as fp:
        json.dump(data, fp, indent=4)

    return


def generate_with_ori_model(model, tokenizer, input_ids_batch, max_new_tokens=10, head_mask=None, head_importance=None):
    with torch.no_grad():
        output_dict_ori = model.generate(input_ids_batch, max_new_tokens=max_new_tokens, pad_token_id=tokenizer.eos_token_id,
                                         return_dict_in_generate=True, output_logits=True,
                                         head_mask=head_mask)
        output = tokenizer.batch_decode(output_dict_ori.sequences, skip_special_tokens=False,
                                        clean_up_tokenization_spaces=False)

    # print(f"ori output: {output}")

    return output


def load_layer_param(model, target_layer, save_path):
    layer_state_dict = torch.load(save_path, map_location=model.device)
    model_dict = model.state_dict()

    for name, param in layer_state_dict.items():
        if name in model_dict:
            model_dict[name].copy_(param)
        else:
            print(f"Not found {name} in model.")

    print(f"Load layer {target_layer} weights from {save_path}")


def save_layer_param(model, target_layer, save_path):
    layer_state_dict = {
        name: param for name, param in model.state_dict().items()
        if f".{target_layer}." in name
    }

    torch.save(layer_state_dict, save_path)
    print(f"Save layer {target_layer} weights to {save_path}")


def get_model_inputs(tokenizer, text, device):
    # inputs: input_ids, attention_mask, offset_mapping
    inputs = tokenizer(text, return_tensors="pt", return_offsets_mapping=True, padding="longest")
    inputs.to(device)

    return inputs


def trace_model(model, input_ids, head_mask, token_num=1, module_prefix="model.layers.", target=""):
    hook_layers = [f'{module_prefix}{l}{target}' for l in range(model.config.num_hidden_layers)]

    with TraceDict(model, layers=hook_layers, retain_output=True) as res:
        model(input_ids, head_mask=head_mask)

    if len(res[hook_layers[0]].output[0].shape) == 3:
        output_res = [res[hook].output[0][0][-token_num:] for hook in hook_layers]
    elif len(res[hook_layers[0]].output[0].shape) == 2:
        output_res = [res[hook].output[0][-token_num:] for hook in hook_layers]
    else:
        raise ValueError("Other res[hook_layers[0]].output[0].shape")

    return output_res


def output_layer_logits(model, tokenizer, input_ids, head_mask, token_num=1):
    hs_res = trace_model(model, input_ids, head_mask, token_num, module_prefix="model.layers.", target="")

    tokens_list = []
    possis_list = []
    for l in range(model.config.num_hidden_layers):
        logits_l = model.lm_head(model.model.norm(hs_res[l]))
        logit_softmax_l = torch.softmax(logits_l, dim=-1)
        top_indices = torch.topk(logits_l, 1, dim=-1, sorted=True).indices

        decoded_tokens = [tokenizer.decode(top_indices[i], skip_special_tokens=False) for i in range(token_num)]
        tokens_list.append(decoded_tokens)
        token_possis = [logit_softmax_l[i][top_indices[i]].item() for i in range(token_num)]
        possis_list.append(token_possis)

        print(f"layer {l}: {decoded_tokens}")

    return tokens_list, possis_list


# Load model&tokenizer
def load_model(model_name, device, half_precision, local=True):
    if local:
        model_dict = {"aya": AYA_NAME_LOCAL,
                      "mistral": MISTRAL_NAME_LOCAL,
                      "llama3": LLAMA3_NAME_LOCAL
                      }
    else:
        model_dict = {"aya": AYA_NAME_LOCAL,
                      "mistral": MISTRAL_NAME_LOCAL,
                      "llama3": LLAMA3_NAME_LOCAL
                      }

    model_path = model_dict.get(model_name)
    if not model_path:
        print("Wrong model name.")
        raise ValueError("Wrong model name.")

    tokenizer = AutoTokenizer.from_pretrained(model_path, token=HF_TOKEN)
    tokenizer.pad_token = tokenizer.eos_token

    if half_precision:
        print("Load model in torch.bfloat16")
        model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN,
                                                     torch_dtype=torch.bfloat16)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_path, token=HF_TOKEN)

    model.eval()
    model.to(device)

    # Calculate the memory usage of LLM
    memory_allocated_gb = torch.cuda.memory_allocated(device=device) / (1024 ** 3)
    print(f"** Memory allocated by model: {memory_allocated_gb} GB\n")

    return model, tokenizer
