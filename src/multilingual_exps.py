import argparse
import torch
import datasets
from tqdm.auto import tqdm
from bert_score import score
import unicodedata
from collections import Counter
import fasttext
import language_heads
import model_handler

path_to_pretrained_model = '../resource/lid.176.bin'
fmodel = fasttext.load_model(path_to_pretrained_model)

MODEL_CHOICES = ["aya", "mistral", "llama3"]
EXP_CHOICES = ["ppl", "ppl-cross", "general_head", "off_target_lan", "attn_transfer", "head_mask"]


def start_exps(input_args):
    device = input_args.device or "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = model_handler.load_model(input_args.model, device, input_args.half_precision, input_args.local)

    res_list = []
    if input_args.exp_type == "ppl":
        test_type_list = ["ori", "random", "specific"]
        res_list = [[] for _ in test_type_list]
        for lan in input_args.lan_list:
            for i, test_type in enumerate(test_type_list):
                ppl = get_ppl_res(model, tokenizer, test_lan=lan, head_lan=lan, data_split="train", data_num=1000,
                                  p=0.02, test_type=test_type, model_name=input_args.model)
                res_list[i].append(ppl)
        print("ppl: ", res_list)
    elif input_args.exp_type == "ppl-cross":
        res_list = [[] for _ in input_args.lan_list]
        for test_lan in input_args.lan_list:
            for i, head_lan in enumerate(input_args.lan_list):
                ppl = get_ppl_res(model, tokenizer, test_lan=test_lan, head_lan=head_lan, data_split="train",
                                  data_num=1000, p=0.02, test_type="specific")
                res_list[i].append(ppl)
        print("ppl-cross: ", res_list)
    elif input_args.exp_type == "general_head":
        test_type_list = ["ori", "random", "general"]
        res_list = [[] for _ in test_type_list]
        for lan in input_args.lan_list:
            for i, test_type in enumerate(test_type_list):
                F1, _ = test_xlsum(model, tokenizer, lan, data_num=500, p=0.01, test_type=test_type)
                res_list[i].append(F1)
        print(res_list)
    elif input_args.exp_type == "off_target_lan":
        test_type_list = ["ori", "specific", "full"]
        res_list = [[] for _ in test_type_list]
        for lan in input_args.lan_list:
            for i, test_type in enumerate(test_type_list):
                _, lan_percent = test_xlsum(model, tokenizer, lan, data_num=500, p=0.01, test_type=test_type)
                res_list[i].append(lan_percent)
        print(res_list)
    elif input_args.exp_type == "attn_transfer":
        test_type_list = ["ori", "test_lan2", "test_lan1"]
        for test_type in test_type_list:
            res_list_item = transfer_attn(model, tokenizer, input_args.lan_list, model_name=input_args.model,
                                          test_type=test_type)
            res_list.append(res_list_item)
        print(res_list)
    elif input_args.exp_type == "head_mask":
        test_type_list = ["ori", "random", "test"]
        res_list = [[] for _ in test_type_list]
        for lan in input_args.lan_list:
            for i, test_type in enumerate(test_type_list):
                acc = get_xquad_accuracy_attn(model, tokenizer, lan, data_split="train[390:1190]",
                                              test_type=test_type, model_name=input_args.model)
                res_list[i].append(round(acc * 100, 2))
        print(res_list)
    else:
        print("Wrong exp-type parameter!")

    return res_list


# PPL
def get_ppl_res(model, tokenizer, test_lan="en", head_lan="en", data_split="train", data_num=100, p=0.01, test_type="specific", model_name="aya", layer=0):
    layer_num = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    n = int(layer_num * num_heads * p)

    # Get head_mask
    head_mask = torch.ones(layer_num, num_heads).to(model.device).to(model.dtype)
    head_mask = language_heads.get_head_mask(model, test_type, head_mask, head_lan, n, model_name=model_name, layer=layer)

    dataset = datasets.load_dataset("json", data_files=f"../data/wikipedia/wikipedia_{test_lan}.json", split=data_split)
    dataset = dataset.shuffle(seed=37).select(range(data_num))

    # Get ppl
    ppl_avg = 0
    for data_dict in tqdm(dataset):
        input_ids = tokenizer(data_dict.get("text"), return_tensors="pt", truncation=True, padding=True, max_length=2048).input_ids.to(model.device)
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids, head_mask=head_mask.to(model.device))
            ppl = torch.exp(outputs.loss)
            ppl_avg += ppl.item()

    ppl_avg /= len(dataset)
    ppl_avg = round(ppl_avg, 2)
    # print(f"{test_lan} - {test_type} ppl: {ppl_avg}")

    return ppl_avg


# XLSUM
def test_xlsum(model, tokenizer, lan, data_num=20, p=0.01, test_type="ori"):
    layer_num = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    n = int(layer_num * num_heads * p)
    # xlsum: summary; text
    dataset = datasets.load_dataset("json", data_files=f"../data/xlsum/xlsum_{lan}.json", split="train")
    dataset = dataset.shuffle(seed=68).select(range(data_num))
    # rouge = load("../utils/rouge.py")
    # bertscore = load("../utils/bertscore.py")

    # Get head_mask
    head_mask = torch.ones(layer_num, num_heads).to(model.device).to(model.dtype)
    head_mask = language_heads.get_head_mask(model, test_type, head_mask, lan, n)

    def generate_summary(text, head_mask, max_new_tokens=80):
        def clean_text(text):
            text = unicodedata.normalize("NFC", text).strip()
            # text = re.sub(r"[\u200b\u00a0\r\n\t]+", "", text)
            return text

        ques_tem_dict = {
            "en": "Summarize the given text briefly. [Text]&&text&& [Summary]",
            "zh": "简短地总结给出的文本。[文本]&&text&& [总结]",
            "vi": "Tóm tắt ngắn gọn văn bản sau. [văn bản]&&text&& [Tóm tắt]",
            "hi": "दिए गए पाठ का संक्षेप में सारांश लिखें। [पाठ]&&text&& [सारांश]",
            "fr": "Résumez brièvement le texte donné. [Texte]&&text&& [Résumé]",
            "th": "สรุปข้อความที่กำหนดโดยย่อ [ข้อความ]&&text&& [สรุป]",
            "es": "Resume brevemente el texto dado. [Texto]&&text&& [Resumen]",
            "it": "Riassumi brevemente il testo fornito. [Testo]&&text&& [Riassunto]",
            "pt": "Resuma brevemente o texto fornecido. [Texto]&&text&& [Resumo]",
            "id": "Ringkas teks berikut secara singkat. [Teks]&&text&& [Ringkasan]",
            "ja": "次の文章を簡潔に要約してください。[文章]&&text&& [要約]",
            "ko": "다음 글을 간단히 요약하세요. [본문]&&text&& [요약]"
        }
        prompt = clean_text(ques_tem_dict.get(lan).replace("&&text&&", text, 1))
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=max_new_tokens, head_mask=head_mask,
                                     pad_token_id=tokenizer.eos_token_id)
            summary = clean_text(tokenizer.decode(outputs[0], skip_special_tokens=False)).replace("<BOS_TOKEN>", "")
        if prompt in summary:
            summary = summary.replace(prompt, "").replace("<s>", "").replace("</s>", "").strip()
        else:
            summary = ""

        return summary

    predictions = []
    references = []
    languages = []
    for data_dict in tqdm(dataset):
        src_text = data_dict["text"]
        ref_summary = data_dict["summary"]
        pred_summary = generate_summary(src_text, head_mask)
        if pred_summary:
            # print("pred_summary: ", pred_summary)  ####
            languages.append(fmodel.predict(pred_summary.replace("\n", ""))[0][0].replace("__label__", ""))
            predictions.append(pred_summary)
            references.append(ref_summary)

    # rouge_result = rouge.compute(predictions=predictions, references=references)
    bert_P, bert_R, bert_F1 = score(
        cands=predictions,
        refs=references,
        model_type="/root/autodl-tmp/bert-base-uncased",
        lang=lan,
        num_layers=12,
        rescale_with_baseline=True
    )
    # bert_result = {"P": bert_P.mean().item(), "R": bert_R.mean().item(), "F1": bert_F1.mean().item()}
    F1 = round(bert_F1.mean().item() * 100, 2)

    print("F1:", F1)
    # print("BERTScore:", bert_result)

    # Language proportion (Exp: Off-target language generation)
    c = Counter(languages)
    lan_num = c.get(lan)
    lan_percent = round(lan_num / len(languages), 2)
    print("lan_percent: ", lan_percent)
    print("languages: ", languages)

    return F1, lan_percent


# Transfer attention
pos_template = {
    "zh": "Alice的职位是&&pos&&。",
    "fr": "Le métier d'Alice est &&pos&&.",
    "hi": "ऐलिस का पेशा &&pos&& है।",
    "vi": "Nghề nghiệp của Alice là &&pos&&.",
    "th": "อาชีพของAliceคือ&&pos&&.",
    "es": "La profesión de Alice es &&pos&&.",
    "it": "La professione di Alice è &&pos&&.",
    "pt": "A profissão de Alice é &&pos&&.",
    "id": "Pekerjaan Alice adalah &&pos&&.",
    "ja": "アリスの職業は&&pos&&です。",
    "ko": "앨리스의 직업은 &&pos&&입니다.",
    "el": "Το επάγγελμα της Alice είναι &&pos&&."
}


pos_dict = {
    "en": ["painter", "scientist", "doctor", "gardener", "lawyer", "dentist", "poet", "writer", "engineer", "director"],
    "zh": ["画家", "科学家", "医生", "园丁", "律师", "牙医", "诗人", "作家", "工程师", "导演"],
    "fr": ["peintre", "scientifique", "médecin", "jardinier", "avocat", "dentiste", "poète", "écrivain", "ingénieur", "réalisateur"],
    "hi": ["चित्रकार", "वैज्ञानिक", "डॉक्टर", "माली", "वकील", "दंत चिकित्सक", "कवि", "लेखक", "इंजीनियर", "निर्देशक"],
    "vi": ["họa sĩ", "nhà khoa học", "bác sĩ", "người làm vườn", "luật sư", "nha sĩ", "nhà thơ", "nhà văn", "kỹ sư", "đạo diễn"],
    "th": ["จิตรกร", "นักวิทยาศาสตร์", "หมอ", "คนสวน", "ทนายความ", "ทันตแพทย์", "กวี", "นักเขียน", "วิศวกร", "ผู้กำกับ"],
    "es": ["pintor", "científico", "médico", "jardinero", "abogado", "dentista", "poeta", "escritor", "ingeniero", "director"],
    "it": ["pittore", "scienziato", "medico", "giardiniere", "avvocato", "dentista", "poeta", "scrittore", "ingegnere", "regista"],
    "pt": ["pintor", "cientista", "médico", "jardineiro", "advogado", "dentista", "poeta", "escritor", "engenheiro", "diretor"],
    "id": ["pelukis", "ilmuwan", "dokter", "tukang kebun", "pengacara", "dokter gigi", "penyair", "penulis", "insinyur", "sutradara"],
    "ja": ["画家", "科学者", "医者", "庭師", "弁護士", "歯科医", "詩人", "作家", "技術者", "映画監督"],
    "ko": ["화가", "과학자", "의사", "정원사", "변호사", "치과의사", "시인", "작가", "엔지니어", "감독"],
    "el": ["ζωγράφος", "επιστήμονας", "γιατρός", "κηπουρός", "δικηγόρος", "οδοντίατρος", "ποιητής", "συγγραφέας", "μηχανικός", "σκηνοθέτης"]
}


def get_target_logit_rank(tokenizer, output_logits, target):
    target_ids = tokenizer(target, return_tensors="pt").input_ids
    target_id = target_ids[0, 1]
    _, sorted_indices = torch.sort(output_logits, descending=True, dim=-1)
    target_rank = torch.nonzero(sorted_indices == target_id).squeeze().item()

    return target_rank


def transfer_attn(model, tokenizer, lan_list, p=0.01, model_name="aya", test_type="test_lan1"):
    layer_num = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    n = int(layer_num * num_heads * p)

    res_list = [[] for _ in lan_list]
    for lan1 in tqdm(lan_list):
        print("lan1: ", lan1)
        total = 0
        lan1_num = 0
        lan2_num = 0
        chose1_percent = 0

        head_mask = torch.ones(layer_num, num_heads).to(model.device).to(model.dtype)

        # Enhance lan-1
        if test_type == "test_lan1" or test_type == "test":
            head_mask = torch.ones(layer_num, num_heads)
            filtered_indices_n = model_handler.load_json(f"../results/{model_name}/head_indices.json").get(lan1)[:n]
            head_mask.view(-1)[filtered_indices_n] = torch.tensor(3.0)  # hyper-parameter
            head_mask = head_mask.to(model.device).to(model.dtype)

        for res_i, lan2 in enumerate(lan_list):
            lan1_num_2 = 0
            lan2_num_2 = 0
            total_2 = 0
            if lan1 == lan2:
                res_list[res_i].append((-1, -1))
                continue

            # Weaken lan-2
            if test_type == "test_lan2" or test_type == "test":
                head_mask = torch.ones(layer_num, num_heads)
                filtered_indices_n = model_handler.load_json(f"../results/{model_name}/final_head_indices.json").get(lan2)[:n]
                head_mask.view(-1)[filtered_indices_n] = torch.tensor(0.0)
                head_mask = head_mask.to(model.device).to(model.dtype)

            for i, pos1 in enumerate(pos_dict["en"]):
                for j, pos2 in enumerate(pos_dict["en"]):
                    if i == j:
                        continue
                    total += 1
                    total_2 += 1
                    prompt = (f"{pos_template.get(lan1)}"
                              f" She likes ice-cream. She always do sports in gym at morning. "
                              f"{pos_template.get(lan2)} "
                              f"Alice's occupation is a")
                    prompt = (prompt.replace("&&pos&&", pos_dict.get(lan1)[i], 1)
                              .replace("&&pos&&", pos_dict.get(lan2)[j], 1))

                    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
                    with torch.no_grad():
                        outputs = model.generate(**inputs, max_new_tokens=5, head_mask=head_mask)
                        gene = tokenizer.decode(outputs[0], skip_special_tokens=False).replace("<BOS_TOKEN>",
                                                                                               "").replace(prompt,
                                                                                                           "").strip()

                        if pos1 in gene or pos_dict.get(lan1)[i] in gene:
                            lan1_num += 1
                            lan1_num_2 += 1

                        elif pos2 in gene or pos_dict.get(lan2)[j] in gene:
                            lan2_num += 1
                            lan2_num_2 += 1

                        # logit rank
                        output_logits = model(**inputs, return_dict=True, head_mask=head_mask).logits
                        logitrank_pos1 = get_target_logit_rank(tokenizer, output_logits[0, -1], " " + pos1)
                        logitrank_pos2 = get_target_logit_rank(tokenizer, output_logits[0, -1], " " + pos2)
                        l_pos1 = get_target_logit_rank(tokenizer, output_logits[0, -1], pos_dict.get(lan1)[i])
                        l_pos2 = get_target_logit_rank(tokenizer, output_logits[0, -1], pos_dict.get(lan2)[j])
                        if min(logitrank_pos1, l_pos1) < min(logitrank_pos2, l_pos2):
                            chose1_percent += 1

            res_list[res_i].append((round(lan1_num_2 / total_2, 2), round(lan2_num_2 / total_2, 2)))

        chose1_percent = round(chose1_percent / total, 2)
        print("chose1_percent: ", chose1_percent)

        lan1_percent = round(lan1_num / total, 2)
        lan2_percent = round(lan2_num / total, 2)
        print(f"total: {total}")
        print(f"1: {lan1_percent}; 2: {lan2_percent}")

    return res_list


def transfer_attn_example(model, tokenizer, model_name, exp_type="enhance"):
    head_mask = torch.ones(32, 32).to(model.device)

    # Enhance zh or Weaken hi to shift output to scientist (in zh context);
    # else: original output of teacher (in hi context)
    if exp_type == "enhance":
        filtered_indices_n = model_handler.load_json(f"../results/{model_name}/head_indices.json").get("zh")[:20]
        head_mask.view(-1)[filtered_indices_n] = torch.tensor(5.0)
    elif exp_type == "weaken":
        # Weaken
        filtered_indices_n = model_handler.load_json(f"../results/{model_name}/head_indices.json").get("hi")[:20]
        head_mask.view(-1)[filtered_indices_n] = torch.tensor(0.0)

    prompt = ("Alice的职位是科学家。She likes ice-cream. She always do sports in gym at morning. "
              "ऐलिस पेशे के द्वारा एक शिक्षक है. Alice's occupation is a")
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=10, head_mask=head_mask.to(model.dtype).to(model.device))
        gene = tokenizer.decode(outputs[0], skip_special_tokens=False).replace("<BOS_TOKEN>", "").replace(prompt,
                                                                                                          "").strip()
        print("\n Generation: ", gene)

        tokens_list, possis_list = model_handler.output_layer_logits(model, tokenizer, inputs.input_ids,
                                                                     head_mask.to(model.dtype).to(model.device),
                                                                     token_num=5)

    return tokens_list, possis_list


# Language-Speciffc Head Mask
def get_xquad_accuracy_attn(model, tokenizer, lan, data_split="train[:10]", test_type="test", layer=0, model_name="aya"):
    num_layers = model.config.num_hidden_layers
    num_heads = model.config.num_attention_heads
    # xquad: context; question; answer
    xquad_prompt = model_handler.load_json("../data/xquad/xquad_prompt.json")
    dataset = datasets.load_dataset("json", data_files=f"../data/xquad/xquad_{lan}.json", split=data_split)

    if test_type == "test":
        head_mask = torch.load(f"../results/{model_name}/head_mask/{model_name}_xquad_{lan}_test.pth").to(
            model.device).to(model.dtype)
    elif test_type == "random":
        head_mask = torch.load(f"../results/{model_name}/head_mask/{model_name}_xquad_{lan}_random.pth").to(
            model.device).to(model.dtype)
    elif test_type == "weaken":
        head_mask = torch.ones(num_layers, num_heads)
        filtered_indices_n = model_handler.load_json(f"../results/{model_name}/head_indices.json").get(lan)[:20]
        head_mask.view(-1)[filtered_indices_n] = torch.tensor(0.0)
    elif test_type == "layer":
        head_mask = torch.ones(num_layers, num_heads)
        head_mask[layer] = torch.tensor(0.0)
    else:  # ori
        head_mask = torch.ones(num_layers, num_heads).to(model.device).to(model.dtype)
    head_mask = head_mask.to(model.device).to(model.dtype)

    true_num = 0
    for data_dict in tqdm(dataset):
        context = data_dict.get("context", "")
        question = data_dict.get("question", "")
        answer = data_dict.get("answer", "").strip(".").lower().strip()
        prompt = xquad_prompt.get(f"xquad_{lan}").replace("&&context&&", context, 1).replace("&&question&&", question, 1)

        input_ids = model_handler.get_model_inputs(tokenizer, prompt, model.device).input_ids

        gene_attn = model_handler.generate_with_ori_model(model, tokenizer, input_ids, 10, head_mask=head_mask)
        new_gene_attn = gene_attn[0].replace("<BOS_TOKEN>", "").replace(prompt, "").lower().strip()

        if answer in new_gene_attn:
            true_num += 1

    accuracy = true_num / len(dataset)
    print(f"{lan} - {test_type}: {accuracy}")

    return accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the modifier")

    parser.add_argument("--model", default="aya", choices=MODEL_CHOICES, help="model to apply the modifier")
    parser.add_argument("--device", help="training device")
    parser.add_argument("-b", "--half-precision", action="store_true", default=False,
                        help="set precision to torch.bfloat16")
    parser.add_argument("-v", "--verbose", action="store_true", default=False, help="verbose")
    parser.add_argument("--local", type=bool, default=True, help="if load model from local dir")
    parser.add_argument("--lan-list", type=str, default="en", nargs='+', help="language list to test")
    parser.add_argument("--exp-type", default="ppl", choices=EXP_CHOICES, help="type of experiments")

    args = parser.parse_args()

    start_exps(args)
