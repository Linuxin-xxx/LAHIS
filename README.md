# Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models

## Abstract
Large language models (LLMs) increasingly support multilingual understanding and generation. Meanwhile, efforts to interpret their internal mechanisms have emerged, offering insights to enhance multilingual performance. While multi-head self-attention (MHA) has proven critical in many areas, its role in multilingual capabilities remains underexplored. In this work, we study the contribution of MHA in supporting multilingual processing in LLMs. We propose Language Attention Head Importance Scores (LAHIS), an effective and efficient method that identifies attention head importance for multilingual capabilities via a single forward and backward pass through the LLM. Applying LAHIS to Aya-23-8B, Llama-3.2-3B, and Mistral-7B-v0.1, we reveal the existence of both language-specific and language-general heads. Language-specific heads enable cross-lingual attention transfer to guide the model toward target language contexts and mitigate off-target language generation issue, contributing to addressing challenges in multilingual LLMs. We also introduce a lightweight adaptation that learns a soft head mask to modulate attention outputs over language heads, requiring only 20 tunable parameters to improve XQuAD accuracy. Overall, our work enhances both the interpretability and multilingual capabilities of LLMs from the perspective of MHA.


## Prerequisite
In the experiments, the target LLM needs to support the ***head_mask*** parameter.

Otherwise, the following code needs to be added to the LLM source code:
```python
if head_mask is not None:
    attn_weights = attn_weights * head_mask
```
Also, the function parameter passing should be updated accordingly.

## How to Start
- **Obtain attention head importance matrix on multilingual capabilities via LAHIS:**
```
python3 attn_matrix.py --model aya -b --lan en --data-num 1000
```
  
Output:

results/{model_name}/{model_name}_{lan}.pth -> Attention head importance matrix

  
- **Obtain the language heads indices of the specified LLM:**
```
python3 language_heads.py --model aya -b --lan-list en zh hi vi es
```
Output:

results/{model_name}/head_indices.json -> Indices of language-specific heads

results/{model_name}/repeated_indices.json -> Indices of language-general heads


- **Obtain the language-specific head mask:**
```
python3 lan_head_mask.py --model aya -b --lan en
```
Output:

results/{model_name}/head_mask/{model_name}\_xquad_{lan}_{test_type}.pth -> language-specific head mask 

  
- **Reproduce multilingual experiments:**
```
python3 multilingual_exps.py --model aya -b --exp-type ppl --lan-list en zh hi vi es
```
exp-type: ppl, ppl-cross, general_head, off_target_lan, attn_transfer, head_mask


## How to Cite
```
@misc{liu2025focusinglanguagerevealingexploiting,
      title={Focusing on Language: Revealing and Exploiting Language Attention Heads in Multilingual Large Language Models}, 
      author={Xin Liu and Qiyang Song and Qihang Zhou and Haichao Du and Shaowen Xu and Wenbo Jiang and Weijuan Zhang and Xiaoqi Jia},
      year={2025},
      eprint={2511.07498},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2511.07498}, 
}
```
