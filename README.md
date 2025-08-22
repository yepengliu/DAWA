# DAWA

Official implementation of the LLM watermarking algorithm presented in the paper:

["Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach"](https://arxiv.org/abs/2410.02890) by Haiyun He*, Yepeng Liu*, Ziqiao Wang, Yongyi Mao and Yuheng Bu.


## Introduction

Watermarking has emerged as a crucial method to distinguish AI-generated text from human-created text. Current watermarking approaches often lack formal optimality guarantees or address the scheme and detector design separately. In this paper, we introduce a novel, unified theoretical framework for watermarking Large Language Models (LLMs) that jointly optimizes both the watermarking scheme and detector. Our approach aims to maximize detection performance while maintaining control over the worst-case false positive rate (FPR) and distortion on text quality. 
We derive closed-form optimal solutions for this joint design and characterize the fundamental trade-off between watermark detectability and distortion. Notably, we reveal that the optimal watermarking schemes should be adaptive to the LLM’s generative distribution. Building on our theoretical insights, we propose a distortion-free, distribution-adaptive watermarking algorithm (DAWA) that leverages a surrogate model for model-agnosticism and efficiency. Experiments on Llama2-13B and Mistral-8 $\times$ 7B models confirm the effectiveness of our approach, particularly at ultra-low FPRs. 


<img width="3964" height="1210" alt="workflow" src="https://github.com/user-attachments/assets/ff9dcde3-c711-4b4b-8119-65156516b92e" />



## Installation
```
# download the code
git clone https://github.com/yepengliu/DAWA.git
cd DAWA

# install the required environment 
pip install -r requirements.txt
```

## How to use
### Demo usage: generate a watermarked text given a prompt using Llama-2-13B

```
import torch
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import nltk
# nltk.download('punkt')

from utils import load_model, pre_process
from watermark import Watermark

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# load watermarking model
watermark_model, watermark_tokenizer = load_model("meta-llama/Llama-2-13b-hf")
# load auxiliary model
auxilary_model, auxilary_tokenizer = load_model("meta-llama/Llama-2-7b-hf")

watermark = Watermark(device=device,
                  watermark_tokenizer=watermark_tokenizer,
                  watermark_model=watermark_model,
                  auxilary_tokenizer=auxilary_tokenizer,
                  auxilary_model=auxilary_model,
                  alpha=0.2,
                  top_k=0,   # default 0, not used
                  top_p=1.0,
                  repetition_penalty=1.00,
                  no_repeat_ngram_size=0,
                  max_new_tokens=300,
                  min_new_tokens=200,
                  key=123,
                  T = 1.0,
                  start = 5,
                  )

prompt = " "   # input your prompt here

# generate unwatermarked text
uwm_text = watermark.generate_unwatermarked(prompt)
# generate watermarked text
wm_text = watermark.generate_watermarked(prompt)

# detect unwatermarked text
human_score = watermark.detection(human)

# detect watermarked text
wm_text_score = watermark.detection(wm_text)
```

### Evaluate the detection performance of DAWA on a dataset
```
python generation.py
```

## Citation
If you find this repository useful for your research and applications, please cite our paper:

```
@article{he2024theoretically,
  title={Theoretically Grounded Framework for LLM Watermarking: A Distribution-Adaptive Approach},
  author={He, Haiyun and Liu, Yepeng and Wang, Ziqiao and Mao, Yongyi and Bu, Yuheng},
  journal={arXiv preprint arXiv:2410.02890},
  year={2024}
}
```
