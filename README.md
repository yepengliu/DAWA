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
