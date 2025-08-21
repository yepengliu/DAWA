import os
import torch
import json
import argparse
from datasets import load_dataset
import pandas as pd
from tqdm import tqdm
import nltk
# nltk.download('punkt')

from utils import load_model, pre_process
from watermark import Watermark



def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # load watermarking model
    watermark_model, watermark_tokenizer = load_model(args.watermark_model)
    # load auxilary model
    auxilary_model, auxilary_tokenizer = load_model(args.auxilary_model)
    # load test dataset.
    dataset = load_dataset("allenai/c4", "realnewslike", streaming=True)
    dataset = pre_process(dataset, min_length=args.min_new_tokens, data_size=500)

    # print(watermark_tokenizer.vocab_size)
    # print(auxilary_tokenizer.vocab_size)

    watermark = Watermark(device=device,
                      watermark_tokenizer=watermark_tokenizer,
                      watermark_model=watermark_model,
                      auxilary_tokenizer=auxilary_tokenizer,
                      auxilary_model=auxilary_model,
                      alpha=args.alpha,
                      top_k=0,   # default 0, not used
                      top_p=1.0,
                      repetition_penalty=args.repetition_penalty,
                      no_repeat_ngram_size=args.no_repeat_ngram_size,
                      max_new_tokens=args.max_new_tokens,
                      min_new_tokens=args.min_new_tokens,
                      key=123,
                      T = args.T,
                      start = args.start,
                      )

    rows = []
    for i in tqdm(range(200)):
        human = dataset[i]['text0']
        prompt = ' '.join(nltk.sent_tokenize(dataset[i]['text'])[:2])   # first two sentences


        uwm_text = watermark.generate_unwatermarked(prompt)
        wm_text = watermark.generate_watermarked(prompt)

        human_score = watermark.detection(human)
        wm_text_score = watermark.detection(wm_text)

        row = {
            'text_id': i,
            'prompt': prompt,
            'human': human,
            'uwm_text': uwm_text,
            'wm_text': wm_text,
            'human_score': human_score,
            'wm_text_score': wm_text_score,
        }
        rows.append(row)
        df = pd.DataFrame(rows)
        df.to_csv(f'{args.output_dir}', index=False)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate watermarked text!")
    parser.add_argument('--watermark_model', default='meta-llama/Llama-2-13b-hf', type=str, \
                        help='Main model, path to pretrained model or model identifier from huggingface.co/models.')
    parser.add_argument('--auxilary_model', default='meta-llama/Llama-2-7b-hf', type=str, \
                        help='Auxilary model')
    parser.add_argument('--alpha', default=0.2, type=float, \
                        help='eta in the paper.')
    parser.add_argument('--T', default=1.0, type=float, \
                        help='Temperature of sampling.')
    parser.add_argument('--repetition_penalty', default=1.0, type=float, \
                        help='Repetition penalty.')
    parser.add_argument('--no_repeat_ngram_size', default=0, type=int, \
                        help='No repeat ngram size.')
    parser.add_argument('--start', default=5, type=int, \
                        help='Start index.')
    parser.add_argument('--max_new_tokens', default=300, type=int, \
                        help='Max tokens.')
    parser.add_argument('--min_new_tokens', default=200, type=int, \
                        help='Min tokens.')
    parser.add_argument('--output_dir', default='output/output.csv', type=str, \
                        help='Output directory.')
    
    
    args = parser.parse_args()
    main(args)



