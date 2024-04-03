import argparse
import json
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, AutoModel, AutoModelForMaskedLM

# source: https://towardsdatascience.com/build-a-recipe-recommender-chatbot-using-rag-and-hybrid-search-part-i-c4aa07d14dcf
def to_sparse_vector(text, tokenizer, model):
    tokens = tokenizer(text, return_tensors='pt')
    output = model(**tokens)
    vec = torch.max(
        torch.log(1 + torch.relu(output.logits)) * tokens.attention_mask.unsqueeze(-1), 
        dim=1,
    )[0].squeeze()

    cols = vec.nonzero().squeeze().cpu().tolist()
    weights = vec[cols].cpu().tolist()
    sparse_dict = dict(zip(cols, weights))
    return sparse_dict

def to_dense_vector():
    pass

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)

    args = parser.parse_args()
    model_name_or_path = args.model_name_or_path

    vocab_file = 'foodbert/data/bert-base-cased-vocab.txt'
    used_ingredients_file = 'foodbert/data/used_ingredients.json'

    model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=model_name_or_path)

    with Path(used_ingredients_file).open() as f:
        used_ingredients = json.load(f)
        
    tokenizer = BertTokenizer(
        vocab_file=vocab_file, 
        do_lower_case=False,
        max_len=128,
        never_split=used_ingredients
    )

if __name__ == '__main__':
    main()