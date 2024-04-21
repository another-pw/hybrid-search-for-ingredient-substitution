import json
import torch

from pathlib import Path
from tqdm import tqdm
from transformers import BertTokenizer, AutoModelForMaskedLM

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

    if len(vec.nonzero()) == 1:
        return { cols: weights }
    
    sparse_vector = dict(zip(cols, weights))
    return sparse_vector

def to_dense_vector(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
        last_hidden_state = outputs.hidden_states[-1]
        dense_vector = last_hidden_state[:, 1, :].squeeze()
    
    return dense_vector

def main():
    vocab_file = 'foodbert/data/bert-base-cased-vocab.txt'
    used_ingredients_file = 'foodbert/data/used_ingredients.json'
    sparse_dense_vectors_dict_export_path = 'data/sparse_dense_embeddings.json'

    bert_output_dir = 'pretrained_model/model/bert_output/checkpoint'
    splade_output_dir = 'pretrained_model/model/splade_output/checkpoint'

    bert_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=bert_output_dir)
    splade_model = AutoModelForMaskedLM.from_pretrained(pretrained_model_name_or_path=splade_output_dir)

    with Path(used_ingredients_file).open() as f:
        used_ingredients = json.load(f)
        
    tokenizer = BertTokenizer(
        vocab_file=vocab_file, 
        do_lower_case=False,
        max_len=128,
        never_split=used_ingredients
    )

    sparse_dense_vectors_dict = {}
    for ingredient in tqdm(used_ingredients, desc='generating sparse-dense embeddings'):
        sparse_vector = to_sparse_vector(text=ingredient, tokenizer=tokenizer, model=splade_model)
        dense_vector = to_dense_vector(text=ingredient, tokenizer=tokenizer, model=bert_model)

        sparse_dense_vectors_dict[ingredient] = {
            'sparse_vector': sparse_vector,
            'dense_vector': dense_vector.tolist()
        }

    with open(sparse_dense_vectors_dict_export_path, 'w') as f:
        json.dump(sparse_dense_vectors_dict, f, indent=2)

if __name__ == '__main__':
    main()