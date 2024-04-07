import json
import argparse

from pathlib import Path
from transformers import BertTokenizer
from transformers import (
    AutoModelForMaskedLM,
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling
)
from datasets import load_dataset

def get_dataset(path, data_files):
    dataset = load_dataset(path=path, data_files=data_files, cache_dir=path)
    return dataset

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str, required=True)
    parser.add_argument('--epochs', type=int, required=True)

    args = parser.parse_args()
    model_name = args.model_name
    num_train_epochs = args.epochs

    output_dir = f'pretrained_model/model/{model_name}_output/checkpoint'
    vocab_file = 'foodbert/data/bert-base-cased-vocab.txt'
    used_ingredients_file = 'foodbert/data/used_ingredients.json'

    dataset_path = 'foodbert/data'
    train_dataset_file = 'sample_train_instructions.txt'
    eval_dataset_file = 'sample_test_instructions.txt'

    if model_name == 'bert':
        model_id = 'google-bert/bert-base-cased'
    elif model_name == 'splade':
        model_id = 'naver/splade-cocondenser-ensembledistil'

    model = AutoModelForMaskedLM.from_pretrained(model_id)

    with Path(used_ingredients_file).open() as f:
        used_ingredients = json.load(f)

    tokenizer = BertTokenizer(
        vocab_file=vocab_file, 
        do_lower_case=False,
        max_len=128,
        never_split=used_ingredients
    )

    tokenize_function = lambda examples : tokenizer(
        examples['text'], 
        padding='max_length',
        truncation=True
    )
    model.resize_token_embeddings(len(tokenizer))

    train_dataset = get_dataset(path=dataset_path, data_files=train_dataset_file)
    tokenized_train_datasets = train_dataset.map(tokenize_function, batched=True)
    eval_dataset = get_dataset(path=dataset_path, data_files=eval_dataset_file)
    tokenized_eval_datasets = eval_dataset.map(tokenize_function, batched=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        prediction_loss_only=True
    )

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=True)
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_train_datasets['train'],
        eval_dataset=tokenized_eval_datasets['train'],
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f'{model_name}: training finished')

if __name__ == '__main__':
    main()