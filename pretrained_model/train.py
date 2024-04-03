import json
import argparse

from pathlib import Path
from transformers import BertTokenizer
from transformers import (
    AutoModel,
    TrainingArguments, 
    Trainer,
    LineByLineTextDataset,
    PreTrainedTokenizer,
    DataCollatorForLanguageModeling
)

def get_dataset(tokenizer: PreTrainedTokenizer, file_path: str, block_size: int):
    return LineByLineTextDataset(tokenizer=tokenizer, file_path=file_path, block_size=block_size)

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

    train_dataset_file = 'foodbert/data/sample_train_instructions.txt'
    eval_dataset_file = 'foodbert/data/sample_test_instructions.txt'

    if model_name == 'bert':
        model_id = 'google-bert/bert-base-cased'
    elif model_name == 'splade':
        model_id = 'naver/splade-cocondenser-ensembledistil'

    model = AutoModel.from_pretrained(model_id)

    with Path(used_ingredients_file).open() as f:
        used_ingredients = json.load(f)

    tokenizer = BertTokenizer(
        vocab_file=vocab_file, 
        do_lower_case=False,
        max_len=128,
        never_split=used_ingredients
    )

    model.resize_token_embeddings(len(tokenizer))

    block_size = tokenizer.model_max_length
    train_dataset = get_dataset(tokenizer=tokenizer, file_path=train_dataset_file, block_size=block_size)
    eval_dataset = get_dataset(tokenizer=tokenizer, file_path=eval_dataset_file, block_size=block_size)

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
        train_dataset=train_dataset,
        eval_dataset=eval_dataset
    )

    trainer.train()
    trainer.save_model(output_dir)
    print(f'{model_name}: training finished')

if __name__ == '__main__':
    main()