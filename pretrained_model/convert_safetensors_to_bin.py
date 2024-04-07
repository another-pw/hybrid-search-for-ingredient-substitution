

import argparse
import torch

from safetensors.torch import load_file

def convert_safetensors_to_bin(safetensors_path, bin_path):
    torch.save(load_file(safetensors_path), f=bin_path)
    print('saved safetensors to bin')

# python -m pretrained_model.convert_safetensors_to_bin --safetensors_path=pretrained_model/model/bert_output/checkpoint/model.safetensors --bin_path=pretrained_model/model/bert_output/checkpoint/pytorch_model.bin

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--safetensors_path', type=str, required=True)
    parser.add_argument('--bin_path', type=str, required=True)

    args = parser.parse_args()
    safetensors_path = args.safetensors_path
    bin_path = args.bin_path

    convert_safetensors_to_bin(safetensors_path=safetensors_path, bin_path=bin_path)

if __name__ == '__main__':
    main()