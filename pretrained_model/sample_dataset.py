import random
from tqdm import tqdm

def read_random_lines(file_path, n):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()
        random.shuffle(lines)
        selected_lines = lines[:n]

    return selected_lines

def save_to_file(lines, output_file):
    with open(output_file, 'w', encoding='utf-8') as file:
        for line in tqdm(lines, desc='writing file'):
            file.write(line)

def do_sample(input_file, output_file, num_sample):
    selected_data = read_random_lines(input_file, num_sample)

    save_to_file(selected_data, output_file)

def main():
    num_train_sample = 1000
    test_size = 20
    num_test_sample = round((test_size / 100) * num_train_sample)

    train_input_file = 'foodbert/data/train_instructions.txt' 
    sample_train_output_file = 'foodbert/data/sample_train_instructions.txt'
    test_input_file = 'foodbert/data/test_instructions.txt' 
    sample_test_output_file = 'foodbert/data/sample_test_instructions.txt'

    do_sample(train_input_file, sample_train_output_file, num_train_sample)
    do_sample(test_input_file, sample_test_output_file, num_test_sample)
    print('sampling finished')

if __name__ == '__main__':
    main()