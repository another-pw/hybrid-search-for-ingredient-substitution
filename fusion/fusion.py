import json
from pathlib import Path

from tqdm import tqdm

def reciprocal_rank_fusion(list1, list2):
    # Combine the two lists
    combined_list = list1 + list2

    # Create a dictionary to store the reciprocal ranks of items
    reciprocal_ranks = {}

    # Calculate reciprocal ranks for each item in the combined list
    for sublist in [list1, list2]:
        for i, item in enumerate(sublist):
            rank = i + 1
            reciprocal_rank = 1 / rank
            if item[0] in reciprocal_ranks:
                reciprocal_ranks[item[0]] += reciprocal_rank
            else:
                reciprocal_ranks[item[0]] = reciprocal_rank
    
    for item in reciprocal_ranks.keys():
        reciprocal_ranks[item] /= 2

    # Sort the items based on their combined reciprocal ranks
    merged_list = sorted(reciprocal_ranks.items(), key=lambda x: x[1], reverse=True)

    return merged_list

def main():
    result_1_path = 'data/dense_results.json'
    result_2_path = 'data/sparse_results.json'
    
    with Path(result_1_path).open() as f:
        result_1 = json.load(f)

    with Path(result_2_path).open() as f:
        result_2 = json.load(f)

    with Path('data/cleaned_thai_ingredients_v3.json').open() as f:
        thai_ingredients = json.load(f)
        thai_ingredients_set = set(thai_ingredients)

    top_k = 5
    subtitute_pairs = set()
    for key in tqdm(result_1.keys(), desc='merging results...'):
        if key in result_1 and key in result_2:
            merged_list = reciprocal_rank_fusion(result_1[key], result_2[key])
            for item, score in merged_list[:top_k]:
                if item in thai_ingredients_set:
                    subtitute_pairs.add((key, item))

    with open('data/merge_results.json', 'w') as f:
        json.dump(list(sorted(subtitute_pairs)), f)

def test():
    # Example lists
    list1 = [
        ["crushed_pretzel", 5.371386763537348],
        ["potato", 5.3761498074461365],
        ["walnut", 5.43291996300824],
        ["jack_daniel", 5.448864671071273],
        ["toffee_sauce", 5.5068910613763835],
    ]

    list2 = [
        ["balsamic_reduction", 5.570558907619466],
        ["crushed_graham_cracker", 5.605292616981849],
        ["potato", 5.61060430105472],
        ["toffee_sauce", 5.621923606653165],
        ["cashew", 5.636458786744597],
    ]

    merged_list = reciprocal_rank_fusion(list1, list2)
    print(merged_list)

if __name__ == '__main__':
    main()
    # test()