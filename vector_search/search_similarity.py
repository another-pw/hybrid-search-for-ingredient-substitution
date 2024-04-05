import json
import numpy as np

from pathlib import Path
from tqdm import tqdm
from sklearn.neighbors import NearestNeighbors

# source: https://towardsdatascience.com/build-a-recipe-recommender-chatbot-using-rag-and-hybrid-search-part-i-c4aa07d14dcf
def adjust_sparse_vector_weight(sparse_dict, alpha):
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    
    weighted_sparse_dict = {}
    for i in sparse_dict.keys():
        weighted_sparse_dict[i] = sparse_dict[i] * (1 - alpha)  

    return weighted_sparse_dict

# source: https://towardsdatascience.com/build-a-recipe-recommender-chatbot-using-rag-and-hybrid-search-part-i-c4aa07d14dcf
def adjust_dense_vector_weight(dense_vector, alpha):
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    
    return [v * alpha for v in dense_vector]

def sparse_dict_to_vector(sparse_dict):
    max_index = max([int(i) for i in sparse_dict.keys()]) + 1
    sparse_vector = np.zeros(max_index)
    
    for index, value in sparse_dict.items():
        sparse_vector[int(index)] = value
    
    return sparse_vector

def fusion():
    pass

def main():
    n_neighbors = 5
    alpha = 1

    used_ingredients_file = 'foodbert/data/used_ingredients.json'
    sparse_dense_vectors_dict_file = 'data/sparse_dense_embeddings.json'
    substitute_pairs_export_path = 'data/substitute_pairs.json'

    with Path(used_ingredients_file).open() as f:
        used_ingredients = json.load(f)

    with Path(sparse_dense_vectors_dict_file).open() as f:
        sparse_dense_vectors_dict = json.load(f)

    sparse_vectors = []
    dense_vectors = []
    ingredient_names = []
    for ingredient in tqdm(used_ingredients, desc='extracting embeddings from json file'):
        sparse_vectors.append(sparse_dict_to_vector(
                sparse_dense_vectors_dict[ingredient]['sparse_vector']
            )
        )
        dense_vectors.append(sparse_dense_vectors_dict[ingredient]['dense_vector'])
        ingredient_names.append(ingredient)

    dense_neighbors = NearestNeighbors(n_neighbors=n_neighbors)
    sprase_neighbors = NearestNeighbors(n_neighbors=n_neighbors)

    dense_neighbors.fit(dense_vectors)
    sprase_neighbors.fit(sparse_vectors)

    subtitute_pairs = set()
    for i in tqdm(range(len(used_ingredients)), desc='matching ingredients'):
        dense_distance, dense_indices = dense_neighbors.kneighbors(
            [adjust_dense_vector_weight(dense_vector=dense_vectors[i], alpha=alpha)], 
            n_neighbors + 1, 
            return_distance=True
        )

        # implement this function
        fusion()

        for di in dense_indices[0]:
            if ingredient_names[i] != ingredient_names[di]:
                subtitute_pairs.add((ingredient_names[i], ingredient_names[di]))

    with Path(substitute_pairs_export_path).open('w') as f:
        json.dump(list(sorted(subtitute_pairs)), f)

if __name__ == '__main__':
    main()