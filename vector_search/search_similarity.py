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

def sparse_dict_to_vector(sparse_dict, max_length):
    sparse_vector = np.zeros(max_length)
    
    for index, value in sparse_dict.items():
        sparse_vector[int(index)] = value
    
    return sparse_vector

def fusion():
    pass

def main():
    metric = 'cosine'
    n_neighbors = 10
    alpha = 0.5

    used_ingredients_file = 'foodbert/data/used_ingredients.json'
    sparse_dense_vectors_dict_file = 'data/sparse_dense_embeddings.json'

    dense_results_export_path = 'data/dense_results.json'
    sparse_results_export_path = 'data/sparse_results.json'
    substitute_pairs_export_path = 'data/substitute_pairs.json'

    with Path(used_ingredients_file).open() as f:
        used_ingredients = json.load(f)

    with Path(sparse_dense_vectors_dict_file).open() as f:
        sparse_dense_vectors_dict = json.load(f)

    sparse_vectors = []
    dense_vectors = []
    ingredient_names = []

    max_length = 0
    for ingredient in tqdm(used_ingredients, desc='finding max length of sparse vector'):
        sparse_dict = sparse_dense_vectors_dict[ingredient]['sparse_vector']
        max_length = max(
            max_length,
            max([int(i) for i in sparse_dict.keys()]) + 1
        )
    
    print(f'sparse vector length: {max_length}')

    for ingredient in tqdm(used_ingredients, desc='extracting embeddings from json file'):
        sparse_vectors.append(sparse_dict_to_vector(
                sparse_dict=sparse_dense_vectors_dict[ingredient]['sparse_vector'],
                max_length=max_length
            )
        )
        dense_vectors.append(sparse_dense_vectors_dict[ingredient]['dense_vector'])
        ingredient_names.append(ingredient)

    dense_neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
    sparse_neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)

    dense_neighbors.fit(dense_vectors)
    sparse_neighbors.fit(sparse_vectors)

    if not Path(sparse_results_export_path).exists():
        print('sparse_results.json not found, generating results list')
        sparse_results = {}
        for i in tqdm(range(len(used_ingredients)), desc='generating results (sparse)'):
            sparse_dict = sparse_dense_vectors_dict[used_ingredients[i]]['sparse_vector']
            weigthed_sparse_dict = adjust_sparse_vector_weight(sparse_dict=sparse_dict, alpha=alpha)
            weigthed_sparse_vector = sparse_dict_to_vector(sparse_dict=weigthed_sparse_dict, max_length=max_length)
            sparse_distance, sparse_indices = sparse_neighbors.kneighbors(
                [weigthed_sparse_vector], 
                n_neighbors + 1, 
                return_distance=True
            )

            substitutes_and_scores = []
            for j, si in enumerate(sparse_indices[0]):
                if ingredient_names[i] != ingredient_names[si]:
                    substitutes_and_scores.append(((ingredient_names[si], sparse_distance[0][j])))

            sparse_results[ingredient_names[i]] = substitutes_and_scores[:n_neighbors]

        with Path(sparse_results_export_path).open('w') as f:
            json.dump(sparse_results, f, indent=2)

        print('saved sparse vector search results')
    else:
        print('sparse_results.json found, loading results')
        with Path(sparse_results_export_path).open() as f:
            sparse_results = list(json.load(f))

    if not Path(dense_results_export_path).exists():
        print('dense_results.json not found, generating results list')
        dense_results = {}
        for i in tqdm(range(len(used_ingredients)), desc='generating results (dense)'):
            weighted_dense_vector = adjust_dense_vector_weight(dense_vector=dense_vectors[i], alpha=alpha)
            dense_distance, dense_indices = dense_neighbors.kneighbors(
                [weighted_dense_vector], 
                n_neighbors + 1, 
                return_distance=True
            )

            substitutes_and_scores = []
            for j, di in enumerate(dense_indices[0]):
                if ingredient_names[i] != ingredient_names[di]:
                    substitutes_and_scores.append(((ingredient_names[di], dense_distance[0][j])))

            dense_results[ingredient_names[i]] = substitutes_and_scores[:n_neighbors]
        
        with Path(dense_results_export_path).open('w') as f:
            json.dump(dense_results, f, indent=2)
    
        print('saved dense vector search results')
    else:
        print('dense_results.json found, loading results')
        with Path(dense_results_export_path).open() as f:
            dense_results = list(json.load(f))
    
    # subtitute_pairs = set()
    # with Path(substitute_pairs_export_path).open('w') as f:
    #     json.dump(list(sorted(subtitute_pairs)), f)

    print('finished')

if __name__ == '__main__':
    main()