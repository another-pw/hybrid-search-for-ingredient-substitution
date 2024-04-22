import json

from pathlib import Path
from gensim.models import Word2Vec
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

def adjust_vector_weight(vector, alpha):
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    
    return [v * alpha for v in vector]

def main():
    alpha = 1
    n_neighbors = 10

    used_ingredients_file = 'foodbert/data/used_ingredients.json'
    word2vec_results_export_path = 'data/word2vec_results.json'
    
    if not Path(word2vec_results_export_path).exists():
        
        with Path(used_ingredients_file).open() as f:
            used_ingredients = json.load(f)

        model = Word2Vec.load('word2vec/model/model.bin')

        ingredient_embeddings = []
        ingredient_names = []
        for ingredient in tqdm(used_ingredients, desc='generating word2vec embeddings...'):
            try:
                embedding = model.wv[ingredient]
                ingredient_embeddings.append(embedding)  # get numpy vector of a word
                ingredient_names.append(ingredient)
            except:
                pass
        
        print(f'{len(ingredient_names)}, {len(ingredient_embeddings)}')
        
        neighbors = NearestNeighbors(n_neighbors=n_neighbors, n_jobs=-1)
        neighbors.fit(ingredient_embeddings)

        results = {}
        for i in tqdm(range(len(ingredient_embeddings)), desc='finding neighbors...'):
            weighted_vector = adjust_vector_weight(ingredient_embeddings[i], alpha)
            distance, indices = neighbors.kneighbors(
                [weighted_vector], 
                n_neighbors + 1,
                return_distance=True
            )

            substitutes_and_scores = []
            for j, idx in enumerate(indices[0]):
                if ingredient_names[i] != ingredient_names[idx]:
                    substitutes_and_scores.append(((ingredient_names[idx], distance[0][j])))
            
            results[ingredient_names[i]] = substitutes_and_scores[:n_neighbors]

        with Path(word2vec_results_export_path).open('w') as f:
            json.dump(results, f, indent=2)
        
        print('saved word2vec vector search results')
    else:
        print('word2vec_results.json found')

if __name__ == '__main__':
    main()