import json
import re

from pathlib import Path
from tqdm import tqdm

def replace_words(text, old_word, new_word):
    # This pattern ensures that the word boundaries are respected
    # so only whole words are replaced, and not parts of other words.
    # \b is a word boundary, \W is any non-word character.
    pattern = r'(\b){}(\b)|(\W)'.format(re.escape(old_word))
    
    def replace(match):
        if match.group(0) == old_word:
            return new_word
        else:
            return match.group(0)  # Return non-word characters as is
    
    return re.sub(pattern, replace, text)

def main():
    substitute_pairs_file_path = 'adjust_recipe/data/food2vec_5nn_result_1.json'
    recipes_file_path = 'adjust_recipe/data/new_normalized_recipe_sample.json'
    new_recipes_export_path = 'adjust_recipe/output/new_recipes.json'

    with Path(substitute_pairs_file_path).open() as f:
        substitute_pairs = json.load(f)

    with Path(recipes_file_path).open() as f:
        recipes = json.load(f)
    
    substitutes_dict = {}
    for ingredient, substitute in tqdm(substitute_pairs, desc='aggregating pairs'):
        ingredient = ingredient.replace(' ', '_')
        substitute = substitute.replace(' ', '_')

        if ingredient not in substitutes_dict:
            substitutes_dict[ingredient] = [substitute]
        else:
            substitutes_dict[ingredient].append(substitute)

    for recipe in tqdm(recipes, desc='replacing ingredients'):
        substitutions = []
        for ingredient in recipe['ingredients']:
            ingredient = ingredient['text']
            if ingredient in substitutes_dict:
                substitutions.append({ ingredient: substitutes_dict[ingredient] })
        
        recipe['substitutions'] = substitutions
    
        for i, instruction in enumerate(recipe['instructions']):
            for substitution in substitutions:
                target = list(substitution.keys())[0]
                replacement = f'{target} ({' / '.join(substitution[target])})'

                recipe['instructions'][i]['text'] = replace_words(instruction['text'], target, replacement)
        
    with Path(new_recipes_export_path).open('w') as f:
        json.dump(recipes, f, indent=2)

    print('replaced ingredients in recipes with its substitutions')

if __name__ == '__main__':
    main()