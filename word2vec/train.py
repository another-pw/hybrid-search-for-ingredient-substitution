from gensim.models import Word2Vec

def main():

    # training data
    with open('foodbert/data/train_instructions.txt', 'r') as f:
        sentences = f.read().splitlines()
        sentences = [[word for word in s.replace('.', ' ').replace(',', ' ').replace('!', ' ').replace('?', ' ').split(' ') if word != ''] for s in sentences]

    print("finished preprocessing sentences")
    # train model
    model = Word2Vec(sentences, min_count=10)  # at least 10 occurances
    # summarize the loaded model
    print(model)
    # save model
    model.save('word2vec/model/model.bin')

if __name__ == '__main__':
    main()