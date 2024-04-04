# source: https://towardsdatascience.com/build-a-recipe-recommender-chatbot-using-rag-and-hybrid-search-part-i-c4aa07d14dcf
def adjust_vector_weight(sparse_dict, dense_vector, alpha):
    if alpha < 0 or alpha > 1:
        raise ValueError("alpha must be between 0 and 1")
    
    weighted_sparse_dict = {}
    for i in sparse_dict.keys():
        weighted_sparse_dict[i] = sparse_dict[i] * (1 - alpha)  

    weighted_dense_vector = [v * alpha for v in dense_vector]
    return weighted_dense_vector, weighted_sparse_dict

def main():
    pass

if __name__ == '__main__':
    main()