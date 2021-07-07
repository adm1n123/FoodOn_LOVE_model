




from gensim.models import KeyedVectors




def test():

    model = KeyedVectors.load_word2vec_format('data/model/word2vec_trained.txt')
    print(model.most_similar(positive=['king', 'woman'], negative=['man']))
    print(model.most_similar(positive=['apple']))
    print(model.most_similar(positive=['fruit']))
















if __name__ == '__main__':
    test()