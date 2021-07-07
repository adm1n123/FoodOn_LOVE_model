

import os
import pickle
import logging as log
from word2vec import WordEmbeddings
from wiki import Wikipedia
from foodon import FoodOn


def main():
    config()
    # 1. get corpus and train vectors.

    we = WordEmbeddings()
    # we.get_pretrain_vectors()

    foodon = FoodOn()

    wiki = Wikipedia()
    # wiki.parse_wiki()

    # we.train_embeddings()

    foodon.populate_foodon()






    return None

def train_embeddings():

    return None




def config():
    # create logger
    logger = log.getLogger()
    # logger.setLevel(log_level)

    # create formatter
    formatter = log.Formatter('%(asctime)s %(levelname)s %(filename)s: %(message)s')

    # create and set console handler
    stream_handler = log.StreamHandler()
    # stream_handler.setLevel(log_level)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)


if __name__ == '__main__':
    main()