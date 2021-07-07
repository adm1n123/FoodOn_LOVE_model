

import os
import pickle
import logging as log
from word2vec import WordEmbeddings
from wiki import Wikipedia
from foodon import FoodOn
from utils.utilities import load_pkl



def main():
    config()

    # train_embeddings()

    # create_and_run_model()

    analyse_model()


    return None

def train_embeddings():
    we = WordEmbeddings()
    # we.get_pretrain_vectors()
    wiki = Wikipedia()
    # wiki.parse_wiki()
    # we.train_embeddings()
    return None

def create_and_run_model():
    # creating the graph and seeding the class.
    foodon = FoodOn()

    # mapping the rest entities to class.
    foodon.populate_foodon()
    return None

def analyse_model():
    populated_filepath = 'data/scores/populated.pkl'
    iteration_populated_dict = load_pkl(populated_filepath)

    skeleton_and_entities_pkl = 'data/FoodOn/skeleton_candidate_classes_dict.pkl'
    skeleton_candidate_classes_dict, candidate_entities = load_pkl(skeleton_and_entities_pkl)

    candidate_ontology_pkl = 'data/FoodOn/candidate_classes_dict.pkl'
    candidate_classes_dict = load_pkl(candidate_ontology_pkl)



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