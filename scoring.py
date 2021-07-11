# standard libraries
import logging as log
import os
import sys
from time import time
import multiprocessing
import itertools
import math
import random

# third party libraries
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
import textdistance
import nltk

# local imports
from fdc_preprocess import FDCPreprocess
from utils.utilities import file_exists, save_pkl, load_pkl


class Scoring:
    def __init__(self, candidate_classes_info, candidate_entities):

        # save arguments
        self.candidate_classes_info = candidate_classes_info    # It is skeleton_candidate_class_info.
        self.candidate_entities = candidate_entities

        self.use_score_file = False # run scoring (parent/sibling) again
        self.use_populated_file = False # run method again.

        # parse config file
        self.alpha = 0.8
        self.num_mapping_per_iteration = 10000
        self.initial_siblings_scores = 'data/scores/siblings_scores.csv'
        self.initial_parents_scores = 'data/scores/parents_scores.csv'
        self.pairs_filepath = 'data/scores/pairs.pkl'
        self.populated_filepath = 'data/scores/populated.pkl'
        # self.preprocess_config_filepath = None # assign object of that class for config values.
        self.similarity_method = 'we_cos'   # method to use to find similarity between labels (we_cos | we_euc | hamming | jaccard | lcsseq | random)

        print('alpha: %f', self.alpha)
        print('num_mapping_per_iteration: %d', self.num_mapping_per_iteration)
        print('initial_siblings_scores: %s', self.initial_siblings_scores)
        print('initial_parents_scores: %s', self.initial_parents_scores)
        print('pairs_filepath: %s', self.pairs_filepath)
        print('populated_filepath: %s', self.populated_filepath)
        print('\n### similarity_method: %s ###\n', self.similarity_method)

        self.fpm = FDCPreprocess()
        self.num_candidate_classes = len(self.candidate_classes_info)
        self.num_candidate_entities = len(self.candidate_entities)
        print('Number of candidate classes: %d', self.num_candidate_classes)
        print('Number of candidate entities: %d', self.num_candidate_entities)

        # extract the seeded entities to make complete list of entities
        seed_entities = self._unpack_sublist([x[1] for _, x in self.candidate_classes_info.items()])    # x[0] is path, x[1] is seeds.
        self.all_entity_labels = list(set(self.candidate_entities + seed_entities))

        self.candidate_classes_label = list(self.candidate_classes_info.keys()) # all labels of candidate class

        # complete list of class labels
        other_classes = self._unpack_sublist([x[0] for _, x in self.candidate_classes_info.items()], depth=2)   # all the classes which are in any path from root to candidate class. It may contain non-candidate class which is in path.
        self.all_class_labels = list(set(self.candidate_classes_label + other_classes)) # all classes in path from root to candidate class. including root and candidate class.

        # calculate embedding lookup table for class / entity labels
        if 'we_' in self.similarity_method:
            self.keyed_vectors = KeyedVectors.load_word2vec_format('data/model/word2vec_trained.txt')
            self.pd_class_label_embeddings = self._calculate_label_embeddings(self.all_class_labels)
            self.pd_entity_label_embeddings = self._calculate_label_embeddings(self.all_entity_labels)

        # do initial calculation of the scores
        self.pd_siblings_scores, self.pd_parents_scores = self._calculate_initial_scores()

    @staticmethod
    def _unpack_sublist(input_list, depth=1):   # flatten the list till given depth.
        for i in range(depth):
            input_list = [item for sublist in input_list for item in sublist]

        return list(set(input_list))

    @staticmethod
    def _cosine_similarity(array1, array2):
        with np.errstate(all='ignore'):
            similarity = np.dot(array1, array2) / (np.linalg.norm(array1) * np.linalg.norm(array2))

        if np.isnan(similarity):
            similarity = -1

        return similarity

    @staticmethod
    def _euclidean_similarity(array1, array2):
        distance = np.linalg.norm(array1-array2)
        if distance == 0:
            return np.inf
        else:
            return (1 / distance)

    def _caculate_embeddings(self, label):  # for a label take weighted average of word vectors.
        label_embedding = 0
        num_found_words = 0

        for word, pos in nltk.pos_tag(label.split(' ')):
            # if word in ['food', 'product']:
            #     continue
            try:
                word_embedding = self.keyed_vectors.get_vector(word)
            except KeyError:
                pass
            else:
                if pos == 'NN': # take NN/NNS if word is noun then give higher weightage in averaging by increasing the vector magnitude.
                    multiplier = 1.15
                else:
                    multiplier = 1
                label_embedding += (multiplier * word_embedding)    # increase vector magnitude if noun.
                num_found_words += 1

        if num_found_words == 0:
            return np.zeros(300)
        else:
            return label_embedding / num_found_words

    def _calculate_label_embeddings(self, index_list):  # make label an index.
        pd_label_embeddings = pd.DataFrame(index=index_list, columns=['preprocessed', 'vector'])
        pd_label_embeddings['preprocessed'] = self.fpm.preprocess_columns(pd_label_embeddings.index.to_series(), load_phrase_model=True)

        # some preprocessed columns are empty due to lemmatiazation, fill it up with original
        empty_index = (pd_label_embeddings['preprocessed'] == '')
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.index.to_series()[empty_index]
        pd_label_embeddings.loc[empty_index, 'preprocessed'] = pd_label_embeddings.loc[empty_index, 'preprocessed'].apply(lambda x: x.lower())  # at least use lowercase as preprocessing.

        pd_label_embeddings['vector'] = pd_label_embeddings['preprocessed'].apply(self._caculate_embeddings)

        return pd_label_embeddings

    def _calculate_siblings_score(self, pair):  # similarity of entity with average of sibling vectors.
        class_label, entity_label = pair[0], pair[1]
        siblings = self.candidate_classes_info[class_label][1]

        if self.similarity_method == 'we_cos':
            num_nonzero_siblings = 0
            siblings_embedding = 0

            for sibling in siblings:
                sibling_embedding = self.pd_entity_label_embeddings.loc[sibling, 'vector']

                if np.count_nonzero(sibling_embedding): # when there are no words in label zero vector is its embedding. so proceed only if it is non-zero.
                    siblings_embedding += sibling_embedding
                    num_nonzero_siblings += 1

            if num_nonzero_siblings == 0:
                return 0

            siblings_embedding /= num_nonzero_siblings
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']

            score = self._cosine_similarity(siblings_embedding, entity_embeddings)
        elif self.similarity_method == 'we_euc':
            num_nonzero_siblings = 0
            siblings_embedding = 0

            for sibling in siblings:
                sibling_embedding = self.pd_entity_label_embeddings.loc[sibling, 'vector']

                if np.count_nonzero(sibling_embedding):
                    siblings_embedding += sibling_embedding
                    num_nonzero_siblings += 1

            if num_nonzero_siblings == 0:
                return 0

            siblings_embedding /= num_nonzero_siblings
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']

            score = self._euclidean_similarity(siblings_embedding, entity_embeddings)
        elif self.similarity_method == 'hamming':
            score = 0
            for sibling in siblings:
                score += textdistance.hamming.normalized_similarity(sibling, entity_label)
            score /= len(siblings)
        elif self.similarity_method == 'jaccard':
            score = 0
            for sibling in siblings:
                score += textdistance.jaccard.normalized_similarity(sibling, entity_label)
            score /= len(siblings)
        elif self.similarity_method == 'lcsseq':
            score = 0
            for sibling in siblings:
                score += textdistance.lcsseq.normalized_similarity(sibling, entity_label)
            score /= len(siblings)
        elif self.similarity_method == 'random':
            score = random.uniform(0, 1)
        else:
            raise ValueError('Invalid similarity method: %s', self.similarity_method)

        return score

    def _calculate_siblings_score2(self, pair):  # similarity of entity with average of sibling vectors.
        class_label, entity_label = pair[0], pair[1]
        siblings = self.candidate_classes_info[class_label][1]

        if self.similarity_method == 'we_cos':
            num_nonzero_siblings = 0
            siblings_embedding = 0

            for sibling in siblings:
                sibling_embedding = self.pd_entity_label_embeddings.loc[sibling, 'vector']
                if np.count_nonzero(sibling_embedding): # when there are no words in label zero vector is its embedding. so proceed only if it is non-zero.
                    sibling_embedding = sibling_embedding / np.linalg.norm(sibling_embedding)
                    siblings_embedding += sibling_embedding
                    num_nonzero_siblings += 1

            if num_nonzero_siblings == 0:
                return 0

            siblings_embedding /= num_nonzero_siblings
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']

            score = self._cosine_similarity(siblings_embedding, entity_embeddings)
            return score

    def _calculate_parents_score(self, pair):   # cosine similarity of class label vector and entity label vector.
        class_label, entity_label = pair[0], pair[1]

        if self.similarity_method == 'we_cos':
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']
            class_embeddings = self.pd_class_label_embeddings.loc[class_label, 'vector']
            score = self._cosine_similarity(class_embeddings, entity_embeddings)
        elif self.similarity_method == 'we_euc':
            entity_embeddings = self.pd_entity_label_embeddings.loc[entity_label, 'vector']
            class_embeddings = self.pd_class_label_embeddings.loc[class_label, 'vector']
            score = self._euclidean_similarity(class_embeddings, entity_embeddings)
        elif self.similarity_method == 'hamming':
            score = textdistance.hamming.normalized_similarity(class_label, entity_label)
        elif self.similarity_method == 'jaccard':
            score = textdistance.jaccard.normalized_similarity(class_label, entity_label)
        elif self.similarity_method == 'lcsseq':
            score = textdistance.lcsseq.normalized_similarity(class_label, entity_label)
        elif self.similarity_method == 'random':
            score = random.uniform(0, 1)
        else:
            raise ValueError('Invalid similarity method: %s', self.similarity_method)

        return score

    def _calculate_initial_scores(self):
        # calculate siblings score
        if file_exists(self.initial_siblings_scores) and self.use_score_file:
            print('Pre-calculated siblings scores found.')
            pd_siblings_scores = pd.read_csv(self.initial_siblings_scores, index_col=0)
        else:
            entity_class_pairs = list(itertools.product(self.candidate_classes_label, self.candidate_entities)) # for each class one by one all entities are taken
            print('Calculating siblings score...')
            t1 = time()
            with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
                results = p.map(self._calculate_siblings_score, entity_class_pairs)
            t2 = time()
            print('Elapsed time for calculating siblings score: %.2f minutes', (t2-t1)/60)
            results = np.array(results).reshape(self.num_candidate_classes, self.num_candidate_entities)    # after reshape first row is similarity of 1st class with all entities. since pairs were generated in this manner.
            pd_siblings_scores = pd.DataFrame(results, index=self.candidate_classes_label, columns=self.candidate_entities)
            pd_siblings_scores.to_csv(self.initial_siblings_scores)

        # calculate parents score
        if file_exists(self.initial_parents_scores) and self.use_score_file:
            print('Pre-calculated parents scores found.')
            pd_parents_scores = pd.read_csv(self.initial_parents_scores, index_col=0)
        else:
            entity_class_pairs = list(itertools.product(self.candidate_classes_label, self.candidate_entities))
            print('Calculating parents score...')
            t1 = time()
            with multiprocessing.Pool(processes=16, maxtasksperchild=1) as p:
                results = p.map(self._calculate_parents_score, entity_class_pairs)
            t2 = time()
            print('Elapsed time for calculating parents score: %.2f minutes', (t2-t1)/60)
            results = np.array(results).reshape(self.num_candidate_classes, self.num_candidate_entities)
            pd_parents_scores = pd.DataFrame(results, index=self.candidate_classes_label, columns=self.candidate_entities)
            pd_parents_scores.to_csv(self.initial_parents_scores)
        return pd_siblings_scores, pd_parents_scores

    def run_iteration(self):
        print('Running iteration')
        if file_exists(self.pairs_filepath) and file_exists(self.populated_filepath) and self.use_populated_file:
            print('Pre-calculated iterations found.')
            iteration_pairs = load_pkl(self.pairs_filepath)
            iteration_populated_dict = load_pkl(self.populated_filepath)
            return iteration_pairs, iteration_populated_dict

        num_iterations = math.floor(self.num_candidate_entities / self.num_mapping_per_iteration)
        iteration_pairs = {}
        iteration_populated_dict = {}

        iteration = 0
        while len(self.candidate_entities) > 0:
            print('Updating scores. Iteration: %d/%d', iteration, num_iterations)
            t1 = time()

            # calculate score
            pd_scores = self.alpha*self.pd_siblings_scores + (1-self.alpha)*self.pd_parents_scores

            # find top N unique entities with highest score
            num_scores = pd_scores.shape[0] * pd_scores.shape[1]    # no. of rows * no. of cols
            pd_top_scores = pd_scores.stack().nlargest(num_scores).reset_index()    # stack(): for each row index stack its cols, sort and reset_index since candi_class_label was index.
            pd_top_scores.columns = ['candidate class', 'candidate entity', 'score']    # candidate entity is stacked column.
            pd_top_scores.drop_duplicates(subset='candidate entity', inplace=True)  # best score(matching class) of each entity is taken (two or more entities many have best score for same class)

            print('Top scores: \n%s', str(pd_top_scores.head()))

            top_n_scores = list(zip(pd_top_scores['candidate class'], pd_top_scores['candidate entity']))
            top_n_scores = top_n_scores[0:self.num_mapping_per_iteration]   # map only specified no. of largest scores.

            # populate skeleton using selected entity
            for (candidate_class, candidate_entity) in top_n_scores:
                self.candidate_classes_info[candidate_class][1].append(candidate_entity)

            # save progress
            iteration_pairs[iteration] = top_n_scores.copy()
            iteration_populated_dict[iteration] = self.candidate_classes_info.copy()    # copy current populated skeleton.

            if len(self.candidate_entities) <= self.num_mapping_per_iteration:
                break

            classes_to_update = list(set([x[0] for x in top_n_scores]))
            entities_to_remove = list(set([x[1] for x in top_n_scores]))

            # remove selected entities from candidate entities and scores
            self.candidate_entities = list(set(self.candidate_entities) - set(entities_to_remove))
            self.pd_siblings_scores = self.pd_siblings_scores.drop(labels=entities_to_remove, axis=1)   # remove mapped entities cols.
            self.pd_parents_scores = self.pd_parents_scores.drop(labels=entities_to_remove, axis=1)

            # if alpha is 0, no need to update siblings score.
            if self.alpha == 0.0:
                print('Skipping siblings score update since alpha is 0.')
            else:
                # update siblings score. parent score is same since its label vector.
                entity_class_pairs = list(itertools.product(classes_to_update, self.candidate_entities))    # no. of siblings update for classes update score.

                results = []
                for pair in entity_class_pairs:
                    results.append(self._calculate_siblings_score(pair))

                results = np.array(results).reshape(
                    len(classes_to_update), len(self.candidate_entities))

                pd_siblings_to_update = pd.DataFrame(results, index=classes_to_update, columns=self.candidate_entities)
                self.pd_siblings_scores.update(pd_siblings_to_update)   # corresponding class entity pairs are updated.

            t2 = time()
            print('Elapsed time for updating scores: %.2f minutes', (t2-t1)/60)

            iteration += 1

        save_pkl(iteration_pairs, self.pairs_filepath)
        save_pkl(iteration_populated_dict, self.populated_filepath)

        return iteration_pairs, iteration_populated_dict
