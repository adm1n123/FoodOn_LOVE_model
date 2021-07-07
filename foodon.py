import pandas as pd
import logging as log
import os
import random
import networkx as nx
from utilities import file_exists, save_pkl, load_pkl
from scoring import Scoring

class FoodOn:

    def __init__(self):
        self.csv_file = 'data/FoodOn/FOODON.csv'
        self.pairs_file = 'data/FoodOn/foodonpairs.txt' # save all pairs (parent, child) of FoodOn to here
        self.use_pairs_file = True  # don't generate again

        self.pd_foodon_pairs = self.generate_pairs()
        self.all_classes, self.all_entities = self.get_classes_and_entities()   # all_entities are instances(not classes)
        self.foodon_graph, self.graph_dict, self.graph_dict_flip = self.generate_graph()

        self.full_ontology_pkl = 'data/FoodOn/full_classes_dict.pkl'    # save full ontology dict to here
        self.candidate_ontology_pkl = 'data/FoodOn/candidate_classes_dict.pkl'  # save ground truth ontology (excluding classes without entities) dict to here
        self.skeleton_and_entities_pkl = 'data/FoodOn/skeleton_candidate_classes_dict.pkl'  # save skeleton ontology dict and entities to populate to here
        self.overwrite_pkl = False   # if True, create and overwrite previous pickle file
        self.num_seeds = 2  # minimum number of labeled data for each class
        self.num_min_extracted_entities = 1 # if total entities are less than num_seeds take this as number of seeds.

    def generate_pairs(self):   # tested
        print('generating child parent pairs')
        if os.path.isfile(self.pairs_file) and self.use_pairs_file:
            print('Using pre-generated pairs file.')
            return pd.read_csv(self.pairs_file, sep='\t')

        # read foodon csv file
        foodon = pd.read_csv(self.csv_file, usecols=['Class ID', 'Parents', 'Preferred Label'])
        temp = foodon[['Class ID', 'Preferred Label']].copy()
        labels = temp.set_index('Class ID')['Preferred Label'].to_dict()
        # take child and parents classes
        child_parents = (foodon[['Class ID', 'Parents']].copy()).rename(columns={'Class ID': 'Child'})
        # split parent classes if there are more than one parent to child.
        pairs = []
        for _, row in child_parents.iterrows():
            parents = str(row['Parents'])
            parent_classes = parents.split('|')
            for parent in parent_classes:
                pairs.append([str(row['Child']), parent])
        # take child and parent class
        foodonDF = pd.DataFrame(pairs, columns=['Child', 'Parent']) # multiple parents are split over rows
        foodonDF = self.filter_ontology(foodonDF, 'http://purl.obolibrary.org/obo/FOODON_00001872') # this class is under progress so don't include it.
        foodonDF = self.get_subtree(foodonDF, 'http://purl.obolibrary.org/obo/FOODON_00001002')     # take subtree of FOODON_00001002 class this is 'foodon product type' because we are working with this subtree only.
        # replace class id with the label for both child and parent and store it in foodonpairs.txt
        for _, row in foodonDF.iterrows():
            row['Child'] = labels[row['Child']]
            row['Parent'] = labels[row['Parent']]   # if error: item not found in dict use if condition and don't label parent leave class id.
        foodonDF.drop_duplicates(inplace=True, ignore_index=True)
        foodonDF.to_csv(self.pairs_file, sep='\t', index=False)

        return foodonDF

    def filter_ontology(self, df, classname):
        # Remove class and its children from the ontology. Works only if the children are leaf nodes. because grandchild will not be removed. but finding subtree after it will not include grandchild
        indexes = df[df['Parent'] == classname].index
        df.drop(indexes, inplace=True)
        indexes = df[df['Child'] == classname].index
        df.drop(indexes, inplace=True)
        return df

    def get_subtree(self, df, root):
        # return the all the classes in subtree of root
        subtreeDF, nextlevelclasses = self.get_level_classes(df, [root])

        while len(nextlevelclasses) > 0:
            level_pairs, nextlevelclasses = self.get_level_classes(df, nextlevelclasses)
            subtreeDF = pd.concat([subtreeDF, level_pairs], ignore_index=True)
        return subtreeDF

    def get_level_classes(self, df, parent_classes):
        # return the parent child pair of all the parent classes
        pairs = []
        non_leaf_children = []
        for parent in parent_classes:
            selected_pairs = df[df['Parent'] == parent]
            for _, row in selected_pairs.iterrows():
                pairs.append([row['Child'], row['Parent']])
                next_level = df[df['Parent'] == row['Child']]
                if not next_level.empty:
                    non_leaf_children.append(row['Child'])

        level_pairs = pd.DataFrame(pairs, columns=['Child', 'Parent'])
        return level_pairs, non_leaf_children

    def get_classes_and_entities(self):
        classes = self.pd_foodon_pairs['Parent'].tolist()   # every non-leaf is a class.
        classes = list(set(classes))
        classes.sort()
        print('Found %d classes.', len(classes))

        child = self.pd_foodon_pairs['Child'].tolist()
        child = list(set(child))
        child.sort()
        entities = [c for c in child if c not in classes]   # child which is also parent is not leaf(instance)
        print('Found %d entities.', len(entities))
        return classes, entities

    def generate_graph(self):
        graph_dict = {class_label: idx for idx, class_label in enumerate(self.all_classes)}     # label to index
        graph_dict_flip = {idx: class_label for idx, class_label in enumerate(self.all_classes)}    # index to label

        G = nx.DiGraph()
        for _, row in self.pd_foodon_pairs.iterrows():
            if row['Parent'] in self.all_classes and row['Child'] in self.all_classes:
                node_from = graph_dict[row['Parent']]
                node_to = graph_dict[row['Child']]
                G.add_edge(node_from, node_to)
        return G, graph_dict, graph_dict_flip

    def get_candidate_classes(self):    # all classes which have at least one entity(instance) is candidate class.
        print('Generating dictionary of candidate classes.')
        if os.path.isfile(self.candidate_ontology_pkl) and not self.overwrite_pkl:
            print('Using pre-generated candidate classes dictionary file: %s', self.candidate_ontology_pkl)
            return load_pkl(self.candidate_ontology_pkl)

        candidate_classes_dict = {}
        for class_label in self.all_classes:
            pd_match = self.pd_foodon_pairs[self.pd_foodon_pairs['Parent'] == class_label]
            children = pd_match['Child'].tolist()   # children could be classes and/or entities(instances)
            children_entities = [c for c in children if c in self.all_entities] # take only child entities not child classes.

            if len(children_entities) > 0:
                node_from = self.graph_dict['foodon product type']  # Root class for our experiments.
                node_to = self.graph_dict[class_label]
                paths = []
                if class_label == 'foodon product type':
                    paths.append(tuple(['foodon product type']))    # path from root to root
                else:
                    for path in nx.all_simple_paths(self.foodon_graph, source=node_from, target=node_to):   # all acyclic path from root to current class.(path includes root & current class). there could be more than one path.
                        translated_path = [self.graph_dict_flip[p] for p in path]   # convert integer value of class to label.
                        paths.append(tuple(translated_path[::-1]))  # reverse the path
                candidate_classes_dict[class_label] = (paths, children_entities)    # for each class save all paths to root and all entities
        print('Found %d candidate classes out of %d all classes.', len(candidate_classes_dict.keys()), len(self.all_classes))
        save_pkl(candidate_classes_dict, self.candidate_ontology_pkl)
        return candidate_classes_dict

    def get_seeded_skeleton(self, candidate_classes_dict):
        print('Generating dictionary of skeleton candidate classes.')
        if file_exists(self.skeleton_and_entities_pkl) and not self.overwrite_pkl:
            print('Using pickled skeleton file: %s', self.skeleton_and_entities_pkl)
            return load_pkl(self.skeleton_and_entities_pkl)

        skeleton_candidate_classes_dict = {}
        candidate_entities = []
        for candidate_class in candidate_classes_dict.keys():   # candidate_classes_dict is [class_lable: (path, entities)]
            entities = candidate_classes_dict[candidate_class][1]   # take entities of class

            if len(entities) <= self.num_seeds: # if there are few entities then take fewer seeds for such classes.
                temp_num_seeds = len(entities) - self.num_min_extracted_entities
                if temp_num_seeds > 0:
                    seeds = random.sample(entities, temp_num_seeds) # randomly sample few entities as labeled data.
                    candidate_entities.extend(list(set(entities) - set(seeds))) # take candidate entities(unlabeled) to be mapped to class
                else:
                    seeds = entities.copy() # take all as seed no entity to populate for this class.
            else:   # if there are sufficient entities
                seeds = random.sample(entities, self.num_seeds)
                candidate_entities.extend(list(set(entities) - set(seeds))) # take candidate entities(unlabeled) to be mapped to class

            skeleton_candidate_classes_dict[candidate_class] = (candidate_classes_dict[candidate_class][0], seeds)  # store path and seeds in skeleton dict

        candidate_entities = list(set(candidate_entities))  # even if an entity belongs to more than one class assume it belongs to one class and we will map it to only one class(in paper).
        candidate_entities.sort()
        print('Found %d candidate entities to populate out of %d all entities.', len(candidate_entities), len(self.all_entities))
        return_value = (skeleton_candidate_classes_dict, candidate_entities)
        save_pkl(return_value, self.skeleton_and_entities_pkl)
        return return_value


    def populate_foodon(self):
        candi_class_dict = self.get_candidate_classes()
        skeleton_candi_class_dict, candi_entities = self.get_seeded_skeleton(candi_class_dict)

        # TODO: run and debug before scoring see everything is fine
        scoring = Scoring(
            skeleton_candi_class_dict,
            candi_entities)

        scoring.run_iteration()

