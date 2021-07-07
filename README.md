1.
Install packages
gensim
numpy
pandas
matplotlib
scikit-learn
scipy
argparse
configparser
cython
pattern   # done "Downloading mysqlclient-2.0.3-cp38-cp38-win_amd64.whl" this was installed during pattern but "sudo apt-get install libmysqlclient-dev" this was not installed mentioned in prerequisites 2a.   may be it is already included in mysqlclient verify.
wikipedia
networkx
textdistance


2.
run main.py


similarity of entity with all sibling is taken and also similarity with class label is taken 
then score = alpha * sibling + (1-alpha) * class label


TODO:
wiki corpus has only 4349 sentences to train collect more (try firing failed queries again)

get_seeded_skeleton() this method is generatig skeleton graph and some entities are taken as seed and some are kept
as candidate entites(which has to be populated/mapped to class).
there are some entites which are in more than one class but in this method only unique entites are taken i.e. author
is assuming a entity belongs to one class only(in paper) 
improvement could be store the count of classes entity belongs to and then during population assign this entity to 
top k classes(by matching).
If an entity is belonging to more than one class then it might be ambiguous so assigning it to more than one class
migh inccur more loss. 
Try if entity belongs to more than one class ignore that or keep those entities in seed(labeled) and populate entity
belonging to one class.


change similarity_method=random


we are not using phrases. all the labels are space separated and average of word vector is taken as label vector.
foodonpairs.txt columns are not preprocessed and phrases generation is also skipped.


calculate_parents_score  here class label is used for similarity calculation between entity and class(not sibling) 
entities similarity is taken with siblings of class and with class label.
siblings vector is average of siblings.
class vector is average of label word vectors.
Parent class means class in which entity is categorized because entity itself is called class hence parent class is 
class in which entity and siblings are present.


Since parent class vector(class lebel vectors) are precomputed
and for similarity purpose take array of class vectors and array of entity vectors then do multiplication to get
result fast due to vectorization.

for sibling vectors store the sum of subling vector and when new sibling added just add vector 
then take array of sibling vectors and array of entities do vector multiplication.