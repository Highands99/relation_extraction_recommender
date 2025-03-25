import pickle

from ariadne.Recommender import *
from ariadne.server import Server
from ariadne.contrib.rel_predict import RelPreClassifier

# http://localhost:5000/
server = Server()
server.add_classifier("wikidata_rel", RelPreClassifier())

if __name__ == "__main__":
    server.start()
