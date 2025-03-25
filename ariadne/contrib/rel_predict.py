from pathlib import Path
import pickle

from cassis import Cas

import pandas as pd

from ariadne.contrib.inception_util import SENTENCE_TYPE, SPAN_TYPE, RELATION_TYPE, create_relation_prediction, extract_dictionary_from_tensor
from ariadne.classifier import Classifier
from ariadne.Recommender import * 


class RelPreClassifier(Classifier):
    def __init__(self, model_name: str = 'Recommender_instance.pkl', model_directory: Path = 'ariadne/models'):
        super().__init__(model_directory=model_directory)
        try:
            model_path = model_directory + "/" + model_name
            print(model_path)
            with open(model_path, 'rb') as model_conf:
                Recommender = pickle.load(model_conf)
                Recommender.initialize_NN()
                self._model = Recommender
        except OSError:
            print(f"Recommender instance model NOT FOUND in {model_directory}")
 
    def predict(self, cas: Cas, layer: str, feature: str, project_id: str, document_id: str, user_id: str):
        cas_sentences = cas.select(SENTENCE_TYPE)

        for cas_sentence in cas_sentences:

            frase = {
                'sentence':[],
                'head_start':[],
                'head_end':[],
                'tail_start':[],
                'tail_end':[]
            }
            
            cas_relation = cas.select_covered(RELATION_TYPE, cas_sentence)[0]
            cas_head = cas_relation.Governor
            cas_tail = cas_relation.Dependent
            cas_label = cas_relation.label

            head_start = cas_head.begin - cas_sentence.begin
            head_end = cas_head.end - cas_sentence.begin
            tail_start = cas_tail.begin - cas_sentence.begin
            tail_end = cas_tail.end - cas_sentence.begin

            frase['sentence'].append(cas_sentence.get_covered_text())
            frase['head_start'].append(head_start)
            frase['head_end'].append(head_end)
            frase['tail_start'].append(tail_start)
            frase['tail_end'].append(tail_end)

            frase = pd.DataFrame(frase)
            
            tensor = self._model.compute_predictions_and_scores(frase)

            label, score = extract_dictionary_from_tensor(tensor)

            if label != cas_label and label is not None:
                prediction = create_relation_prediction(cas, layer, feature,cas_head, cas_tail, label, score, auto_accept=True)
                cas.add(prediction)
