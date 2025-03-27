from typing import Optional

from cassis import Cas
from cassis.typesystem import FeatureStructure

SPAN_TYPE = "custom.Span"
RELATION_TYPE = "custom.Relation"
SENTENCE_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Sentence"
TOKEN_TYPE = "de.tudarmstadt.ukp.dkpro.core.api.segmentation.type.Token"
IS_PREDICTION = "inception_internal_predicted"
FEATURE_NAME_SCORE_SUFFIX = "_score"
FEATURE_NAME_SCORE_EXPLANATION_SUFFIX = "_score_explanation"
FEATURE_NAME_AUTO_ACCEPT_MODE_SUFFIX = "_auto_accept"

def create_relation_prediction(
    cas: Cas,
    layer: str,
    feature: str,
    source: FeatureStructure,
    target: FeatureStructure,
    label: Optional[str] = None,
    score: Optional[int] = None,
    score_explanation: Optional[str] = None,
    auto_accept: Optional[bool] = None,
) -> FeatureStructure:
    AnnotationType = cas.typesystem.get_type(layer)

    fields = {
        "begin": target.begin,
        "end": target.end,
        "Governor": source,
        "Dependent": target,
        IS_PREDICTION: True,
        feature: label,
        #"identifier": ""
    }
    prediction = AnnotationType(**fields)

    if score is not None:
        prediction[f"{feature}{FEATURE_NAME_SCORE_SUFFIX}"] = score

    if score_explanation is not None:
        prediction[f"{feature}{FEATURE_NAME_SCORE_EXPLANATION_SUFFIX}"] = score_explanation

    if auto_accept is not None:
        prediction[f"{feature}{FEATURE_NAME_AUTO_ACCEPT_MODE_SUFFIX}"] = auto_accept

    return prediction

def extract_dictionary_from_tensor(tensor_output):
    for key, value in tensor_output.items():
        print(key)
        if isinstance(key, str) and isinstance(value, float):
            return key, value
    return None, None
