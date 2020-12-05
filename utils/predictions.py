import requests
import json
import pandas as pd

def generate_predictions_dataset(query_dataset_path="../data/testing_set.json", endpoint="http://localhost:8000/api/intent", save_on_disk=False):
    """
    Return list of dict containing intent probabilities for each sentence in the query dataset (in the same order)
    """
    with open(query_dataset_path, encoding="utf8") as f:
        data = json.load(f)
    query_prefix = endpoint + "?sentence="
    dataset = []
    for query_data in data:
        r = requests.get(query_prefix + query_data["sentence"])
        dataset.append(r.json())

    if save_on_disk:
        with open('../data/predict.json', 'w') as outfile:
            json.dump(dataset, outfile, indent=4)

    return dataset

def get_detected_intent(prediction):
    """
    Return the most probable intent (str) from a prediction (dict containing intent probabilities)
    """
    return max(prediction, key=lambda x: prediction[x])

def stats_by_intent(dataset, predictions):
    """
    Returns a dict with TP, FP, TP+FN, TP+FP for each intent + "global" intent
    """
    stats_by_intent = {}

    # add key for each intent + global 
    for intent in predictions[0].keys():
        stats_by_intent[intent] = {"TP": 0, "FP": 0, "TP+FN": 0, "TP+FP": 0}

    # computes stats
    for i, query in enumerate(dataset):
        detected_intent = get_detected_intent(predictions[i])
        true_intent = query["intent"]

        if detected_intent == true_intent:
            stats_by_intent[detected_intent]["TP"] += 1
        
        stats_by_intent[detected_intent]["TP+FP"] += 1
        stats_by_intent[true_intent]["TP+FN"] += 1

    # isolate FP for convenience 
    for intent in stats_by_intent:
        stats_by_intent[intent]["FP"] = stats_by_intent[intent]["TP+FP"] - stats_by_intent[intent]["TP"]
    
    # precision and recall
    for intent in stats_by_intent:
        if stats_by_intent[intent]["TP+FP"] != 0:
            stats_by_intent[intent]["precision"] = stats_by_intent[intent]["TP"] / stats_by_intent[intent]["TP+FP"]
        if stats_by_intent[intent]["TP+FN"] != 0:
            stats_by_intent[intent]["recall"] = stats_by_intent[intent]["TP"] / stats_by_intent[intent]["TP+FN"]
    
    return stats_by_intent

def print_stats_by_intent(stats):
    df = pd.DataFrame(stats)
    print(df.T)