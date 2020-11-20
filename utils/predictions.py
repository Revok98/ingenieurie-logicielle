import requests
import json

def generate_predictions_dataset(query_dataset_path="../data/testing_set", endpoint="http://localhost:8080/api/intent", save_on_disk=False):
    """
    Return list of dict containing intent probabilities for each sentence in the query dataset (in the same order)
    """
    with open(query_dataset_path) as f:
        data = json.load(f)
    query_prefix = endpoint + "?sentence="
    dataset = []
    for query_data in data:
        r = requests.get(query_prefix + query_data["sentence"])
        dataset.append(r.json)

    if save_on_disk:
        with open('predict.json', 'w') as outfile:
            json.dump(dataset, outfile, indent=4)

    return dataset

def get_detected_intent(prediction):
    """
    Return the most probable intent (str) from a prediction (dict containing intent probabilities)
    """
    return max(prediction, key=lambda x: prediction[x])

def stats_by_intent(dataset, predictions):
    pass