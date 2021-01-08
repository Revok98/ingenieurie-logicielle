from typing import Optional
from locust import HttpUser, between, task
from fastapi import FastAPI
from model import Interpret


app = FastAPI()
interpreter = Interpret()

def call_model(sentence : str) :
    return interpreter.predict(sentence)
    #return sentence


# @app.on_event('startup')
# def load_model():
#     #metadata = Metadata.load("NLU/nlu")   # where model_directory points to the folder the model is persisted in
#     interpreter.load()


@app.get("/")
async def read_root():
    first_str = {"root_page":"no result", "bienvenue sur la page d'accueil":"no result"}
    return first_str


@app.get("/api/{demande}")
#@app.get("/items/{item_id}")
async def read_item(demande: str, sentence:str):
    if(demande != "intent") :
        return {"detail" : "not found"}
    new_str = ""
    for i in sentence :
        if(i != "\"") :
            new_str += i
    new_str = call_model(new_str)
    print(new_str)
    return new_str

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)





# # No configuration for the NLU pipeline was provided. The following default pipeline was used to train your model.
# # If you'd like to customize it, uncomment and adjust the pipeline.
# # See https://rasa.com/docs/rasa/tuning-your-model for more information.
  # - name: WhitespaceTokenizer
  # - name: RegexFeaturizer
  # - name: LexicalSyntacticFeaturizer
  # - name: CountVectorsFeaturizer
  # - name: CountVectorsFeaturizer
  #   analyzer: char_wb
  #   min_ngram: 1
  #   max_ngram: 4
  # - name: DIETClassifier
  #   epochs: 100
  # - name: EntitySynonymMapper
  # - name: ResponseSelector
  #   epochs: 100
  # - name: FallbackClassifier
  #   threshold: 0.3
  #   ambiguity_threshold: 0.1

#   language: fr
# policies:
# - name: TEDPolicy
#   max_history: 5
#   epochs: 100
# pipeline:
# - name: "CountVectorsFeaturizer"
# - name: "EmbeddingIntentClassifier"
#   intent_tokenization_flag: true
#   intent_split_symbol: "+"
