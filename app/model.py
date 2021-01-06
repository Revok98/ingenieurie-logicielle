from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu import config

class Interpret :
    interpreter = None
    def __init__(self, mu=None):
        self.interpreter = Interpreter.load("model")

    def predict(self,stri : str):
        dico_iter = self.interpreter.parse(stri)['intent_ranking']
        dico_ret = {}
        for i in dico_iter :
            dico_ret[i['name']] = str(i['confidence'])
        print(dico_ret)
        return dico_ret
