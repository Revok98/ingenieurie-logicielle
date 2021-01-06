from rasa_nlu.model import Metadata, Interpreter
from rasa_nlu import config

class Interpret :
    interpreter = None
    def __init__(self, mu=None):
        self.interpreter = Interpreter.load("model")

    def predict(self,stri : str):
        dico_iter = self.interpreter.parse(stri)['intent_ranking']
        print(dico_iter)
        if(dico_iter == []) :
            dico_ret = {"irrelevant":"1.0","provide-showtimes":"0.0","find-around-me":"0.0","find-restaurant":"0.0","find-hotel":"0.0","purchase":"0.0","find-flight":"0.0","find-train":"0.0"}
        else :
            dico_ret = {}
            for i in dico_iter :
                dico_ret[i['name']] = str(i['confidence'])
        print(dico_ret)
        return dico_ret
