from deepod.models.tabular import ICL as icl

class ICL():
    def __init__(self):
       pass
    
    def predict(self, x):
        return self.clf.decision_function(x.numpy())
    
    def fit(self, x):
        self.clf = icl()
        self.clf.fit(x.numpy())