from deepod.models.tabular import SLAD as slad

class SLAD():
    def __init__(self):
       pass
    
    def predict(self, x):
        return self.clf.decision_function(x.numpy())
    
    def fit(self, x):
        self.clf = slad()
        self.clf.fit(x.numpy())
        
