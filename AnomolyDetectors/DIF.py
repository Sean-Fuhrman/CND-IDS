from deepod.models.tabular import dif

class DIF():
    def __init__(self):
       pass
    
    def predict(self, x):
        return self.clf.decision_function(x.numpy())
    
    def fit(self, x):
        self.clf = dif.DeepIsolationForest()
        self.clf.fit(x.numpy())
        
