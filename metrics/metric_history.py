import numpy as np

class MetricHistory():
    def __init__(self):
        self.metric = np.array([])
    
    def __call__(self, value):
        self.metric= np.append(self.metric, value)
        
    def avg(self):
        return np.sum(self.metric)/len(self.metric)
    
    def reset(self):
        self.metric= np.array([])


class MetricHistoryNew():
    def __init__(self):
        self.learning_history = np.array([])
        self.epoch_metric = np.array([])
        self.best = 0
        self.last = 0

    def get_better(self):
        return self.last == self.best
    
    def __call__(self, value):
        self.epoch_metric= np.append(self.epoch_metric, value)

    
    def reset(self):
        self.epoch_metric= np.array([])


    def finalize_epoch(self):

        self.last = np.mean(self.epoch_metric)

        self.learning_history = np.append(
            self.learning_history,
            self.last
        )
        if self.last > self.best:
            self.best = self.last

        self.reset()