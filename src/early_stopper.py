import torch

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode="min"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = None
        self.counter = 0
        self.should_stop = False

    def step(self, metric):
        if self.best is None:
            self.best = metric
            return False

        improvement = (
            metric < self.best - self.min_delta
            if self.mode == "min"
            else metric > self.best + self.min_delta
        )

        if improvement:
            self.best = metric
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop