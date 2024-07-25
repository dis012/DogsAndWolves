import torch
import numpy as np

class EarlyStopping:
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pt'):
        self.patience = patience # How long to wait after last time validation loss improved
        self.verbose = verbose # If True, prints a message for each validation loss improvement
        self.counter = 0 # Counter tracks how many epochs have passed since last improvement
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss # Class is designed to work with higher is better principle, but since lower validation loss is better, we need to negate it

        # First epoch check
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        
        # Improvement check
        elif score < self.best_score + self.delta: # If score is not better than best_score + delta
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        
        # Improvement found
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
        
    def save_checkpoint(self, val_loss, model):
        # Save model if validation loss has decreased
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss