import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=7, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            path (str): Path for the checkpoint to be saved to.
                            Default: 'checkpoint.pt'
            trace_func (function): trace print function.
                            Default: print
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_val_loss = None
        self.best_class_loss = None
        self.best_trans_loss = None
        self.best_rot_loss = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, class_loss, trans_loss, rot_loss, model):
        # Check if validation loss is nan
        if np.isnan(val_loss):
            self.trace_func("Validation loss is NaN. Ignoring this epoch.")
            
            
        if val_loss < self.best_val_loss - self.delta or class_loss < self.best_class_loss - self.delta or trans_loss < self.best_trans_loss - self.delta or rot_loss < self.best_rot_loss - self.delta:
            self.save_checkpoint(val_loss, model)
            self.counter = 0  # Reset counter since improvement occurred
        else:
            self.counter += 1
            
        if self.best_val_loss is None:
            self.best_val_loss = val_loss
        if self.best_class_loss is None:
            self.best_class_loss = class_loss
        if self.best_trans_loss is None:
            self.best_trans_loss = trans_loss
        if self.best_rot_loss is None:
            self.best_rot_loss = rot_loss
        
            
        if val_loss < self.best_val_loss - self.delta:
            self.best_val_loss = val_loss
        if class_loss < self.best_class_loss - self.delta:
            self.best_class_loss = class_loss
        if trans_loss < self.best_trans_loss - self.delta:
            self.best_trans_loss = trans_loss
        if rot_loss < self.best_rot_loss - self.delta:
            self.best_rot_loss = rot_loss

        if self.counter >= self.patience:
            self.early_stop = True

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decreases.'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss