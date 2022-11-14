import numpy as np

def iou(pred, lbl):
    i = np.logical_and(pred, lbl)
    u = np.logical_or(pred, lbl)
    
    return np.sum(i)/np.sum(u)

def acc(pred, lbl):
    return np.sum(lbl==pred)/np.prod(pred.shape)


METRICS = {
    "accuracy":
        acc,
    "iou":
        iou
}