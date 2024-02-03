import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from collections import OrderedDict

def confusion_matrix_(y, y_hat, labels=['bird', 'dog', 'norminet']):
    y = y.flatten()
    y_hat = y_hat.flatten()
    out = []
    for label in labels:
        res = OrderedDict({label:0 for label in labels})
        for y_h, y_ in zip(y_hat, y):
            if y_ == label == y_h:
                res[label] += 1
            elif y_ == label and y_h != label and y_h in res:
                res[y_h] += 1
        out.append(list(res.values()))                
    return np.array(out)
            
        


if __name__=='__main__':
    y_hat = np.array([['norminet'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['bird']])
    y = np.array([['dog'], ['dog'], ['norminet'], ['norminet'], ['dog'], ['norminet']])
    # Example 1:
    ## Output:
    assert np.allclose(confusion_matrix_(y, y_hat), np.array([[0, 0, 0],
    [0, 2, 1],
    [1, 0, 2]]))
    ## sklearn implementation
    print(confusion_matrix(y, y_hat))
    ## Output:
    np.array([[0, 0, 0],
    [0, 2, 1],
    [1, 0, 2]])
    # Example 2:
    ## your implementation
    ## Output:
    assert np.allclose(confusion_matrix_(y, y_hat, labels=['dog', 'norminet']), np.array([[2, 1],
    [0, 2]]))
    ## sklearn implementation
    print(confusion_matrix(y, y_hat, labels=['dog', 'norminet']))
    ## Output:
    np.array([[2, 1],[0, 2]])