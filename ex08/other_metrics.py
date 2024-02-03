import numpy as np

def accuracy_score_(y, y_hat):
    """
    Compute the accuracy score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    Returns:
    The accuracy score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    correct = np.sum(y == y_hat)
    all = len(y)
    return correct/all

def precision_score_(y, y_hat, pos_label=1):
    """
    Compute the precision score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The precision score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = y_hat.flatten()
    y = y.flatten()
    tp = 0
    fp = 0
    for y_h, y_ in zip(y_hat, y):
        if y_ == pos_label and y_h == y_:
            tp += 1
        elif y_ != pos_label and y_h != y_:
            fp += 1
    return tp / (tp + fp)


def recall_score_(y, y_hat, pos_label=1):
    """
    Compute the recall score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Return:
    The recall score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    y_hat = y_hat.flatten()
    y = y.flatten()
    tp = 0
    fn = 0
    for y_h, y_ in zip(y_hat, y):
        if y_ == pos_label and y_h == y_:
            tp += 1
        elif y_ == pos_label and y_h != y_:
            fn += 1
    return tp / (tp + fn)

def f1_score_(y, y_hat, pos_label=1):
    """
    Compute the f1 score.
    Args:
    y:a numpy.ndarray for the correct labels
    y_hat:a numpy.ndarray for the predicted labels
    pos_label: str or int, the class on which to report the precision_score (default=1)
    Returns:
    The f1 score as a float.
    None on any error.
    Raises:
    This function should not raise any Exception.
    """
    precision = precision_score_(y, y_hat, pos_label)
    recall = recall_score_(y, y_hat, pos_label)
    numenator = precision * recall
    denominator = precision + recall
    return 2 * numenator / denominator


if __name__=='__main__':
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    # Example 1:
    y_hat = np.array([1, 1, 0, 1, 0, 0, 1, 1]).reshape((-1, 1))
    y = np.array([1, 0, 0, 1, 0, 1, 0, 0]).reshape((-1, 1))
    # Accuracy

    ## Output:
    assert accuracy_score_(y, y_hat) == 0.5
    ## sklearn implementation
    assert accuracy_score(y, y_hat) == 0.5

    # Precision
    ## Output:
    assert precision_score_(y, y_hat) == 0.4

    ## sklearn implementation
    assert precision_score(y, y_hat) == 0.4
    ## Output:
    0.4
    # Recall
    ## your implementation
    assert np.allclose(recall_score_(y, y_hat), 0.6666666666666666)
    ## Output:
    0.6666666666666666
    ## sklearn implementation
    recall_score(y, y_hat)
    ## Output:
    0.6666666666666666
    # F1-score
    ## your implementation
    assert f1_score_(y, y_hat) == 0.5
    ## Output:
    0.5
    ## sklearn implementation
    f1_score(y, y_hat)

    # Example 2:
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    # Accuracy
    ## your implementation
    accuracy_score_(y, y_hat)
    ## Output:
    assert accuracy_score_(y, y_hat) == 0.625
    ## sklearn implementation
    accuracy_score(y, y_hat)
    ## Output:
    0.625
    # Precision
    ## your implementation
    assert precision_score_(y, y_hat, pos_label='dog') == 0.6

    ## sklearn implementation
    print(precision_score(y, y_hat, pos_label='dog'))
    ## Output:
    0.6
    # Recall
    ## your implementation
    assert recall_score_(y, y_hat, pos_label='dog')==0.75
    ## Output:
    0.75
    ## sklearn implementation
    recall_score(y, y_hat, pos_label='dog')
    ## Output:
    0.75
    # F1-score
    ## your implementation
    assert np.allclose(f1_score_(y, y_hat, pos_label='dog'),0.6666666666666665)
    ## Output:
    0.6666666666666665
    ## sklearn implementation
    f1_score(y, y_hat, pos_label='dog')

    # Example 3:
    y_hat = np.array(['norminet', 'dog', 'norminet', 'norminet', 'dog', 'dog', 'dog', 'dog'])
    y = np.array(['dog', 'dog', 'norminet', 'norminet', 'dog', 'norminet', 'dog', 'norminet'])
    # Precision
    ## your implementation
    assert np.allclose(precision_score_(y, y_hat, pos_label='norminet'),0.6666666666666666)
    ## Output:
    0.6666666666666666
    ## sklearn implementation
    precision_score(y, y_hat, pos_label='norminet')
    ## Output:
    0.6666666666666666
    # Recall
    ## your implementation
    assert np.allclose(recall_score_(y, y_hat, pos_label='norminet'), 0.5)
    ## Output:
    0.5
    ## sklearn implementation
    recall_score(y, y_hat, pos_label='norminet')
    ## Output:
    0.5
    # F1-score
    ## your implementation
    assert np.allclose(f1_score_(y, y_hat, pos_label='norminet'),0.5714285714285715)
    ## Output:
    0.5714285714285715
    ## sklearn implementation
    f1_score(y, y_hat, pos_label='norminet')
    ## Output:
    0.5714285714285715