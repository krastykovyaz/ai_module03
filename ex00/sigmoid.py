import numpy as np

def sigmoid_(x):
    """
    Compute the sigmoid of a vector.
    Args:
    x: has to be a numpy.ndarray of shape (m, 1).
    Returns:
    The sigmoid value as a numpy.ndarray of shape (m, 1).
    None if x is an empty numpy.ndarray.
    Raises:
    This function should not raise any Exception.
    """
    try:
        if not isinstance(x, np.ndarray):
            return None
        if x.size==0:
            return None
        return 1 / (1 + (np.exp(-x)))
    except Exception as e:
        print(e)
        return None
    
if __name__=='__main__':
    # Example 1:
    x = np.array([[-4]])
    # print(sigmoid_(x))
    # Output:
    assert np.allclose(sigmoid_(x), np.array([[0.01798620996209156]]))
    # Example 2:
    x = np.array([[2]])
    sigmoid_(x)
    # Output:
    np.array([[0.8807970779778823]])
    # Example 3:
    x = np.array([[-4], [2], [0]])
    print(sigmoid_(x))
    # Output:
    assert np.allclose(sigmoid_(x), np.array([[0.01798620996209156], [0.8807970779778823], [0.5]]))