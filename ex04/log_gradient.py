import numpy as np

def log_gradient(x, y, theta):
    """Computes a gradient vector from three non-empty numpy.ndarray, with a for-loop. The three arrays must have compatiblArgs:
    x: has to be an numpy.ndarray, a matrix of shape m * n.
    y: has to be an numpy.ndarray, a vector of shape m * 1.
    theta: has to be an numpy.ndarray, a vector of shape (n + 1) * 1.
    Returns:
    The gradient as a numpy.ndarray, a vector of shape n * 1, containing the result of the formula for all j.
    None if x, y, or theta are empty numpy.ndarray.
    None if x, y and theta do not have compatible dimensions.
    Raises:
    This function should not raise any Exception.
    """
    grad = np.zeros(shape=(theta.shape))
    m, n = x.shape
    y_hat = logistic_predict(x, theta)
    for i in range(m):
        grad[0][0] += (y_hat[i][0] - y[i][0])
        for j in range(n):
            grad[j + 1][0] += (y_hat[i][0] - y[i][0]) * (x[i][j])
    return grad / m

def sigmoid(x, theta):
    return 1 / (1 + (np.exp(-(x @ theta))))

def logistic_predict(x, theta, eps=1e-15):
    x_one = np.hstack((np.ones((x.shape[0], 1)),x ))
    pred = sigmoid(x_one, theta)
    return np.clip(pred, eps,np.inf)

if __name__=='__main__':
    # Example 1:
    y1 = np.array([1]).reshape((-1, 1))
    x1 = np.array([4]).reshape((-1, 1))
    theta1 = np.array([[2], [0.5]])
    # Output:
    # assert np.allclose(log_gradient(x1, y1, theta1),
                    #    np.array([[-0.01798621], [-0.07194484]]))
    # Example 2:
    y2 = np.array([[1], [0], [1], [0], [1]])
    x2 = np.array([[4], [7.16], [3.2], [9.37], [0.56]])
    theta2 = np.array([[2], [0.5]])
    # Output:
    assert np.allclose(log_gradient(x2, y2, theta2),
                       np.array([[0.3715235 ],[3.25647547]]))
    # # Example 3:
    # y3 = np.array([[0], [1], [1]])
    # x3 = np.array([[0, 2, 3, 4], [2, 4, 5, 5], [1, 3, 2, 7]])
    # theta3 = np.array([[-2.4], [-1.5], [0.3], [-1.4], [0.7]])
    # # Output:
    # print(log_gradient(x3, y3, theta3))
    # np.array([[-0.55711039],
    # [-0.90334809],
    # [-2.01756886],
    # [-2.10071291],
    # [-3.27257351]])