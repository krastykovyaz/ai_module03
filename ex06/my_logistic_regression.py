import numpy as np

class MyLogisticRegression():
    """
    Description:
    My personnal logistic regression to classify things.
    """
    def __init__(self, theta, alpha=0.001, max_iter=1000):
        self.alpha = alpha
        self.max_iter = max_iter
        self.theta = theta

    def fit_(self, x, y):
        
        def descent(x_, y_, theta):
            m = y.shape[0]
            x_one = np.hstack((np.ones((x.shape[0], 1)), x)) # 3x5
            y_hat = 1 / (1 + np.exp(-(x_one @ theta))) # 3x5 @ 5x1 = 3x1
            y_hat = np.clip(y_hat, 1e-15, np.inf)
            return (x_one.T @ (y_hat - y_)) / m # 5x3 @ 3x1 = 5x1
        for _ in range(self.max_iter):
            gradient = descent(x, y, self.theta)
            self.theta -= gradient * self.alpha

        return self.theta
    
    def predict_(self, x):
        x_one = np.hstack((np.ones((x.shape[0], 1)), x)) # 3x5
        y_hat = 1 / (1 + np.exp(-(x_one @ self.theta)))
        return y_hat
    
    def loss_(self, x_, y):
        x_one = np.hstack((np.ones((x_.shape[0],1)), x_))
        y_hat = x_one @ self.theta
        y_hat = np.clip(y_hat, 1e-15, np.inf)
        return -np.mean((1-y) * np.log(1-y_hat) + (y) * np.log(y_hat))



if __name__=='__main__':
    # from sklearn.linear_model import LogisticRegression
    X = np.array([[1., 1., 2., 3.], [5., 8., 13., 21.], [3., 5., 9., 14.]]) #3x4
    Y = np.array([[1], [0], [1]]) #3x1
    thetas = np.array([[2], [0.5], [7.1], [-4.3], [2.09]]) # 5x1
    mylr = MyLogisticRegression(thetas)
    # lr = LogisticRegression()
    # lr.thetas = thetas
    # lr.fit(X,Y.ravel())
    # assert np.allclose(lr.thetas, mylr.fit_(X,Y))
    # Example 0:
    # print(mylr.predict_(X), lr.predict(X))
    # Output:
    np.array([[0.99930437], [1. ], [1. ]])
    # Example 1:
    print(mylr.loss_(X,Y))
    # Output:
    11.513157421577004
    # Example 2:
    mylr.fit_(X, Y)
    # Output:
    print(np.array([[ 2.11826435],
    [ 0.10154334],
    [ 6.43942899],
    [-5.10817488],
    [ 0.6212541 ]]))
    print(mylr.theta)
    # Example 3:
    print(mylr.predict_(X))
    # Output:
    print(np.array([[0.57606717],
    [0.68599807],
    [0.06562156]]))
    # Example 4:
    print(mylr.loss_(X,Y))
    # Output:
    1.4779126923052268