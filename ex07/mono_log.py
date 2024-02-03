import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class MyLogReg:
    def __init__(self) -> None:
        self.theta = None
        self.alpha = 1e-5
        self.max_iter = 10000

    @staticmethod
    def standard(vls):
        return np.array([((val - np.mean(vls)) / np.std(vls)) for val in vls])

    @staticmethod
    def sigmoid(z, theta):
        return 1 / (1 + np.exp(-(z @ theta)))

    def fit_(self,  x, y):
        x = np.array([self.standard(col) for col in x.T]).T
        self.theta = np.random.rand(x.shape[1] + 1).reshape(-1,1)
        def descent(x_, y_, theta):
            x_one = np.hstack((np.ones((x.shape[0], 1)),x_))
            y_hat = self.sigmoid(x_one, theta)
            y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
            return x_one.T @ (y_hat - y_)
        
        for _ in range(self.max_iter):
            grad = descent(x, y, self.theta)
            self.theta -= (grad * self.alpha)
        return self.theta
    
    def predict_(self, x):
        x = np.array([self.standard(col) for col in x.T]).T
        x_one = np.hstack((np.ones((x.shape[0], 1)), x))
        y_hat = self.sigmoid(x_one, self. theta)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return y_hat
    
    def loss_(self,x_, y):
        y_hat = self.predict_(x_)
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return - np.mean((y * np.log(y_hat)) + ((1-y) * np.log(1-y_hat)))




def create_favorite_planet_label(zipcodes, favorite_zipcode):
    return np.array([1 if zipcode == favorite_zipcode else 0 for zipcode in zipcodes])

def plot_scatter_plots(X, y, predictions):
    num_features = X.shape[1]
    fig, axes = plt.subplots(nrows=num_features, ncols=num_features, figsize=(15, 15))
    for i in range(num_features):
        for j in range(num_features):
            if i != j:
                axes[i, j].scatter(X[:, i], X[:, j], c=y, cmap=plt.cm.Paired, edgecolors='k')
                axes[i, j].scatter(X[:, i], X[:, j], c=predictions, cmap=plt.cm.Paired, marker='x', s=100)
                axes[i, j].set_xlabel(f'Feature {i + 1}')
                axes[i, j].set_ylabel(f'Feature {j + 1}')

    plt.show()

   
def evaluation(x_tt, predictions, y_test):
    # Evaluate the model
    correct_predictions = np.sum(predictions.round() == y_test)
    total_predictions = len(y_test)
    accuracy = correct_predictions / total_predictions
    print(f'Fraction of correct predictions: {accuracy:.2%}')

    # Plot scatter plots
    plot_scatter_plots(x_tt, y_test, predictions)



def split_data(x, y, test_size=.2):
    idxs = list(range(len(x)))
    np.random.shuffle(idxs)
    part_tr, part_te = idxs[int(len(idxs) * .2):], idxs[:int(len(idxs) * .2)]
    x_train, x_test = x[part_tr], x[part_te]
    y_train, y_test= y[part_tr], y[part_te]
    return x_train, x_test, y_train, y_test

def read_data(path_pl, path_sitiz):
    df_sscp = pd.read_csv(path_pl, index_col='Unnamed: 0')
    df_ssc = pd.read_csv(path_sitiz, index_col='Unnamed: 0')
    df_sscp.iloc[:,0] = df_sscp.iloc[:,0].replace(2,0).replace(3,0)
    return df_ssc.values, df_sscp.iloc[:,0].values.reshape(-1,1)
                

def main():
    # from sklearn.linear_model import LogisticRegression
    path_planets = 'solar_system_census_planets.csv'
    path_sitizens = 'solar_system_census.csv'
    X, y = read_data(path_planets, path_sitizens)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=.2)
    mlr = MyLogReg()
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    mlr.fit_(X_train, y_train)
    predictions = mlr.predict_(X_test)
    evaluation(X_test, predictions, y_test)
    return mlr.loss_(X_test, y_test)

if __name__=='__main__':
    print(main())