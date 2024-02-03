import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

PTH_PLANETS = 'solar_system_census_planets.csv'
PTH_SITIZENS = 'solar_system_census.csv'

class MyLogReg:
    def __init__(self) -> None:
        self.theta = None
        self.alpha = 5e-5
        self.max_iter = 100000

    @staticmethod
    def standard(vls):
        return np.array([((val - np.mean(vls)) / np.std(vls)) for val in vls])

    @staticmethod
    def sigmoid(z, theta):
        return 1 / (1 + np.exp(-(z @ theta)))

    def fit_(self,  x, y):
        x = np.array([self.standard(col) for col in x.T]).T
        self.theta = np.random.rand(x.shape[1] + 1).reshape(-1,1) # 4x1
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
    
    def predict_proba(self, x_):
        x = self.standard(x_)
        x_one = np.hstack((np.ones((x.shape[0], 1)), x)) # 3x1->4x1
        y_hat = self.sigmoid(x_one, self.theta) # 
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
    correct_predictions = np.sum(predictions == y_test.flatten())
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

def read_data():
    df_sscp = pd.read_csv(PTH_PLANETS, index_col='Unnamed: 0')
    df_ssc = pd.read_csv(PTH_SITIZENS, index_col='Unnamed: 0')
    return df_sscp, df_ssc

def markup_data(df_, cls):
    out = []
    for val in df_.iloc[:,0].tolist():
        if val == cls:
            out.append(cls)
        else: 
            out.append(0)
    return out

def prepare_data(df_sscp, df_ssc, cls):
    df_sscp.iloc[:,0] = markup_data(df_sscp, cls)
    X, y = df_ssc.values, df_sscp.iloc[:,0].values.reshape(-1,1)
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=.2)
    return X_train, X_test, y_train, y_test

def select_class(classifiers, features):
    class_ = None
    score_ = 0
    for classifier, lr in classifiers.items():
        pred = lr.predict_proba(features).flatten()[0]
        if pred > score_:
            score_ = pred
            class_ = classifier
    return class_





def create_classifiers():
    df_sscp, df_ssc = read_data()
    classes = sorted(df_sscp.iloc[:,0].unique())
    classifiers = {}
    for cls in classes:
        X_train, X_test, y_train, y_test = prepare_data(df_sscp, df_ssc, cls)
        mlr = MyLogReg()
        mlr.fit_(X_train, y_train)
        classifiers[cls] = mlr
    # print(X_test[0].reshape(1,-1).shape, mlr.theta.shape)
    # print()
    predictions = [select_class(classifiers, feature.reshape(1,-1)) for feature in (X_test)]
    evaluation(X_test, np.array(predictions), y_test)
    # return predictions
    # predictions = mlr.predict_(X_test)
    # evaluation(X_test, predictions, y_test)


def main():
    # from sklearn.linear_model import LogisticRegression
    return create_classifiers()
    # lr = LogisticRegression()
    # lr.fit(X_train, y_train)
    # return mlr.loss_(X_test, y_test)

if __name__=='__main__':
    print(main())