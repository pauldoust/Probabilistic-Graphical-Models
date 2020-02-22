import numpy as np
from sklearn.manifold.t_sne import TSNE

def convert_2d(model):
    X = np.array(model.components.X)
    for k in range(model.components.K):
        mu,_ = model.components.rand_k(k)
        mu = mu.reshape(1,-1)
        X = np.concatenate((X, mu), axis=0)


    X_embedded_train_valid = TSNE(n_components=2).fit_transform(X) 
    y_train_valid = model.components.assignments

    centers = X_embedded_train_valid[model.components.X.shape[0]:]
    X_embedded_train_valid = X_embedded_train_valid[:model.components.X.shape[0]]

    return X_embedded_train_valid, y_train_valid, centers
