# Extract PCA components
import numpy as np
from sklearn.decomposition import PCA
ds = np.genfromtxt(
    "/content/PWr-OB-Metrics/datasets/COVID_19.csv", delimiter=',')
X = ds[:, :-1]
y = ds[:, -1]

X = PCA(n_components=8).fit_transform(X)
# chi = SelectKBest(score_func=chi2, k=10).fit_transform(X, y)

ds = np.concatenate((X, y.reshape(y.shape[0], 1)), axis=1)
np.savetxt("/content/PWr-OB-Metrics/datasets" +
           "/COVID19_PCA_8.csv", ds, delimiter=",")

pca = np.genfromtxt(
    "/content/PWr-OB-Metrics/datasets/COVID19_PCA_8.csv", delimiter=',')

print(pca.shape)
