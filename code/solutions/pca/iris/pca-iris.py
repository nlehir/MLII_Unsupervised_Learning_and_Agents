import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.decomposition import PCA

iris = datasets.load_iris()

print(iris.feature_names)
print(iris.target)
print(iris.DESCR)
print(iris.filename)

X, Y = iris.data, iris.target
# used to differentiate the classes
colMap = {0: "indianred", 1: "blue", 2: "darkorchid"}
colors = list(map(lambda x: colMap.get(x), Y))

pca = PCA(n_components=2)
pca.fit(X)

# fit PCA and project the data on
# the principal components
X_projected = PCA(n_components=2).fit_transform(X)

# principal component obtained by the algorithm
print("components")
print(pca.components_)

plt.scatter(X_projected[:, 0], X_projected[:, 1], alpha=0.7, c=colors)
plt.title("pcs iris")
plt.savefig("pca_iris.pdf")
