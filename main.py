import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import time

file_name = "bla.txt"
A = np.loadtxt(file_name, delimiter=',')


kmeans = KMeans(
    init='k-means++',
    n_clusters=3,
    max_iter=300,
    random_state=42
)

start1 = time.time()
res = kmeans.fit(A)
elapsed1 = time.time() - start1
print(f"Kmeans TIME: {elapsed1}")
exit()
silVal = silhouette_score(A, res.labels_)

print(f"silhouette TIME: {elapsed1}")
print(f"sil value: {silVal}")
