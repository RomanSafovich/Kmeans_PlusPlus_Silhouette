import math
import time
import numpy as np

file_name = "data_1_3.txt"
A = np.loadtxt(file_name, delimiter=',')


def min_distance(X, centroids):
    dis = [np.sum(np.square(np.subtract(X, centroid)), axis=1) for centroid in centroids]
    dis = np.array(dis).min(axis=0, keepdims=True)
    return dis


def init_kmeans_pp(X, K):
    centroids = [X[np.random.choice(range(X.shape[0])), :]]
    for _ in range(1, K):
        D_pow = min_distance(X, centroids)
        prob = D_pow / np.sum(D_pow)
        centroid_index = np.random.choice(range(X.shape[0]), p=prob[0])
        centroids.append(X[centroid_index])
        X = np.delete(X, centroid_index, axis=0)  # delete the row
    return centroids


def mean(x, centroids):
    cluster = [[] for i in range(len(centroids))]
    labels = np.zeros(x.shape[0], dtype='int')
    bla_counter = 0
    while True:
        bla_counter += 1
        temp_cluster = cluster.copy()
        cluster = [[] for i in range(len(centroids))]
        lbl_i = 0

        dis = [np.sqrt(np.sum(np.square(np.subtract(x, centroid)), axis=1)) for centroid in centroids]
        dis_idx = np.array(dis).argmin(axis=0)

        for vec_idx, cluster_idx in enumerate(dis_idx):
            cluster[cluster_idx].append(x[vec_idx])
            labels[lbl_i] = cluster_idx
            lbl_i += 1

        """
        start1 = time.time()
        for vec in x:
            temp = []
            for c in centroids:
                temp.append(np.sqrt((((vec - c) ** 2).sum())))  #
            index = temp.index(min(temp))
            cluster[index].append(vec)
            labels[lbl_i] = index
            lbl_i += 1
        elapsed1 = time.time() - start1
        print(f"OLD TIME: {elapsed1}")
        """
        flag = True
        for i in range(len(cluster)):
            equal_arrays = np.array_equal(np.array(cluster[i]), np.array(temp_cluster[i]))
            if not equal_arrays:  # if arrays are not equal, stop loop.
                flag = False
                break
        """
        for i in range(len(cluster)):
            if len(cluster[i]) == len(temp_cluster[i]):
                if not all([np.allclose(x, y) for x, y in zip(cluster[i], temp_cluster[i])]):
                    count += 1
            else:
                count += 1
        if count == 0:
            break
        """
        if flag:
            break
        for i in range(len(cluster)):
            centroids[i] = sum(cluster[i]) / len(cluster[i])
    print(bla_counter)
    return cluster, labels


def compute_distances_no_loops(X, Y):
    p1 = np.sum(X**2, axis=1)[:, np.newaxis]
    p2 = np.sum(Y**2, axis=1)
    p3 = -2 * np.dot(X, Y.T)
    return np.sqrt(np.abs(p1 + p2 + p3), dtype=np.float64)


K = 3
start1 = time.time()
centroids = init_kmeans_pp(A, K)
elapsed1 = time.time() - start1
print(f"INIT KMEANS - our TIME: {elapsed1}")
start1 = time.time()
clusters, labels = mean(A, centroids)
elapsed1 = time.time() - start1
print(f"KMEANS - our TIME: {elapsed1}")

# start1 = time.time()
# silhouette_values = opt_silhouette(A, labels, 3)
# elapsed1 = time.time() - start1
# print(f"SIL TIME: {elapsed1}")
# exit()
silhouette_values = np.zeros(A.shape[0])  # keep silhouette values for each vector

# for vec_idx in range(0, A.shape[0]):

start = time.time()
vec_idx = 0
for cls in range(len(clusters)):

    a = compute_distances_no_loops(np.array(clusters[cls]), np.array(clusters[cls]))
    a = np.sum(a, axis=1) / (a.shape[0] - 1)

    it_ls = [k for k in range(0, K)]
    it_ls.pop(cls)

    # ls = np.arange(K)
    # ls = np.delete(ls, curr_label)
    # for cluster_lbl in it_ls:
    # ls2 = ((curr_vec - np.array(clusters[cluster_lbl])) ** 2).sum() ** .5

    ls2 = [np.sum(compute_distances_no_loops(np.array(clusters[cls]),
                                      np.array(clusters[cluster_lbl])), axis=1) / len(clusters[cluster_lbl]) for cluster_lbl in it_ls]

    min_vec = ls2[0]
    for i in range(len(ls2)):
        if min_vec.shape[0] < ls2[i].shape[0]:
            for k in range(ls2[i].shape[0] - min_vec.shape[0]):
                min_vec = np.insert(min_vec, min_vec.shape[0], np.nan, axis=0)
        elif min_vec.shape[0] > ls2[i].shape[0]:
            for k in range(min_vec.shape[0] - ls2[i].shape[0]):
                ls2[i] = np.insert(ls2[i], ls2[i].shape[0], np.nan, axis=0)
        min_vec = np.minimum(min_vec, ls2[i])
    # ls2_len = len(clusters[cluster_lbl])
    # if ls2_len > 0:
    b = min_vec  #

    for i in range(a.shape[0]):
        print(f"{vec_idx}")
        silhouette_values[vec_idx] = np.subtract(b[i], a[i]) / np.maximum(a[i], b[i])
        vec_idx += 1


elapsed = time.time() - start
print(f"time for ours: {elapsed}")
exit()
vec_idx = 0
for curr_vec in A:
    # curr_vec = item
    curr_label = labels[vec_idx]  # The label of claster
    # startA = time.time()
    # a = np.dot(clusters[curr_label], curr_vec)

    # a = compute_distances_no_loops(np.array(clusters[curr_label], dtype=np.float64))
    # a = np.sum(a, axis=1) / (a.shape[0] - 1)

    # a = EDM(curr_vec, clusters[curr_label])
    # a = np.sqrt(np.sum(np.square(np.subtract(curr_vec, clusters[curr_label])))) / len(clusters[curr_label])
    # dis = np.sqrt(np.sum(np.square(np.subtract(curr_vec, clusters[curr_label])), axis=1))
    # dis_idx = np.array(dis).argmin(axis=0)

    # ls_len = len(clusters[curr_label]) - 1
    # if ls_len > 0:
    # a = ls / len(clusters[curr_label])
    # elapsedA = time.time() - startA
    # print(f"time for A: {elapsedA}")

    # startB = time.time()
    # b = float("inf")
    it_ls = [k for k in range(0, K)]
    it_ls.pop(curr_label)
    # ls = np.arange(K)
    # ls = np.delete(ls, curr_label)
    # for cluster_lbl in it_ls:
    # ls2 = ((curr_vec - np.array(clusters[cluster_lbl])) ** 2).sum() ** .5

    ls2 = [np.sqrt(np.sum(np.square(np.subtract(curr_vec, clusters[cluster_lbl])))) / len(clusters[cluster_lbl]) for
           cluster_lbl in it_ls]

    # ls2_len = len(clusters[cluster_lbl])
    # if ls2_len > 0:
    b = min(ls2)  #

    # save minimum avg distance to b
    # if distance_avg_to_cluster < b:
    #    b = distance_avg_to_cluster
    # elapsedB = time.time() - startB
    # print(f"time for B: {elapsedB}")
    # print(f"iter: {vec_idx}, elapsed: {elapsed}")
    silhouette_values[vec_idx] = (b - a[vec_idx]) / max(a[vec_idx], b)
    vec_idx += 1

elapsed = time.time() - start
print(f"time for ours: {elapsed}")
avg_silhouette = silhouette_values.sum() / A.shape[0]
print(avg_silhouette)
