import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from collections import Counter

# 加载保存的模型参数
W_in = np.load("W_in.npy", allow_pickle=True).item()

# 定义一个函数来获取词向量
def get_word_vector(word, matrix):
    if word in matrix:
        return matrix[word]
    else:
        print(f"Word '{word}' not found in vocabulary.")
        return None

# 定义一个函数来计算余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)

# 示例：获取词向量并计算相似度
word1 = "好吃"
word2 = "不好吃"

vector1 = get_word_vector(word1, W_in)
vector2 = get_word_vector(word2, W_in)

if vector1 is not None and vector2 is not None:
    similarity = cosine_similarity(vector1, vector2)
    print(f"'{word1}' and '{word2}' cosine similarity: {similarity}")

# 词向量聚类分析
# 准备词向量数据
words = list(W_in.keys())
vectors = np.array([W_in[word] for word in words])

# 使用 KMeans 进行聚类
num_clusters = 5
kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(vectors)
labels = kmeans.labels_

# 使用轮廓系数评估聚类效果
silhouette_avg = silhouette_score(vectors, labels)
print(f"轮廓系数: {silhouette_avg}")

# 使用 Calinski-Harabasz指数评估聚类效果
ch_score = calinski_harabasz_score(vectors, labels)
print(f"CH指数: {ch_score}")

# 计算并输出每个簇的属性
clusters = {i: [] for i in range(num_clusters)}
for word, label in zip(words, labels):
    clusters[label].append(word)

print("\n群集属性:")
for i in range(num_clusters):
    print(f"\nCluster {i}:")
    print(f"单词: {clusters[i]}")
    cluster_vectors = np.array([W_in[word] for word in clusters[i]])
    cluster_center = np.mean(cluster_vectors, axis=0)
    print(f"簇中心向量: {cluster_center}")
    most_common_words = Counter(clusters[i]).most_common(5)
    print(f"最常用的单词: {most_common_words}")

# 使用 PCA 降维到2D以便可视化
pca = PCA(n_components=2)
reduced_vectors = pca.fit_transform(vectors)

# 可视化聚类结果
plt.figure(figsize=(10, 10))
for i in range(num_clusters):
    cluster_points = reduced_vectors[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], label=f"Cluster {i}")
plt.legend()
plt.title('Word Clusters Visualization')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
