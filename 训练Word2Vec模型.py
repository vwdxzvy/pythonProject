from collections import defaultdict
import numpy as np


# 定义一个函数来计算余弦相似度
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    return dot_product / (norm1 * norm2)


# 读取分词后的文本文件，构建句子列表
sentences = []
word_count = defaultdict(int)
with open("segmented_corpus.txt", "r", encoding="utf-8") as f:
    for line in f:
        words = line.strip().split()
        sentences.append(words)
        for word in words:
            word_count[word] += 1

# 设置超参数
window_size = 5  # 窗口大小
vector_size = 100  # 向量维度
min_count = 5  # 最小词频

# 动态词汇表
word_to_index = {}
index_to_word = {}
index_counter = 0

# 初始化动态权重矩阵字典
W_in = {}
W_out = {}


def get_vector(word, vector_size, matrix):
    if word not in matrix:
        matrix[word] = np.random.uniform(-0.5 / vector_size, 0.5 / vector_size, vector_size)
    return matrix[word]


# 训练 Word2Vec 模型
for sentence in sentences:
    for center_word_index, center_word in enumerate(sentence):
        if word_count[center_word] < min_count:
            continue  # 跳过频率低于 min_count 的词汇

        if center_word not in word_to_index:
            word_to_index[center_word] = index_counter
            index_to_word[index_counter] = center_word
            index_counter += 1

        center_word_vector = get_vector(center_word, vector_size, W_in)

        context_indices = [i for i in range(max(0, center_word_index - window_size),
                                            min(len(sentence), center_word_index + window_size + 1))
                           if i != center_word_index]

        for context_index in context_indices:
            context_word = sentence[context_index]
            if word_count[context_word] < min_count:
                continue  # 跳过频率低于 min_count 的词汇

            if context_word not in word_to_index:
                word_to_index[context_word] = index_counter
                index_to_word[index_counter] = context_word
                index_counter += 1

            context_word_vector = get_vector(context_word, vector_size, W_out)

            # 计算预测向量和误差
            predicted_vector = np.dot(center_word_vector, context_word_vector)
            error = predicted_vector - 1  # 目标是 context_word_vector

            # 更新权重矩阵
            W_out[context_word] -= 0.01 * error * center_word_vector
            W_in[center_word] -= 0.01 * error * context_word_vector

# 保存训练好的模型参数
np.save("W_in.npy", W_in)
np.save("W_out.npy", W_out)
