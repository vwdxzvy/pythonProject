import os
import re
import jieba

# 设置语料库文件路径
corpus_file = "语料库.txt"

# 读取语料库文件
with open(corpus_file, "r", encoding="utf-8") as f:
    corpus_text = f.read()

# 预处理：去除标点符号、特殊字符等
corpus_text = re.sub(r'[^\w\s]', '', corpus_text)  # 仅保留字母、数字和空白字符

# 分词
words = jieba.cut(corpus_text)

# 将处理后的结果保存为文本文件
with open("processed_corpus.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(words))
