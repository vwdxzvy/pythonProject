import jieba

# 读取处理后的语料库文本
with open("processed_corpus.txt", "r", encoding="utf-8") as f:
    corpus_text = f.read()

# 使用jieba库进行分词
seg_list = jieba.cut(corpus_text)

# 将分词结果保存为新的文本文件
with open("segmented_corpus.txt", "w", encoding="utf-8") as f:
    f.write(" ".join(seg_list))
