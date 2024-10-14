from flask import Flask, render_template, request, jsonify
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

app = Flask(__name__)


# 加载数据集并初始化向量化器和LSA模型
newsgroups = fetch_20newsgroups(subset='all')
vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, stop_words='english')
X = vectorizer.fit_transform(newsgroups.data)

# 使用SVD进行降维
svd_model = TruncatedSVD(n_components=100)
lsa_matrix = svd_model.fit_transform(X)


def search_engine(query):
    # 处理用户查询
    query_vector = vectorizer.transform([query])
    query_lsa = svd_model.transform(query_vector)

    # 计算查询和文档的余弦相似度
    similarities = cosine_similarity(query_lsa, lsa_matrix)[0]

    # 获取前5个最相似的文档
    top_indices = similarities.argsort()[-5:][::-1]
    top_documents = [newsgroups.data[i] for i in top_indices]
    top_similarities = [similarities[i] for i in top_indices]

    return top_documents, top_similarities, top_indices.tolist()  # 转换 indices 为列表


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/search', methods=['POST'])
def search():
    query = request.form['query']
    documents, similarities, indices = search_engine(query)
    return jsonify({'documents': documents, 'similarities': similarities, 'indices': indices}) 

if __name__ == '__main__':
    app.run(debug=True)
