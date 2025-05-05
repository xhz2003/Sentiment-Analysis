# import jieba
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.svm import SVC
# from sklearn.metrics import accuracy_score, classification_report
#
# # 1. 数据加载
# # 读取评论数据
# # data = pd.read_csv('./Dataset/谭松波酒店情感分析语料.txt')
# # 使用'gbk'编码读取文件
# data = pd.read_csv('./Dataset/谭松波酒店情感分析语料.txt', encoding='gbk')
# # 删除包含NaN值的行
# data = data.dropna(subset=['review'])
# # 查看数据格式
# print(data.head())
#
# # 2. 分词处理
# def chinese_tokenizer(text):
#     return jieba.lcut(text)  # 使用jieba进行中文分词
#
# # 3. 特征提取
# # 使用TF-IDF对文本进行向量化
# tfidf_vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=None)
# X = tfidf_vectorizer.fit_transform(data['review'])
#
# # 4. 划分训练集和测试集
# y = data['label']
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
#
# # 5. 训练支持向量机（SVM）模型
# # 使用SVC进行情感分析
# model = SVC(kernel='linear')  # 使用线性核
# model.fit(X_train, y_train)
#
# # 6. 预测
# y_pred = model.predict(X_test)
#
# # 7. 评估模型
# print(f'准确率: {accuracy_score(y_test, y_pred)}')
# print('分类报告:')
# print(classification_report(y_test, y_pred))
#
# # 8. 使用模型进行新评论预测
# def predict_sentiment(comment):
#     comment_tfidf = tfidf_vectorizer.transform([comment])
#     prediction = model.predict(comment_tfidf)
#     return '正面' if prediction == 1 else '负面'
#
# # 示例：预测一个新评论
# new_comment = "这次住的酒店非常满意，服务很好，房间干净"
# print(f'评论情感预测: {predict_sentiment(new_comment)}')


import jieba
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib  # 用于保存和加载模型


# 1. 定义分词函数
def chinese_tokenizer(text):
    return jieba.lcut(text)  # 使用jieba进行中文分词


def train():
    # 1. 数据加载（指定编码格式）
    data = pd.read_csv('./Dataset/谭松波酒店情感分析语料.txt', encoding='gbk')  # 如果'gbk'不行，试试'latin1'
    # 删除包含NaN值的行
    data = data.dropna(subset=['review'])
    # 查看数据格式
    print(data.head())
    # 2. 特征提取
    # 使用TF-IDF对文本进行向量化
    tfidf_vectorizer = TfidfVectorizer(tokenizer=chinese_tokenizer, stop_words=None)
    X = tfidf_vectorizer.fit_transform(data['review'])
    # 3. 划分训练集和测试集
    y = data['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    # 4. 训练支持向量机（SVM）模型
    model = SVC(kernel='linear', probability=True)  # 使用线性核并开启概率预测
    model.fit(X_train, y_train)
    # 5. 保存模型
    joblib.dump(model, './SVM/svm_sentiment_model.pkl')
    joblib.dump(tfidf_vectorizer, './SVM/tfidf_vectorizer.pkl')
    print("模型和TF-IDF向量化器已保存！")
    # 6. 评估模型
    y_pred = model.predict(X_test)
    print(f'准确率: {accuracy_score(y_test, y_pred)}')
    print('分类报告:')
    print(classification_report(y_test, y_pred))


# 7. 加载模型和向量化器
def load_model():
    # 加载模型和TF-IDF向量化器
    model = joblib.load('./SVM/svm_sentiment_model.pkl')
    tfidf_vectorizer = joblib.load('./SVM/tfidf_vectorizer.pkl')
    return model, tfidf_vectorizer


# 8. 预测函数
def predict_sentiment(comment):
    """
    预测评论的情感类别及其概率

    :param comment: 输入评论文本
    :return: (predicted_class, predicted_score) 预测的类别和预测"正面"的概率
    """
    # 加载已保存的模型和TF-IDF向量化器
    model, tfidf_vectorizer = load_model()

    # 将评论转换为TF-IDF特征
    comment_tfidf = tfidf_vectorizer.transform([comment])

    # 使用模型进行预测
    out = model.predict_proba(comment_tfidf)  # 获取概率值

    # 类别标签
    classes = ["负面", "正面"]

    # 输出预测结果
    predicted_class = classes[out.argmax(axis=1).item()]  # 获取预测类别
    predicted_score = out[0][1].item()  # 获取"正面"的概率

    return predicted_class, predicted_score


data = pd.read_csv('./Dataset/谭松波酒店情感分析语料.txt', encoding='gbk')  # 如果'gbk'不行，试试'latin1'
print(len(data))
    # 示例：预测一个新评论
print("成功导入机器学习模型")
predicted_class, predicted_score = predict_sentiment("今天天气真好")
print(f'评论情感预测: {predicted_class}')
print(f'预测“正面”的概率: {predicted_score}')
