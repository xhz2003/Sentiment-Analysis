import json
from flask import Flask, jsonify, render_template
from flask_cors import CORS

from Bert中文情感分析模型 import Predict_emo  # 导入情感分析模型

app = Flask(__name__)
CORS(app)  # 启用 CORS，允许跨域请求

# 读取并处理JSON文件
def process_json():
    # 读取本地的clean_data.json文件
    try:
        with open('cleaned_data.json', 'r', encoding='utf-8') as file:
            data = json.load(file)  # 将JSON字符串转为字典/列表
    except Exception as e:
        return {"error": "读取文件失败", "message": str(e)}

    # 初始化计数器
    positive_count = 0
    negative_count = 0

    # 遍历评论内容并进行情感分析
    for item in data:
        # 获取“评论内容”字段
        comment = item.get('评论内容', '')
        if comment:  # 如果评论内容不为空
            predicted_class, predicted_score = Predict_emo(comment)

            # 判断评论是正向还是负向
            if predicted_class == '正面':
                positive_count += 1
            elif predicted_class == '负面':
                negative_count += 1

    # 返回分析结果
    return {
        "positive_count": positive_count,
        "negative_count": negative_count
    }


# 路由：批量情感分析
@app.route('/analyze_json', methods=['POST'])
def analyze_json():
    # 进行情感分析
    result = process_json()

    # 返回JSON响应
    return jsonify(result)


# 路由：主页
@app.route('/')
def index():
    return render_template('CS2.html')


if __name__ == '__main__':
    app.run(debug=True)
