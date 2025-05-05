'''
情感分析+评论抓取+数据清洗
'''
import os

import torch
import json
import time
import requests
import pandas as pd
from pyecharts.charts import Map
from pyecharts.options import VisualMapOpts, TitleOpts, LabelOpts
from pyecharts.render import make_snapshot
from snapshot_selenium import snapshot as driver
import json
from collections import defaultdict
import re
from flask import Flask, request, jsonify, render_template
from io import StringIO
from Bert中文情感分析模型 import Predict_emo  # 默认导入深度学习的predict函数
from flask_cors import CORS



# 初始化 Flask 应用
app = Flask(__name__)
CORS(app)

cookie = "buvid3=1D80C9DF-869F-FCDB-9968-2D07CAB908B012906infoc; b_nut=1733368712; _uuid=E2E5FFD8-9644-771D-36109-583798110F71D13288infoc; enable_web_push=DISABLE; buvid4=8E227413-129D-186B-8B60-A6ADB398338514247-024120503-DKuZUl4GF69xRrUU6Pk8uw%3D%3D; buvid_fp=0ef70c256bb125590cc92880679d1264; rpdid=|(JYYRk)YJ|Y0J'u~JJJY)mlY; DedeUserID=667083136; DedeUserID__ckMd5=08b3143ab475d911; header_theme_version=CLOSE; CURRENT_QUALITY=80; hit-dyn-v2=1; bp_t_offset_667083136=1016666639258615808; SESSDATA=8cea49da%2C1751382296%2C5a444%2A11CjDtom5YcNuCjFLi-v1_S98rXMAkw14CGm8hkE0U-iIAl-xegDV0dTOsKihX7cDPVxkSVklBY2JJTFhkbEVTUi00enBmT1pvRi03TGpKYUt1bzYxaUZLcU15X2FMSDFMZ0tucE5Udm43N0ZMakgyelZMb2EtSGNyMWhsMVFldUl3RjdxQ3pZbVlRIIEC; bili_jct=3213005af38553f107e36dd757676149; sid=6lg2xhdx; CURRENT_FNVAL=4048; LIVE_BUVID=AUTO6017358835326818; bili_ticket=eyJhbGciOiJIUzI1NiIsImtpZCI6InMwMyIsInR5cCI6IkpXVCJ9.eyJleHAiOjE3MzYxNDMyNDgsImlhdCI6MTczNTg4Mzk4OCwicGx0IjotMX0.GE5kDGdZc0TQdIQ2lGDNQYNAf_tVlDjHeDJAg0pVVmE; bili_ticket_expires=1736143188; PVID=8; b_lsid=4A6FE811_1942CD8E1F6; bsource=search_bing; bmg_af_switch=1; bmg_src_def_domain=i0.hdslb.com; home_feed_column=4; browser_resolution=718-941"
# 设置请求头（模拟浏览器请求）
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest",
    "cookie": cookie
}

@app.route('/model-selection', methods=['POST'])
def model_selection():
    # 获取从前端发送的 JSON 数据
    data = request.get_json()
    model_index = data.get('index')

    # 根据 index 执行相应的逻辑
    print(f"模型索引接收到: {model_index}")
    if model_index == 1:
        from 支持向量机中文情感分析模型 import Predict_emo
        print("导入机器学习模型")
    # 不需要返回数据，仅做处理
    return '', 200  # 返回空响应，表示成功


# 时间差转换函数
def convert_to_hours(time_diff_str):
    # 定义单位转换关系
    unit_to_hours = {
        '分钟': 1 / 60,
        '小时': 1,
        '天': 24,
        '月': 24 * 30,  # 假设一个月有30天
        '年': 24 * 365  # 假设一年有365天
    }
    # 使用正则表达式提取数字和单位
    match = re.search(r'(\d+\.?\d*)\s*(分钟|小时|天|月|年)', time_diff_str)
    if match:
        value = float(match.group(1))  # 提取数字
        unit = match.group(2)  # 提取单位
        # 转换为小时
        return round(value * unit_to_hours.get(unit, 1), 1)  # 默认单位是小时, 保留一位小数
    else:
        return 0  # 如果没有匹配到，返回0

# 数据清洗函数
def clean_data(input_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    df = pd.DataFrame(data)
    # 1. 补齐“回复数”的缺失值为0
    df['回复数'] = df['回复数'].fillna(0)

    # 2. 处理“回复时间差”，保留数字部分
    df['回复时间差'] = df['回复时间差'].apply(lambda x: convert_to_hours(str(x)))

    # 3. 对“回复数”字段处理（提取数字）
    df['回复数'] = df['回复数'].apply(lambda x: int(re.findall(r'\d+', str(x))[0]) if isinstance(x, str) else x)

    # 4. 转换数据类型
    df['点赞数量'] = pd.to_numeric(df['点赞数量'], errors='coerce')
    df['用户当前等级'] = pd.to_numeric(df['用户当前等级'], errors='coerce')

    # 5. 处理缺失值
    df['性别'] = df['性别'].fillna('未知')
    df['评论内容'] = df['评论内容'].fillna('无评论')

    # 6. 性别列转换：男 -> 0，女 -> 1，保密 -> 2
    gender_map = {'男': 0, '女': 1, '保密': 2}
    df['性别'] = df['性别'].map(gender_map).fillna(2)  # 如果是“未知”或其他，默认设置为 2 (保密)

    # 7. 去除评论内容中的多余空格和特殊字符
    df['评论内容'] = df['评论内容'].apply(lambda x: re.sub(r'\s+', ' ', str(x)).strip())

    # 8. 将“回复时间”字段转换为datetime格式
    df['回复时间'] = pd.to_datetime(df['回复时间'], errors='coerce')

    # 9. 去除重复记录
    df = df.drop_duplicates()

    # 10. 对点赞数量进行分箱
    bins = [0, 50, 100, 200, 500, 1000, float('inf')]
    labels = ['低', '中', '较高', '高', '非常高', '极高']
    df['点赞数量_分类'] = pd.cut(df['点赞数量'], bins=bins, labels=labels)

    # 将处理后的DataFrame转换回JSON格式
    cleaned_data = df.to_json(orient='records', force_ascii=False, date_format='iso')

    # 保存为JSON文件
    output_file = 'cleaned_data.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_data)

    return output_file

# 获取评论的函数
def fetch_comments(video_id, max_pages=5):  # 最大页面数量可调整
    comments = []
    last_count = 0
    for page in range(1, max_pages + 1):
        url = f'https://api.bilibili.com/x/v2/reply/main?next={page}&type=1&oid={video_id}&mode=3'
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                if data['data']['replies'] == None:
                    break
                if data and 'replies' in data['data']:
                    for comment in data['data']['replies']:
                        comment_info = {
                            '用户昵称': comment['member']['uname'],
                            '评论内容': comment['content']['message'],
                            '性别': comment['member']['sex'],
                            '用户当前等级': comment['member']['level_info']['current_level'],
                            '点赞数量': comment['like'],
                            'IP属地': comment['reply_control']['location'].split("：")[1],
                            '回复数': comment['reply_control'].get('sub_reply_entry_text', None),
                            '回复时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comment['ctime'])),
                            '回复时间差': comment['reply_control']['time_desc'],
                        }
                        comments.append(comment_info)

                if last_count == len(comments):
                    break
                last_count = len(comments)
            else:
                break
        except requests.RequestException as e:
            print(f"请求出错: {e}")
            break
        time.sleep(1)
    return comments
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
# 处理用户上传的csv文件
def process_comments(file):
    # 尝试使用多种编码格式读取文件
    raw_data = file.stream.read()  # 读取文件内容

    # 尝试不同的编码格式读取
    try:
        df = pd.read_csv(StringIO(raw_data.decode('utf-8')))  # 尝试utf-8编码
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(StringIO(raw_data.decode('gbk')))  # 如果utf-8失败，尝试gbk编码
        except UnicodeDecodeError:
            try:
                df = pd.read_csv(StringIO(raw_data.decode('ascii')))  # 如果gbk失败，尝试ascii编码
            except UnicodeDecodeError:
                # 如果ascii也失败，使用ISO-8859-1
                df = pd.read_csv(StringIO(raw_data.decode('ISO-8859-1')))
    # 初始化计数器
    positive_count = 0
    negative_count = 0
    # 遍历评论列进行情感分析
    for sentence in df['评论']:
        # 获取预测结果
        predicted_class, predicted_score = Predict_emo(sentence)
        print(sentence,"预测结果"+predicted_class,predicted_score)
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
# 路由：情感分析页面
@app.route('/', methods=['GET', 'POST'])
def index():
    # if request.method == 'POST':
    #     text = request.form['text']
    #     result = predict_sentiment(text)
    #     positive_prob = result[0][1].item()  # 获取正向情感的概率
    #     return render_template('index_v2.html', result=positive_prob)
    return render_template('index_v11.html', result=None)

# 路由：评论抓取
@app.route('/fetch_comments', methods=['POST'])
def export_comments():
    video_url = request.json.get('video_url')
    if not video_url:
        return {'error': 'No video URL provided'}, 400

    video_id = video_url.split("/")[4]
    print(video_id)
    comments = fetch_comments(video_id)
    if not comments:
        return {'error': 'No comments found for this video'}, 404

    with open("comments.json", "w", encoding="utf-8") as f:
        json.dump(comments, f, ensure_ascii=False, indent=4)
    return jsonify(comments)

# 路由：数据清洗
@app.route('/clean_data', methods=['POST'])
def clean_and_export_data():
    input_file = 'comments.json'
    # 调用数据清洗函数
    output_file = clean_data(input_file)

    # 返回清洗后的数据
    with open(output_file, 'r', encoding='utf-8') as f:
        cleaned_comments = json.load(f)
    return jsonify(cleaned_comments)
#路由：数据探索性分析
@app.route("/endpoint",methods=['POST'])
def keshihua_data():
    import seaborn as sns
    import pandas as pd
    import jieba
    from collections import Counter
    import matplotlib.pyplot as plt
    print("开始绘图...")
    plt.switch_backend('Agg')
    # from win32comext.shell.demos.IActiveDesktop import opts
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 步骤一（替换sans-serif字体）
    plt.rcParams['axes.unicode_minus'] = False  # 步骤二（解决坐标轴负数的负号显示问题）
    # sns.set(font='SimHei',font_scale=1.5)
    # 读取JSON文件
    df = pd.read_json('cleaned_data.json')
    # '''
    # 展示图1：统计性别占比
    # '''
    gender_count = df['性别'].value_counts()
    # 定义淡色系的颜色，呈现青春活力感
    colors = ['#FFB3C1', '#A3D3FF', '#B5E8B2']  # 淡粉色、淡蓝色和淡绿色
    # 绘制饼状图
    plt.figure(figsize=(10, 6))
    # 创建饼状图
    plt.pie(gender_count, labels=['男', '女', '保密'], autopct='%1.1f%%', startangle=90, colors=colors,
            wedgeprops={'edgecolor': 'black', 'linewidth': 1, 'linestyle': 'solid'})
    # 设置标题
    plt.title('性别占比', fontsize=14, fontweight='bold')
    # 保证饼图是圆形的
    plt.axis('equal')
    # 显示图表
    plt.savefig('static/Picture/fig1.png', dpi=600)
    plt.close()
    # '''
    # 展示图2：点赞分类量与用户等级之间的关系
    # '''
    plt.figure(figsize=(10, 6))
    sns.countplot(x='用户当前等级', hue='点赞数量_分类', data=df, palette='Set3')
    plt.title('点赞量分类与用户等级的关系')
    plt.xlabel('用户等级')
    plt.ylabel('数量')
    plt.savefig('static/Picture/fig2.png', dpi=600)
    plt.close()
    # '''
    # 展示图3：随小时分布的评论数
    # '''
    df['回复时间'] = pd.to_datetime(df['回复时间'])
    # 提取小时
    df['小时'] = df['回复时间'].dt.hour
    # 统计每小时的评论数量
    hourly_comments = df['小时'].value_counts().sort_index()
    # 设置颜色为淡色系
    line_color = '#a8c9e5'  # 淡蓝色
    background_color = '#f4f7f9'  # 背景浅灰色
    # 可视化每小时评论数量分布
    plt.figure(figsize=(10, 6), facecolor=background_color)
    # 绘制折线图
    hourly_comments.plot(kind='line', marker='o', linestyle='-', color=line_color, linewidth=2)
    # 设置标题和标签
    plt.title('每小时评论数量分布', fontsize=16, fontweight='bold', color='#333333')
    plt.xlabel('小时', fontsize=14, color='#666666')
    plt.ylabel('评论数量', fontsize=14, color='#666666')
    # 设置x轴和y轴刻度的字体大小
    plt.xticks(range(24), fontsize=12, color='#666666')
    plt.yticks(fontsize=12, color='#666666')
    # 添加网格线，设置透明度和线条样式
    plt.grid(True, linestyle='--', color='gray', alpha=0.5)
    # 显示图表
    plt.tight_layout()
    plt.savefig('static/Picture/fig3.png', dpi=600)
    plt.close()
    '''
    展示图4：词云图展示
    '''
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    # 提取所有评论内容字段
    comments = df['评论内容'].dropna().str.cat(sep=' ')
    # 创建一个词云对象
    wordcloud = WordCloud(
        font_path='msyh.ttc',  # 使用支持中文的字体路径，确保中文可以显示
        width=800,  # 词云的宽度
        height=600,  # 词云的高度
        background_color='white',  # 背景色
        max_words=200,  # 词云中最多显示的词汇数
        max_font_size=100,  # 最大字体大小
        collocations=False  # 禁用词语搭配（防止“高频词”组合）
    ).generate(comments)
    # 显示词云图
    plt.figure(figsize=(10, 8))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')  # 关闭坐标轴
    plt.savefig('static/Picture/fig4.png', dpi=600)
    plt.close()
    '''
    展示图5：最常见的前20个词语
    '''
    # 提取评论内容
    comments = df['评论内容'].dropna().str.cat(sep=' ')
    # 使用结巴进行分词
    words = jieba.lcut(comments)
    # 自定义停用词列表（可以根据需要扩展）
    stopwords = ['的', '是', '了', '在', '和', '有', '也', '就', '都', '很', '与', '吗', '啊', '哦', '呀', '呢', '吧']
    # 过滤停用词
    filtered_words = [word for word in words if word not in stopwords and len(word) > 1]
    # 统计词频
    word_count = Counter(filtered_words)
    # 获取前20个高频词
    most_common_words = word_count.most_common(20)
    words, counts = zip(*most_common_words)
    # 可视化
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), palette='pink')
    plt.title('评论中最常见的20个词语')
    plt.xlabel('频率')
    plt.ylabel('词语')
    plt.grid(axis='x', linestyle='--', alpha=0.6)
    plt.savefig('static/Picture/fig5.png', dpi=600)
    plt.close()
    '''
    #展示图6：IP地址分布
    '''

    # 省份转换函数
    def province(pro):
        provinces = ["北京市", "天津市", "河北省", "山西省", "内蒙古自治区", "辽宁省", "吉林省", "黑龙江省", "上海市",
                     "江苏省", "浙江省", "安徽省", "福建省", "江西省", "山东省", "河南省", "湖北省", "湖南省",
                     "广东省", "广西壮族自治区", "海南省", "重庆市", "四川省", "贵州省", "云南省", "西藏自治区",
                     "陕西省", "甘肃省", "青海省", "宁夏回族自治区", "新疆维吾尔自治区", "台湾省", "香港特别行政区",
                     "澳门特别行政区"]

        for p in provinces:
            if pro in p:
                return p  # 返回省份前两位
        return None  # 如果没有匹配到，返回 None

    # 读取评论数据
    with open('cleaned_data.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
    # 统计各省份出现次数
    province_count = defaultdict(int)

    # 遍历数据，统计每个省份出现的次数
    for entry in data:
        ip_location = entry.get("IP属地", "")
        if ip_location:  # 确保IP属地存在
            province_name = province(ip_location)  # 获取规范的省份名称
            if province_name:
                province_count[province_name] += 1

    # 输出统计结果（可选）
    print(province_count)

    # 将统计结果转化为绘图所需的格式
    data_list = [(province_name, count) for province_name, count in province_count.items()]

    # 创建地图对象
    map = Map()  # 构建地图对象
    map.add("IP属地统计", data_list, "china", is_map_symbol_show=False)  # 设置不显示省份名称
    # 设置全局选项
    map.set_global_opts(
        visualmap_opts=VisualMapOpts(
            is_show=True,  # 是否显示
            is_piecewise=True,  # 是否分段
            pieces=[
                {"min": 1, "max": 5, "label": "1~5人", "color": "#B2D8FF"},
                {"min": 6, "max": 10, "label": "6~10人", "color": "#72B6FF"},
                {"min": 11, "max": 15, "label": "11~15人", "color": "#4099FF"},
                {"min": 16, "max": 20, "label": "16~20人", "color": "#0066CC"},
                {"min": 21, "max": 25, "label": "21~25人", "color": "#003366"},
                {"min": 26, "label": "26+", "color": "#001933"},
            ],
            pos_top="20%",  # Move the legend higher (you can adjust the percentage)
            pos_left="80%",
        )
    )
    # 去掉省份名称的标签
    map.set_series_opts(label_opts=LabelOpts(is_show=False))
    # 渲染到PNG文件
    make_snapshot(driver, map.render(), "static/Picture/fig6.png")
    time.sleep(10)
    # picture_folder = 'static/Picture'
    # # 获取文件夹中的所有文件
    # for filename in os.listdir(picture_folder):
    #     # 检查文件是否为图片文件（根据扩展名判断）
    #     if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
    #         # 拼接出文件的完整路径
    #         file_path = os.path.join(picture_folder, filename)
    #
    #         # 打开图片
    #         with Image.open(file_path) as img:
    #             # 获取原始DPI
    #             dpi = img.info.get('dpi', (72, 72))  # 默认DPI为72，如果没有DPI信息
    #
    #             # 调整图片大小为600x360，使用LANCZOS滤波器（即抗锯齿）
    #             resized_img = img.resize((60, 36), Image.Resampling.LANCZOS)
    #
    #             # 设置调整后图片的DPI保持不变
    #             resized_img.save(file_path, dpi=dpi)
    #         print(f"图片 {filename} 已调整为 600x360 大小，DPI保持不变")
    is_suss=True
    return jsonify(is_suss)
# 路由：情感分析 API 接口
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # 获取前端传来的 JSON 数据
        data = request.get_json()
        text = data.get('text1', None)
        # 检查是否有文本数据
        if not text:
            return jsonify({"error": "Text is required"}), 400
        # # 使用模型进行预测
        result = Predict_emo(text)[1]
        # 返回预测结果
        return jsonify({"result": result*2-1 })  # 正向情感的概率
    except Exception as e:
        return jsonify({"error": str(e)}), 500



# 路由：对爬虫下来的评论进行批量情感分析
@app.route('/analyze_json', methods=['POST'])
def analyze_json():
    # 进行情感分析
    result = process_json()
    print(result)
    # 返回JSON响应
    return jsonify(result)

# 路由：对用户上传的文件进行批量情感分析
@app.route('/analyze', methods=['POST'])
def analyze():
    # 获取上传的 CSV 文件
    file = request.files['file']
    # 进行情感分析
    result = process_comments(file)

    # 返回JSON响应
    return jsonify(result)

# 运行 Flask 应用
if __name__ == '__main__':
    app.run(debug=True)
