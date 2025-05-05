import seaborn as sns
import pandas as pd
import jieba
from collections import Counter
import matplotlib.pyplot as plt
# from win32comext.shell.demos.IActiveDesktop import opts
plt.rcParams['font.sans-serif'] = ['SimHei'] # 步骤一（替换sans-serif字体）
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

# '''
# 展示图2：点赞分类量与用户等级之间的关系
# '''
plt.figure(figsize=(10, 6))
sns.countplot(x='用户当前等级', hue='点赞数量_分类', data=df, palette='Set3')
plt.title('点赞量分类与用户等级的关系')
plt.xlabel('用户等级')
plt.ylabel('数量')
plt.savefig('static/Picture/fig2.png', dpi=600)

# '''
# 展示图3：随小时分布的评论数
# '''
# 将回复时间转换为datetime格式
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
    width=800,             # 词云的宽度
    height=600,            # 词云的高度
    background_color='white',  # 背景色
    max_words=200,         # 词云中最多显示的词汇数
    max_font_size=100,     # 最大字体大小
    collocations=False     # 禁用词语搭配（防止“高频词”组合）
).generate(comments)
# 显示词云图
plt.figure(figsize=(10, 8))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')  # 关闭坐标轴
plt.savefig('static/Picture/fig4.png', dpi=600)
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
'''
#展示图6：IP地址分布
'''
# 假设这是你从数据中统计出来的IP属地的评论数量
IP_count = df['IP属地'].value_counts()

# 转换为DataFrame
import pandas as pd
IP_count_df = pd.DataFrame(list(IP_count.items()), columns=['IP属地', '评论数量'])

# 按评论数量降序排序
IP_count_df = IP_count_df.sort_values(by='评论数量', ascending=True)

# 绘制条形图
plt.figure(figsize=(10, 6))
plt.barh(IP_count_df['IP属地'], IP_count_df['评论数量'], color='skyblue')

# 设置标题和标签
plt.title("各IP属地评论数量", fontsize=16)
plt.xlabel("评论数量", fontsize=12)
plt.ylabel("IP属地", fontsize=12)

# 显示图表
plt.tight_layout()
plt.savefig('static/Picture/fig6.png', dpi=600)

