# import requests
# import json
# import datetime
#
# # 视频ID（从视频链接中提取）
# # video_id = "BV1wZ421e7Fr"
# #前端只需要传入这个视频地址即可
# video_url = "https://www.bilibili.com/video/BV1KJzbYcEFo/?spm_id_from=333.1007.tianma.1-2-2.click&vd_source=1cb6a173fe0aa49ea45009f6f5263c90"
# video_id = video_url.split("/")[4]
# # B站评论API的URL，使用video_id来拼接
# url = f"https://api.bilibili.com/x/v2/reply/main?next=0&type=1&oid={video_id}"
# # 设置请求头（模拟浏览器请求）
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
#     "Accept": "application/json, text/javascript, */*; q=0.01",
#     "X-Requested-With": "XMLHttpRequest"
# }
# import requests
# import time
# import csv
# def fetch_comments(video_id, max_pages=15):  # 最大页面数量可调整
#     comments = []
#     last_count = 0
#     for page in range(1, max_pages + 1):
#         url = f'https://api.bilibili.com/x/v2/reply/main?next=1&type=1&oid={video_id}&mode=3'
#         try:
#             # 添加超时设置
#             response = requests.get(url, headers=headers, timeout=10)
#             if response.status_code == 200:
#                 data = response.json()
#                 print(page)
#                 if data['data']['replies'] == None:
#                     break
#                 if data and 'replies' in data['data']:
#                     for comment in data['data']['replies']:
#                         comment_info = {
#                             '用户昵称': comment['member']['uname'],
#                             '评论内容': comment['content']['message'],
#                             '性别': comment['member']['sex'],
#                             '用户当前等级': comment['member']['level_info']['current_level'],
#                             '点赞数量': comment['like'],
#                             '回复数': comment['reply_control'].get('sub_reply_entry_text', None),
#                             '回复时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comment['ctime'])),
#                             '回复时间差': comment['reply_control']['time_desc']
#                         }
#                         comments.append(comment_info)
#                 if last_count == len(comments):
#                     break
#                 last_count = len(comments)
#             else:
#                 break
#         except requests.RequestException as e:
#             print(f"请求出错: {e}")
#             break
#         # 控制请求频率
#         time.sleep(1)
#     return comments
#
#
# def save_comments_to_csv(comments, video_bv):
#     with open(f'{video_bv}.csv', mode='w', encoding='ANSI',
#               newline='') as file:
#         writer = csv.DictWriter(file,
#                                 fieldnames=['用户昵称', '性别', '评论内容', '用户当前等级',
#                                             '点赞数量', '回复数','回复时间',"回复时间差"])
#         writer.writeheader()
#         for comment in comments:
#             writer.writerow(comment)
#             print(comment)
# video_name = 'video'  # 视频名字
# comments = fetch_comments(video_id)
# save_comments_to_csv(comments, video_name)  # 会将所有评论保存到一个csv文件

import requests
import json
import datetime
import time

# 视频ID（从视频链接中提取）
video_url = "https://www.bilibili.com/video/BV1KJzbYcEFo/?spm_id_from=333.1007.tianma.1-2-2.click&vd_source=1cb6a173fe0aa49ea45009f6f5263c90"
video_id = video_url.split("/")[4]

# B站评论API的URL，使用video_id来拼接
url = f"https://api.bilibili.com/x/v2/reply/main?next=0&type=1&oid={video_id}"
# 设置请求头（模拟浏览器请求）
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36 Edg/131.0.0.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "X-Requested-With": "XMLHttpRequest"
}

def fetch_comments(video_id, max_pages=15):  # 最大页面数量可调整
    comments = []
    last_count = 0
    for page in range(1, max_pages + 1):
        url = f'https://api.bilibili.com/x/v2/reply/main?next=1&type=1&oid={video_id}&mode=3'
        try:
            # 添加超时设置
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                data = response.json()
                print(page)
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
                            '回复数': comment['reply_control'].get('sub_reply_entry_text', None),
                            '回复时间': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(comment['ctime'])),
                            '回复时间差': comment['reply_control']['time_desc']
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
        # 控制请求频率
        time.sleep(1)
    return comments

def save_comments_to_json(comments, video_bv):
    with open(f'{video_bv}.json', 'w', encoding='utf-8') as file:
        json.dump(comments, file, ensure_ascii=False, indent=4)

video_name = 'video'  # 视频名字
comments = fetch_comments(video_id)
save_comments_to_json(comments, video_name)  # 会将所有评论保存到一个json文件