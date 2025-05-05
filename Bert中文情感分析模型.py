# import torch
# from transformers import BertTokenizer
#
# # 快速演示
# device = 'cuda' if torch.cuda.is_available() else 'cpu'
# print('device=', device)
#
# from transformers import BertModel
#
# # 加载预训练模型
# pretrained = BertModel.from_pretrained('./bert/bert-base-chinese')
# tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-chinese')
#
# # 需要移动到cuda上
# pretrained.to(device)
#
# # 不训练,不需要计算梯度
# for param in pretrained.parameters():
#     param.requires_grad_(False)
#
# # 定义下游任务模型
# class Model(torch.nn.Module):
#
#     def __init__(self):
#         super().__init__()
#         self.fc = torch.nn.Linear(768, 2)
#
#     def forward(self, input_ids, attention_mask, token_type_ids):
#         with torch.no_grad():
#             out = pretrained(input_ids=input_ids,
#                              attention_mask=attention_mask,
#                              token_type_ids=token_type_ids)
#
#         out = self.fc(out.last_hidden_state[:, 0])
#         out = out.softmax(dim=1)
#         return out
#
# # 创建模型实例
# model = Model()
#
# # 使用 map_location 来确保在没有 CUDA 的设备上加载模型
# model.load_state_dict(torch.load('./bert/model_weights.pth', map_location=device))
#
# # 将模型移动到设备
# model.to(device)
#
# # 设置模型为评估模式
# model.eval()
#
# # 测试
# def test():
#     model.eval()
#     correct = 0
#     total = 0
#     sentence = "今天考试拿到了满分，老师奖励我棒棒糖，嘿嘿"
#
#     # Tokenize and encode the sentence
#     inputs = tokenizer.encode_plus(
#         sentence,  # Text to be tokenized
#         add_special_tokens=True,  # Add [CLS] and [SEP] tokens
#         max_length=500,  # Max length for padding/truncation
#         padding='max_length',  # Pad to the max length
#         truncation=True,  # Truncate if text is longer than MAX_LEN
#         return_tensors='pt'  # Return as PyTorch tensors
#     )
#
#     # Get input tensors
#     input_ids = inputs['input_ids'].to(device)  # Move to GPU if necessary
#     attention_mask = inputs['attention_mask'].to(device)  # Move to GPU if necessary
#     token_type_ids = inputs['token_type_ids'].to(device)  # Move to GPU if necessary
#
#     with torch.no_grad():
#         out = model(input_ids=input_ids,
#                     attention_mask=attention_mask,
#                     token_type_ids=token_type_ids)
#     classes = ["负面", "正面"]
#     print(out[0][1].item())
#     print(classes[out.argmax(dim=1).item()])
# test()



# predict.py
import torch
from transformers import BertTokenizer, BertModel
# 设置设备
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载预训练模型和tokenizer
pretrained = BertModel.from_pretrained('./bert/bert-base-chinese')
tokenizer = BertTokenizer.from_pretrained('./bert/bert-base-chinese')

# 将模型移到指定设备
pretrained.to(device)

# 不训练模型
for param in pretrained.parameters():
    param.requires_grad_(False)
# 定义下游任务模型
class Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(768, 2)
    def forward(self, input_ids, attention_mask, token_type_ids):
        with torch.no_grad():
            out = pretrained(input_ids=input_ids,
                             attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        out = self.fc(out.last_hidden_state[:, 0])
        out = out.softmax(dim=1)
        return out


# 创建模型实例
model = Model()

# 加载保存的权重
model.load_state_dict(torch.load('./bert/model_weights.pth', map_location=device))

# 将模型移动到设备
model.to(device)

# 设置模型为评估模式
model.eval()


def Predict_emo(sentence: str) -> tuple:
    """
    使用训练好的模型进行文本分类，返回预测结果。
    :param sentence: 输入文本
    :return: 预测类别（"负面" 或 "正面"）
    """
    # Tokenize and encode the sentence
    inputs = tokenizer.encode_plus(
        sentence,  # 需要分类的文本
        add_special_tokens=True,  # 添加 [CLS] 和 [SEP] token
        max_length=500,  # 最大长度
        padding='max_length',  # 填充到最大长度
        truncation=True,  # 截断长于最大长度的文本
        return_tensors='pt'  # 返回 PyTorch 张量
    )

    # 获取输入的张量
    input_ids = inputs['input_ids'].to(device)
    attention_mask = inputs['attention_mask'].to(device)
    token_type_ids = inputs['token_type_ids'].to(device)

    # 获取模型输出
    with torch.no_grad():
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    token_type_ids=token_type_ids)

    # 类别标签
    classes = ["负面", "正面"]
    # 输出预测结果
    predicted_class = classes[out.argmax(dim=1).item()]
    predicted_score = out[0][1].item()  # 获取"正面"的概率
    return predicted_class, predicted_score
