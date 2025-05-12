import json
from torch.utils.data import Dataset
import random
random.seed(2024)

# 定义模板
TEMPLATES = {
    "QA": (
        "Below is a question:\n"
        "{question}\n\n"
        "Below are related passages:\n"
        "{reference}\n\n"
        "Your task is to answer the question strictly based on the related passages.\n"
        "In case the passages do not contain the necessary information to answer the question, please reply with: 'Unable to answer based on given passages.'\n"
        "If you cannot answer the question precisely, please reply with: 'Sorry, this question is beyond my ability.'\n"
        "Output:"
    ),
    "Summary": (
        "Below are some news\n"
        "{reference}\n\n"
        "Your task is to write a summary of the news.\n"
        "If you cannot summarize the news precisely, please reply with: 'Sorry, this question is beyond my ability.'\n"
        "Output:"
    ),
    "Data2txt": (
        "Your task is to write an objective overview about the following local business based only on the provided structured data in the JSON format. \
            You should include details and cover the information mentioned in the customers' review. The overview should be 100 - 200 words. Don't make up information.\n"
        "If you cannot summarize the data precisely, please reply with: 'Sorry, this question is beyond my ability.'\n"
        "Below are the structured data:\n"
        "{reference}\n\n"
        "Output:"
    )
}

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

def process_dialog(data, tokenizer):
    question = data['question']
    reference = data['reference']
    response = data['chosen']
    task = data['task_type']

    if task=='QA':
        prompt = TEMPLATES["QA"].format(question=question, reference=reference)
    else:
        prompt = TEMPLATES[task].format(reference=reference)
    
    inputs = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False, truncation=True) + [tokenizer.eos_token_id]
    
    # 生成标签
    labels = tokenizer.encode(response, add_special_tokens=False, truncation=True)
    labels = labels + [tokenizer.eos_token_id]  # 追加结束标记
    
    # 确保标签的形状与输入相同
    input_length = len(inputs)
    labels_length = len(labels)
    
    if labels_length < input_length:
        labels = labels + [tokenizer.pad_token_id] * (input_length - labels_length)  # 填充标签
    elif labels_length > input_length:
        labels = labels[:input_length]  # 截断标签

    return inputs, labels

class CaseDetectDataset(Dataset):
    def __init__(self, tokenizer, args, train=True):
        self.ann = []
        with open(args.train_file if train else args.eval_file, "r") as f:
            for line in f:
                d = json.loads(line)
                self.ann.append(d)

        self.train = train
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):
        ann = self.ann[index]        
        inputs, labels = process_dialog(ann, self.tokenizer)
        return {
            "input_ids": inputs,
            "labels": labels
        }


# import json
# from torch.utils.data import Dataset
# import random
# random.seed(2024)

# # 定义模板

# def process_dialog(data, tokenizer):
#     instruction = data['instruction']
#     question = data['input']
#     response = data['output']
#     prompt = f"[INST] {instruction} This is the question: {question} [/INST]"
    
#     inputs = [tokenizer.bos_token_id] + tokenizer.encode(prompt, add_special_tokens=False, truncation=True) + [tokenizer.eos_token_id]
    
#     # 生成标签
#     labels = tokenizer.encode(response, add_special_tokens=False, truncation=True)
#     labels = labels + [tokenizer.eos_token_id]  # 追加结束标记
    
#     # 确保标签的形状与输入相同
#     input_length = len(inputs)
#     labels_length = len(labels)
    
#     if labels_length < input_length:
#         labels = labels + [tokenizer.pad_token_id] * (input_length - labels_length)  # 填充标签
#     elif labels_length > input_length:
#         labels = labels[:input_length]  # 截断标签

#     return inputs, labels

# class CaseDetectDataset(Dataset):
#     def __init__(self, tokenizer, args, train=True):
#         self.ann = []
#         with open(args.train_file if train else args.eval_file, "r") as f:
#             for line in f:
#                 d = json.loads(line)
#                 self.ann.append(d)

#         self.train = train
#         self.tokenizer = tokenizer

#     def __len__(self):
#         return len(self.ann)

#     def __getitem__(self, index):
#         ann = self.ann[index]        
#         inputs, labels = process_dialog(ann, self.tokenizer)
#         return {
#             "input_ids": inputs,
#             "labels": labels
#         }
