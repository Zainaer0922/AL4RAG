import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

model_name = "your_base_model"
peft_name = 'your_peft_model'
# print(peft_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
model = PeftModel.from_pretrained(model, peft_name)
model.eval() 
tokenizer.pad_token = tokenizer.eos_token

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

def generate_responses(reference, question, task, n_answers=3, max_length=1024, temperature=0.7):

    if task=='QA':
        prompt = TEMPLATES["QA"].format(question=question, reference=reference)
    else:
        prompt = TEMPLATES[task].format(reference=reference)
    
    inputs = tokenizer(prompt, return_tensors='pt', padding=True)

    device = model.device if hasattr(model, 'device') else next(model.parameters()).device
    inputs = {k: v.to(device) for k, v in inputs.items()}

    
    outputs = model.generate(
        input_ids=inputs['input_ids'],
        attention_mask=inputs['attention_mask'],
        max_new_tokens=max_length,
        num_return_sequences=n_answers,
        temperature=temperature,
        top_p=0.9,
        repetition_penalty=1.1,
        do_sample=True
    )

    answers = []
    for i in range(n_answers):
        answer = tokenizer.decode(outputs[i], skip_special_tokens=True)
        answer = answer[len(prompt):].strip()
        answers.append(answer)
    
    return answers


def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def write_jsonl(file_path, item):
    with open(file_path, 'a', encoding='utf-8') as f:
        f.write(json.dumps(item) + '\n')

input_file_path = 'your_test_set'
output_file_path = 'your_output_dir'

data = read_jsonl(input_file_path)

for item in data:
    reference = item['reference']
    question = item['question']
    task = item['task_type']
    for i in range(5):
        generated_answers = generate_responses(reference, question, task, n_answers=1)
        item['generated_answers'] = generated_answers
        output_file = output_file_path.replace('.jsonl', f'_{i+1}.jsonl')
        write_jsonl(output_file, item)

print(f"Generated responses saved to {output_file_path}")
