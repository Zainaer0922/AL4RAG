import json
import random
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
random.seed(2024)

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

def load_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

def append_jsonl(data, file_path):
    with open(file_path, 'a', encoding='utf-8') as f:
        for entry in data:
            f.write(json.dumps(entry) + "\n")

def group_by_task_type(data):
    grouped_data = {}
    for entry in data:
        task_type = entry.get('task_type')
        if task_type not in grouped_data:
            grouped_data[task_type] = []
        grouped_data[task_type].append(entry)
    return grouped_data

def select_initial_samples(data, percentage=0.05):
    sample_size = int(len(data) * percentage)
    initial_samples = random.sample(data, sample_size)
    remaining_samples = [d for d in data if d not in initial_samples]
    return initial_samples, remaining_samples

def fill_template(dataset):
    filled_texts = []
    for item in dataset:
        task_type = item['task_type']
        if item['chosen']=="Sorry, this question is beyond my ability.":
            answer = item['rejected']
        else:
            answer = item['chosen']
        template = TEMPLATES[task_type]
        if task_type=="QA":
            filled_text = template.format(question=item["question"], reference=item["reference"])
        else:
            filled_text = template.format(reference=item["reference"])
        filled_texts.append(filled_text)
    return filled_texts

def get_prompt_embeddings(data, vectorizer):
    texts = fill_template(data)
    tfidf_matrix = vectorizer.transform(texts)
    return tfidf_matrix

def get_answer_embeddings(data, vectorizer):
    texts = []
    for item in data:
        if item['chosen']=="Sorry, this question is beyond my ability.":
            texts.append(item['rejected'])
        else:
            texts.append(item['chosen'])
    tfidf_matrix = vectorizer.transform(texts)
    return tfidf_matrix

def get_question_embeddings(data, vectorizer):
    texts = [d['question'] for d in data]
    tfidf_matrix = vectorizer.transform(texts)
    return tfidf_matrix

def get_reference_embeddings(data, vectorizer):
    texts = [str(d['reference']) for d in data]
    tfidf_matrix = vectorizer.transform(texts)
    return tfidf_matrix

def select_min_sim(prompt_sim, answer_sim):
    final_similarities = []
    for i in range(len(prompt_sim)):
        row_similarities = []
        for j in range(len(answer_sim[0])):
            min_similarity = min(prompt_sim[i][j], answer_sim[i][j])
            row_similarities.append(min_similarity)
        final_similarities.append(row_similarities)
    return final_similarities

def select_max_sim(prompt_sim, answer_sim):
    final_similarities = []
    for i in range(len(prompt_sim)):
        row_similarities = []
        for j in range(len(answer_sim[0])):
            max_similarity = max(prompt_sim[i][j], answer_sim[i][j])
            row_similarities.append(max_similarity)
        final_similarities.append(row_similarities)
    return final_similarities

def weighted_sim(question_sim, reference_sim, answer_sim):
    final_similarities = []
    for i in range(len(question_sim)):
        row_similarities = []
        for j in range(len(question_sim[0])):
            weighted_similarity = (question_sim[i][j] + reference_sim[i][j]) / 2
            row_similarities.append(weighted_similarity)
        final_similarities.append(row_similarities)
    return final_similarities

def idds_score(unlabeled_prompt, unlabeled_question, unlabeled_reference, unlabeled_answer, labeled_prompt, labeled_question, labeled_reference, labeled_answer, lambda_param=0.67):
    sim_to_labeled_prompt = cosine_similarity(unlabeled_prompt, labeled_prompt)
    sim_to_unlabeled_prompt = cosine_similarity(unlabeled_prompt, unlabeled_prompt)
    sim_to_labeled_question = cosine_similarity(unlabeled_question, labeled_question)
    sim_to_unlabeled_question = cosine_similarity(unlabeled_question, unlabeled_question)
    sim_to_labeled_reference = cosine_similarity(unlabeled_reference, labeled_reference)
    sim_to_unlabeled_reference = cosine_similarity(unlabeled_reference, unlabeled_reference)
    sim_to_labeled_answer = cosine_similarity(unlabeled_answer, labeled_answer)
    sim_to_unlabeled_answer = cosine_similarity(unlabeled_answer, unlabeled_answer)

    # labeled_tfidf = select_min_sim(sim_to_labeled_prompt,sim_to_labeled_question)
    # unlabeled_tfidf = select_min_sim(sim_to_unlabeled_prompt,sim_to_unlabeled_question)
    labeled_tfidf = select_min_sim(weighted_sim(sim_to_labeled_question,sim_to_labeled_reference,sim_to_labeled_answer), sim_to_labeled_prompt)
    unlabeled_tfidf = select_min_sim(weighted_sim(sim_to_unlabeled_question,sim_to_unlabeled_reference,sim_to_unlabeled_answer), sim_to_unlabeled_prompt)
    # labeled_tfidf = sim_to_labeled_prompt
    # unlabeled_tfidf = sim_to_unlabeled_prompt
    sim_to_labeled = np.mean(labeled_tfidf, axis=1)
    sim_to_unlabeled = np.mean(unlabeled_tfidf, axis=1)
    
    scores = lambda_param * sim_to_unlabeled - (1 - lambda_param) * sim_to_labeled
    return scores

def select_samples_by_idds(labeled_data, grouped_unlabeled_data, vectorizer_q, vectorizer_r, vectorizer_p, vectorizer_a, total_unlabeled_data_size, selection_percentage=0.05, lambda_param=0.67):
    selected_samples = []
    remaining_samples = []
    
    total_sample_size = int(total_unlabeled_data_size * selection_percentage / len(grouped_unlabeled_data))
    
    for task_type, unlabeled_data in grouped_unlabeled_data.items():
        if len(unlabeled_data) == 0:
            continue

        labeled_question = get_question_embeddings(labeled_data, vectorizer_q)
        labeled_reference = get_reference_embeddings(labeled_data, vectorizer_r)
        labeled_prompt = get_prompt_embeddings(labeled_data, vectorizer_p)
        labeled_answer = get_answer_embeddings(labeled_data, vectorizer_a)
        
        unlabeled_question = get_question_embeddings(unlabeled_data, vectorizer_q)
        unlabeled_reference = get_reference_embeddings(unlabeled_data, vectorizer_r)
        unlabeled_prompt = get_prompt_embeddings(unlabeled_data, vectorizer_p)
        unlabeled_answer = get_answer_embeddings(unlabeled_data, vectorizer_a)
        
        scores = idds_score(unlabeled_prompt, unlabeled_question, unlabeled_reference, unlabeled_answer, labeled_prompt, labeled_question, labeled_reference, labeled_answer, lambda_param=lambda_param)
        
        sample_size = min(total_sample_size, len(unlabeled_data))
        top_k_indices = np.argsort(scores)[-sample_size:]
        
        selected_samples.extend([unlabeled_data[i] for i in top_k_indices])
        remaining_samples.extend([unlabeled_data[i] for i in range(len(unlabeled_data)) if i not in top_k_indices])
    
    return selected_samples, remaining_samples

def iterative_idds_sampling(file_path, output_file, iterations=4, initial_percentage=0.05, selection_percentage=0.05, lambda_param=0.67):
    # You can change the selected data proportion and the number of iterations here.
    data = load_jsonl(file_path)
    
    labeled_data, unlabeled_data = select_initial_samples(data, percentage=initial_percentage)
    
    vectorizer_q = TfidfVectorizer()
    vectorizer_r = TfidfVectorizer()
    vectorizer_p = TfidfVectorizer()
    vectorizer_a = TfidfVectorizer()
    vectorizer_q.fit([d['question'] for d in data])
    vectorizer_r.fit([str(d['reference']) for d in data])
    vectorizer_p.fit(fill_template(data))
    vectorizer_a.fit([d['chosen'] if d['chosen']!="Sorry, this question is beyond my ability." else d['rejected'] for d in data])
    
    grouped_unlabeled_data = group_by_task_type(unlabeled_data)
    total_unlabeled_data_size = len(data)
    
    append_jsonl(labeled_data, output_file)
    
    for iteration in range(iterations):
        print(f"Iteration {iteration + 1}")
        selected_samples, remaining_unlabeled_data = select_samples_by_idds(labeled_data, grouped_unlabeled_data, vectorizer_q, vectorizer_r, vectorizer_p, vectorizer_a, total_unlabeled_data_size, selection_percentage=selection_percentage, lambda_param=lambda_param)
        
        labeled_data.extend(selected_samples)
        
        grouped_unlabeled_data = group_by_task_type(remaining_unlabeled_data)
        
        append_jsonl(selected_samples, output_file)
        
        print(f"Selected {len(selected_samples)} samples in iteration {iteration + 1}")
        print(f"Remaining unlabeled samples: {len(remaining_unlabeled_data)}")
    
    return labeled_data, remaining_unlabeled_data

file_path = 'path_to_your_training_set'
output_file = 'your_output_path'
labeled_data, remaining_unlabeled_data = iterative_idds_sampling(file_path, output_file)
