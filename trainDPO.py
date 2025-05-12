import json
from trl.commands.cli_utils import DPOScriptArguments, TrlParser
from trl.trainer.utils import SIMPLE_CHAT_TEMPLATE
import torch
import random
from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from accelerate import PartialState
from sklearn.model_selection import train_test_split
from peft import LoraConfig, PeftModel
import pandas as pd
from trl import (
    DPOConfig,
    DPOTrainer,
    ModelConfig,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
    maybe_extract_prompt,
    maybe_apply_chat_template,
)
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

def preprocess_data(example):
    task = example['task_type']
    chosen_text = random.choice(example['chosen']) if isinstance(example['chosen'], list) else example['chosen']
    rejected_text = random.choice(example['rejected']) if isinstance(example['rejected'], list) else example['rejected']
    
    if task=='QA':
        prompt = TEMPLATES["QA"].format(question=example['question'], reference=example['reference'])
    else:
        prompt = TEMPLATES[task].format(reference=example['reference'])
    
    return {
        "prompt": prompt,
        "chosen": chosen_text,
        "rejected": rejected_text,
    }

if __name__ == "__main__":
    parser = TrlParser((DPOScriptArguments, DPOConfig, ModelConfig))
    args, training_args, model_config = parser.parse_args_and_config()

    ################
    # Model & Tokenizer
    ###################
    torch_dtype = torch.bfloat16
    # (
    #     model_config.torch_dtype
    #     if model_config.torch_dtype in ["auto", None]
    #     else getattr(torch, model_config.torch_dtype)
    # )
    quantization_config = get_quantization_config(model_config)
    model_kwargs = dict(
        revision=model_config.model_revision,
        attn_implementation=model_config.attn_implementation,
        torch_dtype=torch_dtype,
        use_cache=False if training_args.gradient_checkpointing else True,
        device_map='auto',
        quantization_config=quantization_config,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
    )
    model = PeftModel.from_pretrained(model, 'your_sft_model')
    peft_config = LoraConfig(
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            bias="none",
            task_type="CAUSAL_LM"
        )
    if peft_config is None:
        ref_model = AutoModelForCausalLM.from_pretrained(
            model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code, **model_kwargs
        )
    else:
        ref_model = None
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path, trust_remote_code=model_config.trust_remote_code
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_CHAT_TEMPLATE
    if args.ignore_bias_buffers:
        # torch distributed hack
        model._ddp_params_and_buffers_to_ignore = [
            name for name, buffer in model.named_buffers() if buffer.dtype == torch.bool
        ]

    ################
    # Dataset
    ################
    file_path1 = args.dataset_name
    file_path2 = "path_to_your_eval_set"
    data1 = []
    data2 = []

    try:
        with open(file_path1, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    if 'reference' in example and isinstance(example['reference'], (dict, list)):
                        example['reference'] = json.dumps(example['reference'])
                    data1.append(example)
                except json.JSONDecodeError as e:
                    print(f"JSON Decoding Error: {e}")
    except FileNotFoundError:
        print(f"File {file_path1} not found.")

    try:
        with open(file_path2, 'r') as f:
            for line in f:
                try:
                    example = json.loads(line)
                    if 'reference' in example and isinstance(example['reference'], (dict, list)):
                        example['reference'] = json.dumps(example['reference'])
                    data2.append(example)
                except json.JSONDecodeError as e:
                    print(f"JSON Decoding Error: {e}")
    except FileNotFoundError:
        print(f"File {file_path2} not found.")

    dataset1 = Dataset.from_list(data1)
    dataset2 = Dataset.from_list(data2)
    train_df = pd.DataFrame(dataset1)
    test_df = pd.DataFrame(dataset2)
    train_data = Dataset.from_pandas(train_df)
    test_data = Dataset.from_pandas(test_df)
    train_data = train_data.map(preprocess_data)
    test_data = test_data.map(preprocess_data)

    ##########
    # Training
    ################
    trainer = DPOTrainer(
        model,
        ref_model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        tokenizer=tokenizer,
        peft_config=peft_config,
        max_prompt_length=1024
    )

    trainer.train()
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model(training_args.output_dir)
