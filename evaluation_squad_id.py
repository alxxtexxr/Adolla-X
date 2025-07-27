import os
import requests
import json
import fire
import torch
from huggingface_hub import snapshot_download
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import LoraConfig, AutoPeftModelForCausalLM
from pprint import pprint
from vastai_sdk import VastAI

from src.utils import evaluate_squad

def main(
    hf_lora_id,
    model_type, # 'base' | 'lora' | 'nero'
    test_size,
    data_path = 'data/train-SQuAD-id.json',
    vastai_api_key = None,
    vastai_instance_id = None,
):
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Download the trained LoRA adapter to the local directory
    lora_dir = hf_lora_id.split('/')[-1]
    snapshot_download(
        repo_id=hf_lora_id, 
        local_dir=lora_dir, 
        # ignore_patterns='checkpoint-*/*',
    )
    print("Hugging Face LoRA ID:", hf_lora_id)

    # Load LoRA configuration
    lora_config = LoraConfig.from_pretrained(lora_dir)

    # Download data
    os.makedirs('data', exist_ok=True)
    if not os.path.exists(data_path):
        print("Data not found. Downloading...")
        response = requests.get('https://raw.githubusercontent.com/Wikidepia/SQuAD-id/refs/heads/master/data/train-SQuAD-id.json')
        response.raise_for_status()  # Ensure download was successful
        with open(data_path, 'wb') as f:
            f.write(response.content)
        print("Download complete.")
    else:
        print("Data already exists. Skipping download.")

    # Load and convert the data to Huggingface format
    # Source: https://github.com/Wikidepia/indonesian_datasets/blob/master/question-answering/squad/convert_huggingface.py
    with open(data_path, 'r') as f:
        content = json.load(f)
    
    hf_data = []
    for data in content["data"]:
        title = data["title"]
        for paragraph in data["paragraphs"]:
            context = paragraph["context"]
            for qa in paragraph["qas"]:
                fill = {
                    "id":  qa["id"],
                    "title": title,
                    "context": context,
                    "question": qa["question"],
                    "answers": {"answer_start": [], "text": []}
                }
                if qa["is_impossible"]:
                    answers = qa["plausible_answers"]
                else:
                    answers = qa["answers"]
                for answer in answers:
                    fill["answers"]["answer_start"].append(answer["answer_start"])
                    fill["answers"]["text"].append(answer["text"])
                hf_data.append(fill)

    # Create Huggingface dataset
    data = Dataset.from_list(hf_data)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)

    # Load model
    if model_type == 'lora':
        model = AutoPeftModelForCausalLM.from_pretrained(lora_dir, device_map='auto')
    else:
        model = AutoModelForCausalLM.from_pretrained(lora_config.base_model_name_or_path, device_map='auto')
    model.eval()

    # Evaluate the model
    results = evaluate_squad(data, model, tokenizer, device, test_size=test_size, lang='id')
    pprint(results)

    # Save the evaluation results
    if model_type == 'lora':
        save_dir = os.path.join('evaluations/squad_id', hf_lora_id)
    else:
        save_dir = os.path.join('evaluations/squad_id', lora_config.base_model_name_or_path)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f'{test_size}.json')
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)  # indent=4 makes it pretty-printed
    print("Evaluation results is saved to:", save_path)

    # Stop vast.ai instance
    if vastai_api_key and vastai_instance_id:
        vast_sdk = VastAI(api_key=vastai_api_key)
        vast_sdk.stop_instance(id=vastai_instance_id)

if __name__ == '__main__':
    fire.Fire(main)