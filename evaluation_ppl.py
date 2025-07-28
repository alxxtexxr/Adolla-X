import fire
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, AutoPeftModelForCausalLM
from tqdm import tqdm
from pprint import pprint

def compute_ppl(model, encodings, max_seq_length, stride=512, return_avg_nll=False):
    seq_length = encodings.input_ids.size(1)

    nll_sum = 0.0
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_length, stride)):
        end_loc = min(begin_loc + max_seq_length, seq_length)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc] #.to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(input_ids, labels=target_ids)

            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss

        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (target_ids != -100).sum().item()  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = num_valid_tokens - batch_size  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens

        prev_end_loc = end_loc
        if end_loc == seq_length:
            break

    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    if return_avg_nll:
        return ppl, avg_nll
    return ppl

def format_gsm8k_prompts(examples, eos_token):
    gsm8k_prompt = """### Instruction:
Solve the following math problem step by step.

### Question: 
{question}

### Answer: 
{answer}""" + eos_token
    
    return {'text': [gsm8k_prompt.format(question=question, answer=answer) for question, answer in zip(examples['question'], examples['answer'])]}

def format_prompts(examples, eos_token):
    return {'text': [example + eos_token for example in examples['text']]}

def count_total_tokens(dataset, tokenizer):
    total_tokens = 0
    for text in dataset['text']:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        total_tokens += len(tokens)
    return total_tokens

def main(
    # Project configuration
    hf_lora_id,
    model_type, # 'base' | 'lora'
    seed = 69,
    
    # Data configuration
    test_size = 1000,
    max_seq_length = 1024,
    
    # Model configuration
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.
):
    # Get task and language
    task, lang, _ = hf_lora_id.split('B-')[-1].split('K-')[0].split('-')
    print("Task:", task)
    print("Language:", lang)
    
    # Set up Hugging Face configuration
    hf_data_id_map = {
        'wikipedia': 'wikimedia/wikipedia',
        'gsm8k': 'openai/gsm8k',
    }
    hf_data_id = hf_data_id_map[task]
    hf_data_dir = f'20231101.{lang}' if task == 'wikipedia' else 'main'
    hf_data_split = f'train[-{test_size}:]'

    # Download the trained LoRA adapter to the local directory
    from huggingface_hub import snapshot_download
    lora_dir = hf_lora_id.split('/')[-1]
    snapshot_download(
        repo_id=hf_lora_id, 
        local_dir=lora_dir, 
        ignore_patterns=[f'checkpoint-{i}/*' for i in range(0, 1870, 50) if i not in [650, 1250, 1875]],
    )
    print("Hugging Face LoRA ID:", hf_lora_id)

    # Load LoRA configuration
    lora_config = LoraConfig.from_pretrained(lora_dir)

    # ==== DATA ====
    tokenizer = AutoTokenizer.from_pretrained(lora_dir)
    dataset = load_dataset(hf_data_id, data_dir=hf_data_dir, split=hf_data_split)
    eos_token = tokenizer.eos_token

    if task == 'gsm8k':
        dataset = dataset.map(format_gsm8k_prompts, batched=True)
    else:
        dataset = dataset.map(format_prompts, batched=True)
    
    encodings = tokenizer("\n\n".join(dataset['text']), return_tensors='pt')
    print(dataset)

    # Count total tokens
    total_token_count = count_total_tokens(dataset, tokenizer)
    print(f"Total tokens in dataset: {total_token_count}")

    # ==== MODEL ====
    if model_type == 'lora':
        # Load the LoRA-adapted model
        model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(lora_dir, 'checkpoint-650'), device_map='auto')
        save_dir = os.path.join('evaluations', hf_lora_id)
    else:
        model = AutoModelForCausalLM.from_pretrained(lora_config.base_model_name_or_path, device_map='auto')
        save_dir = os.path.join('evaluations', lora_config.base_model_name_or_path)
    model.eval()

    ppl, avg_nll = compute_ppl(base_model, encodings, max_seq_length, return_avg_nll=True)
    print("Avgerage Negative Log-Likelihood (NLL):", avg_nll.item())
    print("Perplexity (PPL):", ppl.item())

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{task}_{lang}_{test_size}.json')
    with open(save_path, 'w') as f:
        json.dump({ 'ppl': ppl, 'avg_nll': avg_nll }, f)
    print("Evaluation results are saved to:", save_path)

if __name__ == '__main__':
    fire.Fire(main)