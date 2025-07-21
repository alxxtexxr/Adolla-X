from unsloth import FastLanguageModel, UnslothTrainer, UnslothTrainingArguments, is_bf16_supported
import fire
import torch
from itertools import islice
from datetime import datetime
from datasets import load_dataset, Dataset
from trl import SFTTrainer
from transformers import TrainingArguments

def main(
    lang, # 'en' | 'id' | 'es'
    task, # 'wikipedia' | 'gsm8k'
    seed = 69,

    # Data configuration
    train_size = 5000,
    test_size = 1000,
    max_seq_length = 1024,

    # Model configuration
    dtype = None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = True, # Use 4bit quantization to reduce memory usage. Can be False.

    # LoRA configuration
    lora_target_modules = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj'],
    lora_r = 8,
    lora_alpha = 16,
    
    # Resume training configuration
    resume_model_id = None,
):
    hf_data_id_map = {
        'wikipedia': 'wikimedia/wikipedia',
        'gsm8k': 'openai/gsm8k',
    }
    hf_data_id = hf_data_id_map[task]
    
    hf_data_dir_map = {
        'wikipedia_en': '20231101.en',
        'wikipedia_id': '20231101.id',
        'wikipedia_es': '20231101.es',
        'gsm8k_en': 'main',
    }
    hf_data_dir = hf_data_dir_map[task + '_' + lang]

    resume_from_checkpoint = bool(resume_model_id)
    if resume_from_checkpoint:
        hub_model_id = resume_model_id
        project_name = hub_model_id.split('/')[-1]
        model_name = project_name
    
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=hub_model_id, local_dir=model_name)
    else:
        model_name = 'unsloth/Meta-Llama-3.1-8B'
        project_name = f'L3.1-8B-{task}-{lang}-{train_size//1000}K-LoRA-v{datetime.now().strftime("%Y%m%d%H%M%S")}'
        hub_model_id = f'alxxtexxr/{project_name}'
    print("Resume from checkpoint:", resume_from_checkpoint)
    print("Project name:", project_name)
    print("Hugging Face model ID:", hub_model_id)

    # Load the model and the tokenizer
    model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    # Set up the PEFT model
    model = FastLanguageModel.get_peft_model(
        model,
        random_state=seed,
        target_modules=lora_target_modules,
        r=lora_r,
        lora_alpha=lora_alpha,   
        lora_dropout=0, # Supports any, but = 0 is optimized
        bias='none',    # Supports any, but = 'none' is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing=False, # True or 'unsloth' for very long context
        use_rslora=False,
        loftq_config=None,
    )

    streamed = load_dataset(hf_data_id, data_dir=hf_data_dir, split='train', streaming=True)
    subset = list(islice(streamed, (train_size + test_size)))
    dataset = Dataset.from_list(subset)
    eos_token = tokenizer.eos_token

    def format_gsm8k_prompts(examples):
        gsm8k_prompt = """### Instruction:
Solve the following math problem step by step.
    
### Question: 
{question}
    
### Answer: 
{answer}""" + eos_token
            
        return {'text': [gsm8k_prompt.format(question=question, answer=answer) 
                         for question, answer in zip(examples['question'], examples['answer'])]}
        
    def format_prompts(examples):
        return {'text': [example + eos_token for example in examples['text']]}
    
    if task == 'gsm8k':
        dataset = dataset.map(format_gsm8k_prompts, batched=True)
    else:
        dataset = dataset.map(format_prompts, batched=True)

    dataset_split = dataset.train_test_split(train_size=train_size, test_size=test_size, seed=seed)

    # Train the model
    trainer = UnslothTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset_split['train'],
        # eval_dataset=dataset_split['test'],
        dataset_text_field='text',
        max_seq_length=max_seq_length,
        dataset_num_proc=8,
    
        args=TrainingArguments(
            seed=seed,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            num_train_epochs=3,
            # max_steps=3, # For debugging
            warmup_ratio=0.05,
            learning_rate=2e-4,
            lr_scheduler_type='cosine',
            optim='paged_adamw_8bit', # 'paged_adamw_8bit' | 'adamw_8bit'
            weight_decay=0.00,
            max_grad_norm=0.3,
            fp16=(not is_bf16_supported()),
            bf16=is_bf16_supported(),
    
            # Eval arguments
            # eval_strategy='steps',
            # eval_steps=10,
            
            # Logging arguments
            logging_strategy='steps',
            logging_steps=1,
            # logging_first_step=True,
            report_to=['tensorboard', 'wandb'],
    
            # Saving arguments
            save_strategy='steps',
            save_steps=50,
            # save_steps=1, # For debugging
            save_total_limit=5, # 1 best + 4 recent checkpoints. Warning: It doesn't work
            
            # With load_best_model_at_end=True, your save_strategy will be ignored and default to eval_strategy.
            # So you will find one checkpoint at the end of each epoch.
            # https://discuss.huggingface.co/t/trainer-not-saving-after-save-steps/5464
            # load_best_model_at_end=True, 
    
            output_dir=project_name,
            hub_model_id=hub_model_id,
            push_to_hub=True,
    
            hub_strategy='all_checkpoints',
            hub_always_push=True,
        ),
    )
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

if __name__ == '__main__':
    fire.Fire(main)