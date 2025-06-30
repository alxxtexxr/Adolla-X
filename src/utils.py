import time
import cpuinfo
import torch
import evaluate
from tqdm import tqdm

def get_device_info(device):
    if device.type == 'cuda':
        num_devices = torch.cuda.device_count()
        devices = [torch.cuda.get_device_name(i) for i in range(num_devices)]
        return devices
    else:
        return [cpuinfo.get_cpu_info()['brand_raw']]

def generate_answer_squad(example, model, tokenizer, device, max_new_tokens=50, lang='en'):
    if lang == 'id':
        prompt = f"Konteks: {example['context']}\nPertanyaan: {example['question']}\nJawaban:"
        separator = "Jawaban"
    else:
        prompt = f"Context: {example['context']}\nQuestion: {example['question']}\nAnswer:"
        separator = "Answer"
    
    inputs = tokenizer(prompt, return_tensors='pt')

    # Move inputs to the first device of the model
    first_device = next(model.parameters()).device
    inputs = {k: v.to(first_device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_new_tokens)
        
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True).split(separator)[-1].strip()
    return {'id': example['id'], 'prediction_text': answer}

def evaluate_squad(data, model, tokenizer, device, test_size=100, lang='en'):
    start_time = time.time()
    
    subset = data.select(range(test_size))
    predictions = []
    for example in tqdm(subset, desc="Evaluating", unit="sample"):
        prediction = generate_answer_squad(example, model, tokenizer, device, lang=lang)
        predictions.append(prediction)
    references = [{'id': example['id'], 'answers': example['answers']} for example in subset]

    metric = evaluate.load('squad')
    results = metric.compute(predictions=predictions, references=references)
    results['duration'] = time.time() - start_time
    results['devices'] = get_device_info(device)
    return results