import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
import time

custom_cache_dir = './cache' # cloud
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")
batch_size = 12 # 48 documents as a batch
max_length = 1024 # max length to generate (should be <= model max length, 2048 for Pythia)
num_docs = 4800 # total number of documents to generate 400!
#steps = [150, 5000, 11000, 20000, 30000, 45000, 140000, 105000, 165000, 255000, 395000, 615000, 928646] # /2
revisions = [
    "stage1-step1100-tokens10B",
    "stage1-step5000-tokens42B",
    "stage1-step7000-tokens59B",
    "stage1-step12000-tokens101B",
    "stage1-step19850-tokens167B",
    "stage1-step31000-tokens261B",
    "stage1-step48000-tokens403B",
    "stage1-step72000-tokens604B",
    "stage1-step107500-tokens902B",
    "stage1-step167000-tokens1401B",
    "stage1-step239000-tokens2005B",
    "stage1-step358000-tokens3004B",
    "stage1-step596057-tokens5001B",
]

# chunk_size_time = None
accumulated_loss = torch.zeros(len(revisions))
total_tokens = torch.zeros(len(revisions))
accumulated_logit_std = torch.zeros(len(revisions))
accumulated_logit_mean = torch.zeros(len(revisions))
skewness = torch.zeros(len(revisions))
skewness_extreme = torch.zeros(len(revisions))
logit_range = torch.zeros(len(revisions))

model_name = "allenai/OLMo-2-1124-13B"
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=custom_cache_dir, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

dataset = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)

for rev_idx, rev in enumerate(revisions):
    data_iter = iter(dataset)

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        revision=rev,
        cache_dir=custom_cache_dir,
        torch_dtype = torch.bfloat16,
        device_map="auto",
        offload_buffers=True
    ).eval()

    start_time = time.time()

    for step in range(num_docs // batch_size):
        batch = [next(data_iter)['text'] for _ in range(batch_size)]
        inputs = tokenizer(batch, return_tensors="pt", padding="max_length",
            truncation=True,
            max_length=max_length) # ids and attention masks (batch_size, seq_len)
        labels = inputs.input_ids.clone()
        labels[labels == tokenizer.pad_token_id] = -100 # (batch_size, seq_len)
        
        valid_mask = (inputs['attention_mask'][:, 1:] > 0) & (inputs['attention_mask'][:, :-1] > 0) # (batch_size, seq_len-1)
        
        with torch.no_grad():
            out = model(
                **inputs.to(model.device),
                labels=labels.to(model.device)
            )

        valid_mask = valid_mask.to(out.logits.device)
        logits = out.logits[:,:-1][valid_mask].to(torch.float32) # (num_valid_tokens, vocab_size)
        valid_labels = labels[:,1:].to(out.logits.device)[valid_mask] # (num_valid_tokens,)
        # gather the logits corresponding to the valid labels
        correct_logits = logits.gather(1, valid_labels.unsqueeze(-1)).squeeze(-1) # (num_valid_tokens,)
        total_tokens[rev_idx] += logits.shape[0]
        accumulated_logit_std[rev_idx] += logits.std(dim = -1).sum().item() / max_length
        accumulated_logit_mean[rev_idx] += logits.mean(dim = -1).sum().item() / max_length
        accumulated_loss[rev_idx] += out.loss.to(torch.float32).item() * logits.shape[0]
        skewness[rev_idx] += (correct_logits - logits.mean(dim = -1)).sum().item() / max_length
        # skeewness_extreme: find the maximum logits - mean logits
        skewness_extreme[rev_idx] += (logits.max(dim = -1).values - logits.mean(dim = -1)).sum().item() / max_length
        logit_range[rev_idx] += (logits.max(dim = -1).values - logits.min(dim = -1).values).sum().item() / max_length

    del model
    torch.cuda.empty_cache()
    print(f"==> Revision {rev}, Time for {num_docs} docs: {time.time() - start_time:.2f} seconds")

to_save = {'loss': accumulated_loss / total_tokens, 'logit_std': accumulated_logit_std / total_tokens * max_length,
            'logit_mean': accumulated_logit_mean / total_tokens * max_length, 'total_tokens': total_tokens,
            'skewness': skewness / total_tokens * max_length, 'skewness_extreme': skewness_extreme / total_tokens * max_length,
            'logit_range': logit_range / total_tokens * max_length}

print(to_save)

#import os
#os.makedirs('/n/netscratch/glassman_lab/Lab/zhentingqi/my_projects/olmo-add/saves', exist_ok=True)
torch.save(to_save,
           f'olmo2-13b-logit-2.pt') # cloud