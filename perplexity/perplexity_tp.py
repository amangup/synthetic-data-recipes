import os
import torch

from datasets import load_dataset
from torch.distributed import destroy_process_group
from torch.distributed._tensor.device_mesh import init_device_mesh
from transformers import LlamaForCausalLM, AutoTokenizer


torch.set_float32_matmul_precision('high')

rank = int(os.environ["RANK"])
world_size = int(os.environ["WORLD_SIZE"])
device = torch.device(f"cuda:{rank}")


def load_model(model_path):
    model = LlamaForCausalLM.from_pretrained(
        model_path,
        low_cpu_mem_usage=True,
    )

    tp_mesh = init_device_mesh("cuda", (world_size,))
    model.tensor_parallel(tp_mesh)
    model.model.embed_tokens.to(device)
    model.model.rotary_emb.to(device)
    model.model.norm.to(device)
    
    for layer in model.model.layers:
        layer.input_layernorm.to(device)
        layer.post_attention_layernorm.to(device)
    
    model.eval()

    return model


def get_tokens_batch(batch, tokenizer, max_seq_len):
    encodings = tokenizer(
        batch,
        padding='longest',
        return_tensors="pt",
        max_length=max_seq_len,
        truncation=True,
    ).to(device)
    
    return encodings


def get_shift_labels(encodings, context_length, tokenizer):
    labels = encodings.input_ids.clone()
    labels[labels == tokenizer.pad_token_id] = -100
    labels[:, :context_length] = -100
    shift_labels = labels[:, 1:]

    return shift_labels


def batch_compute_perplexity(model, tokenizer, texts, context_length, batch_size, max_seq_len):
    torch.manual_seed(1998)

    nll_sum = torch.tensor(0, dtype=torch.float64, requires_grad=False).to(device)
    n_tokens = torch.tensor(0, dtype=torch.int64, requires_grad=False).to(device)

    for i in range(0, len(texts), batch_size):
        if rank == 0:
            print(f"processing batch: {i//batch_size} out of {len(texts)//batch_size}")

        batch = texts[i:i + batch_size]
        encodings = get_tokens_batch(batch, tokenizer, max_seq_len)
        # if rank == 0:
        #     print(encodings.input_ids.shape)
        
        with torch.no_grad():
            shift_logits = model(
                input_ids=encodings.input_ids,
                attention_mask=encodings.attention_mask
            ).logits[:, :-1]
        
        shift_labels = get_shift_labels(encodings, context_length, tokenizer)
        
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100
        ).type(torch.float64)
        
        num_valid_tokens = (shift_labels != -100).sum()
        
        nll_sum += loss * num_valid_tokens
        n_tokens += num_valid_tokens
            
    avg_nll = nll_sum / n_tokens
    ppl = torch.exp(avg_nll)
        
    return nll_sum, n_tokens, avg_nll, ppl


def main():
    #model_path = "/root/code/synthetic-data-recipes/diversity/ft_models/llama3_1_70B/fulltext_conditioned_20epochs_lrcosine/epoch_16"
    model_path = "meta-llama/Llama-3.1-70B"
    context_length = 16
    batch_size = 1
    num_samples = 512
    max_seq_len = 2048
    
    ds_list = [
        'amang1802/synthetic_data_fulltext_conditioned_L3.3_70B_deduped'
    ]
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.pad_token = tokenizer.eos_token

    model = load_model(model_path)

    scores = {}
    for ds_id in ds_list:
        ds = load_dataset(ds_id)
        for split in ['train', 'test']:
            if rank==0:
                print(f"Running for {ds_id}:{split}")
            
            texts = ds[split].shuffle(seed=1998).select(range(num_samples))['synthetic_content']
            metrics = batch_compute_perplexity(model, tokenizer, texts, context_length, batch_size, max_seq_len)
            scores[(ds_id, split)] = metrics

    if rank==0:
        print(scores)
    
    destroy_process_group()


# Run as: torchrun --nproc-per-node 4 perplexity_tp.py
if __name__ == '__main__':
    main()