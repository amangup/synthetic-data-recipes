## Training using torchtune

- [Installing torchtune](https://pytorch.org/torchtune/stable/install.html)
- [End-to-end workflow](https://pytorch.org/torchtune/stable/tutorials/e2e_flow.html)
- [Text completion datasets](https://pytorch.org/torchtune/stable/basics/text_completion_datasets.html)
- [Checkpointer](https://pytorch.org/torchtune/stable/deep_dives/checkpointer.html)
- 

### Download the model

Typically, we would already have the model downloaded from 

```
# huggingface-cli scan-cache

meta-llama/Llama-3.1-70B                                  model           141.1G       36 3 days ago    3 days ago    main /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-70B                                    
meta-llama/Llama-3.1-8B                                   model            16.1G       11 2 days ago    2 days ago    main /root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B
```

### Choose recipe

```
# tune ls

full_finetune_distributed                llama2/7B_full                          
                                         llama2/13B_full                         
                                         llama3/8B_full                          
                                         llama3_1/8B_full                        
                                         llama3_2/1B_full                        
                                         llama3_2/3B_full                        
                                         llama3/70B_full                         
                                         llama3_1/70B_full                       
                                         llama3_3/70B_full
                                         gemma2/2B_full                          
                                         gemma2/9B_full                          
                                         gemma2/27B_full                         
            
```

While writing this README, I chose to use the `llama3_1/8B_full` recipe.

### Update the config
We can copy the recipe's config like this:

```
# tune cp llama3_1/8B_full ./llama3_1_8B.yaml
```

Let's make some changes until we are happy with the config and we can launch the full run.

Afte the changes, we can run validation:

```
# tune validate llama3_1_8B.yaml 
Config is well-formed!
```

#### Paths

- `output_dir`: `./ft_models/llama3_1_8B/unconditioned/`
- `tokenizer.path`: `/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/original/tokenizer.model`
- `checkpointer.checkpoint_dir`: `/root/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B/snapshots/d04e592bb4f6aa9cfee91e2e20afa771667e1d4b/`

Note that for tokenizer, we need the `tokenizer.model` file, not the `tokenizer.json`.

#### Dataset

- `dataset._component_`: `torchtune.datasets.text_completion_dataset`
- `dataset.source`: `amang1802/synthetic_data_unconditioned_L3.1_70B_deduped`
- `dataset.split`: `train`
- `dataset.column`: `synthetic_content`
- `dataset.packed`: `True`

#### Training params
- `seed`: `1998`
- `tokenizer.max_seq_len`: `4096` - this is the seq length Llama 3 models were trained with
- `optimizer.lr`: `1e-6` - this is the close to ending value of pretraining of Llama 3 models
- `optimizer_in_bwd`: `True`
- `compile`: `True`
- `enable_activation_checkpointing`: `True`
- `custom_sharded_layers`: commented
- `batch_size`: `32` - max before we get OOM with this config, on 4xH200

Note that it may be possible to optimize the speed further while keeping the effective batch size same: `32 x 4 x 4096`
`
### Run

```
tune run --nproc_per_node 4 full_finetune_distributed --config ./llama3_1_8B.yaml
```

Example log line:

```
Step 5 | loss:0.35119619965553284 lr:1e-06 tokens_per_second_per_gpu:9077.1552734375 peak_memory_active:85.53374147415161 peak_memory_alloc:85.53374147415161 peak_memory_reserved:104.0546875
```