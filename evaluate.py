import os
import sys
import random
import numpy as np
from models.LMClass import LMClass
import torch
import time
from datautils import get_loaders
from lm_eval import evaluator
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.omniquant import omniquant
from tqdm import tqdm
import utils
from pathlib import Path
from categories import subcategories, categories

from models.int_llama_layer import QuantLlamaDecoderLayer
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.int_linear import QuantLinear

from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM


os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
cache_dir = './cache'
seqlen = 2048
seed = 2
model_path = "/mnt/shared/online/rewrite/moe_human_refine_v1v2v3_0218base_fix_iter109_hf_awq_scale_and_Omni"
model_path = "/mnt/shared/online/rewrite/moe_human_refine_v1v2v3_0218base_fix_iter109_hf_test_quant"
model_family = "Mixtral"


config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
if hasattr(config, 'quantization_config'):
    delattr(config, "quantization_config")
model  = AutoModelForCausalLM.from_pretrained(model_path, config=config, device_map='auto',torch_dtype=config.torch_dtype, trust_remote_code=True)

# for dataset in ["wikitext2", "ptb", "c4","ptb-new",'c4-new']:
for dataset in ["wikitext2"]:
    cache_testloader = f'./cache/testloader_{model_family}_{dataset}_all.cache'
    if os.path.exists(cache_testloader):
        testloader = torch.load(cache_testloader)
        print(f"load calibration from {cache_testloader}")
    else:
        dataloader, testloader = get_loaders(
            dataset,
            seed=seed,
            model=model_path,
            seqlen=seqlen,
        )
        torch.save(testloader, cache_testloader)
    if "c4" in dataset:
        testenc = testloader
    else:
        testenc = testloader.input_ids

    nsamples = testenc.numel() // seqlen
    use_cache = model.config.use_cache
    model.config.use_cache = False
    model.eval()
    nlls = []
    with torch.no_grad():
        for i in tqdm(range(nsamples)):
            batch = testenc[:, (i * seqlen) : ((i + 1) * seqlen)].to(model.device) # (1, 2048)
            outputs = model(batch)
            logits = outputs.logits # (batch, seq_len, vocab_size)
            shift_logits = logits[:, :-1, :] # (batch, seq_len - 1, vocab_size)
            shift_labels = testenc[:, (i * seqlen) : ((i + 1) * seqlen)][:, 1:].cuda()
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)), #(seq_len - 1, vocab_size)
                shift_labels.view(-1), #(seq_len - 1)
            )
            neg_log_likelihood = loss.float() * seqlen
            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * seqlen))
        print(f'{dataset} : {ppl.item()}')
               