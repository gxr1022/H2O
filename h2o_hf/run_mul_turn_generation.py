#!/usr/bin/env python
# coding=utf-8
# Copyright 2018 Google AI, Google Brain and Carnegie Mellon University Authors and the HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Conditional text generation with the auto-regressive models
"""


import argparse
import logging

import numpy as np
import torch
import json
import tqdm 
import copy 

from transformers import (
    CTRLLMHeadModel,
    CTRLTokenizer,
    GPT2LMHeadModel,
    GPT2Tokenizer,
    OpenAIGPTLMHeadModel,
    OpenAIGPTTokenizer,
    TransfoXLLMHeadModel,
    TransfoXLTokenizer,
    XLMTokenizer,
    XLMWithLMHeadModel,
    XLNetLMHeadModel,
    XLNetTokenizer,
)

from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig

from utils_hh.modify_llama import convert_kvcache_llama_heavy_recent, LlamaAttention_heavy_hitter
# from utils_hh.modify_gptneox import convert_kvcache_gpt_neox_heavy_recent, GPTNeoXAttention_Mask
from utils_hh.modify_opt import convert_kvcache_opt_heavy_recent, OPTAttention_Mask


logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

MAX_LENGTH = int(10000)  # Hardcoded max length to avoid infinite loop

MODEL_CLASSES = {
    "gpt2": (GPT2LMHeadModel, GPT2Tokenizer),
    "ctrl": (CTRLLMHeadModel, CTRLTokenizer),
    "openai-gpt": (OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    "xlnet": (XLNetLMHeadModel, XLNetTokenizer),
    "transfo-xl": (TransfoXLLMHeadModel, TransfoXLTokenizer),
    "xlm": (XLMWithLMHeadModel, XLMTokenizer),
}

# Padding text to help Transformer-XL and XLNet with short prompts as proposed by Aman Rusia
# in https://github.com/rusiaaman/XLNet-gen#methodology
# and https://medium.com/@amanrusia/xlnet-speaks-comparison-to-gpt-2-ea1a4e9ba39e
PREFIX = """In 1991, the remains of Russian Tsar Nicholas II and his family
(except for Alexei and Maria) are discovered.
The voice of Nicholas's young son, Tsarevich Alexei Nikolaevich, narrates the
remainder of the story. 1883 Western Siberia,
a young Grigori Rasputin is asked by his father and a group of men to perform magic.
Rasputin has a vision and denounces one of the men as a horse thief. Although his
father initially slaps him for making such an accusation, Rasputin watches as the
man is chased outside and beaten. Twenty years later, Rasputin sees a vision of
the Virgin Mary, prompting him to become a priest. Rasputin quickly becomes famous,
with people, even a bishop, begging for his blessing. <eod> </s> <eos>"""


def set_seed(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)


ENABLE_Heavy_Hitter_FUNCTIONS = {
    "llama": convert_kvcache_llama_heavy_recent,
    "opt": convert_kvcache_opt_heavy_recent,
    # "gpt_neox": convert_kvcache_gpt_neox_heavy_recent,
}

def load_conversation_from_sharedgpt(file_path, conversation_id):
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Find the conversation with the specified ID
    for item in data:
        if item.get('id') == conversation_id:
            # Extract the conversations list
            conversations = item.get('items', [])
            num_turns = len(conversations)

             # Create new filename based on original path
            file_name = file_path.rsplit('.', 1)[0]  # Remove extension
            new_file_path = f"{file_name}_id{conversation_id}.json"
            
            # Save single conversation to new file
            with open(new_file_path, 'w', encoding='utf-8') as f:
                json.dump(conversations, f, ensure_ascii=False, indent=2)
            
            print(f"Saved conversation {conversation_id} to {new_file_path}")

            return conversations, num_turns
    raise ValueError(f"Conversation with ID {conversation_id} not found.")

def format_conversation(conversations, current_turn):
    conv = conversations[current_turn]
    # print(conv)
    return conv['value']

def  full_cache_generation(model_name, cache_dir, tokenizer, length):
    conversation_id = '1jjEIai'
    conversations, num_turns = load_conversation_from_sharedgpt('/data/home/gexr/H2O/sharegpt_gpt4.json', conversation_id)
    prompt_text = ''
    for i in range(0,num_turns,2):
        prompt_text += format_conversation(conversations, i)
        
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        model.half().eval().cuda()
        input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

        generate_ids = model.generate(input_ids, max_new_tokens=length, use_cache=True)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        # print("################## Generated Context with Full Cache ###################")
        print(result,'\n')
        prompt_text += result

    return prompt_text

def  full_cache_generation_with_gpt_resp(model_name, cache_dir, tokenizer, length):
    conversation_id = '1jjEIai'
    conversations, num_turns = load_conversation_from_sharedgpt('/data/home/gexr/H2O/sharegpt_gpt4.json', conversation_id)
    prompt_text = ''
    for i in range(0,num_turns,2):
        prompt_text += format_conversation(conversations, i)
        
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        model.half().eval().cuda()
        input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)

        generate_ids = model.generate(input_ids, max_new_tokens=length, use_cache=True)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        print(result,'\n')

        prompt_text += format_conversation(conversations,i+1)
    return prompt_text

def  heavy_hitter_generation_with_gpt_resp(model_name,model_arch, cache_dir, tokenizer, length,config):
    conversation_id = '1jjEIai'
    conversations, num_turns = load_conversation_from_sharedgpt('/data/home/gexr/H2O/sharegpt_gpt4.json', conversation_id)
    prompt_text = ''
    for i in range(0,num_turns,2):
        prompt_text += format_conversation(conversations, i)
        
        input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        checkpoint = copy.deepcopy(model.state_dict())
        model = ENABLE_Heavy_Hitter_FUNCTIONS[model_arch](model, config)
        model.load_state_dict(checkpoint)
        model.half().eval().cuda()
        
        generate_ids_hh = model.generate(input_ids, max_new_tokens=length, use_cache=True)
        result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(result_hh)
        prompt_text += format_conversation(conversations,i+1)
            
    return prompt_text

def  heavy_hitter_generation(model_name,model_arch, cache_dir, tokenizer, length,config):
    conversation_id = '1jjEIai'
    conversations, num_turns = load_conversation_from_sharedgpt('/data/home/gexr/H2O/sharegpt_gpt4.json', conversation_id)
    prompt_text = ''
    for i in range(0,num_turns,2):
        prompt_text += format_conversation(conversations, i)
        
        input_ids = tokenizer(prompt_text, add_special_tokens=False, return_tensors='pt').input_ids.to(model.device)
        model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=cache_dir)
        checkpoint = copy.deepcopy(model.state_dict())
        model = ENABLE_Heavy_Hitter_FUNCTIONS[model_arch](model, config)
        model.load_state_dict(checkpoint)
        model.half().eval().cuda()
        
        generate_ids_hh = model.generate(input_ids, max_new_tokens=length, use_cache=True)
        result_hh = tokenizer.batch_decode(generate_ids_hh, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
        print(result_hh)
        prompt_text += result_hh
    return prompt_text


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_arch", type=str, default='llama')
    parser.add_argument("--model_name", type=str, default='/data/home/public/weight/llama2-7b-chat')
    parser.add_argument("--cache_dir", type=str, default='../../checkpoint/')

    parser.add_argument("--heavy_ratio", type=float, default=0.1)
    parser.add_argument("--recent_ratio", type=float, default=0.1)

    parser.add_argument("--length", type=int, default=64)

    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    args = parser.parse_args()

    args.device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    args.n_gpu = 0 if args.no_cuda else torch.cuda.device_count()

    logger.warning(f"device: {args.device}, n_gpu: {args.n_gpu}, 16-bits training: {args.fp16}")
    set_seed(args)


    model_name = args.model_name
    config = AutoConfig.from_pretrained(model_name, cache_dir=args.cache_dir)
    config.heavy_ratio = args.heavy_ratio
    config.recent_ratio = args.recent_ratio

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True, cache_dir=args.cache_dir)

    ######## Generate with Full Cache
    result = full_cache_generation_with_gpt_resp(model_name, args.cache_dir, tokenizer, args.length)
    print(result)
    
    ######### Enable HH
    heavy_hitter_result = heavy_hitter_generation_with_gpt_resp(model_name, args.model_arch, args.cache_dir, tokenizer, args.length, config)
    print(heavy_hitter_result)
  


if __name__ == "__main__":
    main()
    