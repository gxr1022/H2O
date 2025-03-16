from utils_hh.prefix_cache import SessionKVCache, CacheEngine, CacheConfig
from utils_hh.llama import LlamaForCausalLM, MyLlamaForCausalLM
import torch
from typing import Optional, List   
from torch.nn.utils.rnn import pad_sequence

class PrefixCacheScheduler:
    def __init__(self, model: LlamaForCausalLM, cache_config: CacheConfig):
        self.model = model
        self.session_kv_cache = SessionKVCache(cache_config, model.config)
        # self.cache_engine = model.cache_engine
        self.cache_config = cache_config

    def deal_with_prefix_cache(self,input_ids=None,attention_mask=None):
        # match prefix cache
        batch_size = input_ids.shape[0]
        print("deal_with_prefix_cache:input_ids",input_ids)
        # layer_num = self.model.config.num_hidden_layers
        
        if self.cache_config and self.cache_config.enable_prefix_caching:
            new_input_ids_list = []
            prefix_cache_block_ids_list = [[0] for _ in range(batch_size)]
            prefix_lengths_list = [0 for _ in range(batch_size)] 

            for batch_idx in range(batch_size):
                sequence = input_ids[batch_idx]  
                if self.session_kv_cache.sessions != []:
                    matching_session = self.session_kv_cache.find_matching_session(sequence.tolist())
                    if matching_session:
                        cached_blocks = matching_session.block_ids
                        prefix_cache_block_ids_list[batch_idx] = cached_blocks
                        match_length = matching_session.compute_match_length(sequence.tolist())
                        new_tokens = sequence[match_length:]
                        prefix_lengths_list[batch_idx] = match_length #The token length of the matching session
                    else:
                        new_tokens = sequence 
                        # prefix_cache_block_ids_list[batch_idx] = []
                        prefix_lengths_list[batch_idx] = 0

                    new_input_ids_list.append(new_tokens)
                    input_ids = pad_sequence(new_input_ids_list, batch_first=True, padding_value=0)
                else:
                    matching_session = self.session_kv_cache.create_new_session(sequence.tolist(), []) 
                    new_input_ids_list.append(sequence)
            # calculated new KV cache positions 
            new_kv_cache_positions = [None for _ in range(batch_size)]

            for b in range(batch_size):
                if prefix_cache_block_ids_list[b] != []:
                    block_id = prefix_lengths_list[b] // self.cache_config.block_size
                    offset = prefix_lengths_list[b] % self.cache_config.block_size
                    new_kv_cache_positions[b] = [block_id, offset]
                else:
                    new_kv_cache_positions[b] = [0,0]

            # print("new_input_ids_list",new_input_ids_list) 
            # print("prefix_lengths_list",prefix_lengths_list)
            # deal with attention mask
   
            attention_mask = self._adjust_attention_mask(
                attention_mask=attention_mask, 
                prefix_lengths=prefix_lengths_list,
                new_input_ids_list=new_input_ids_list
            )
            return attention_mask, prefix_cache_block_ids_list, new_kv_cache_positions, prefix_lengths_list
        
    def generate(
        self,
        model,
        tokenizer,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs
    ):  
        # print("generate:input_ids",input_ids)

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        
        attention_mask, prefix_cache_block_ids_list, new_kv_cache_positions, prefix_lengths_list = self.deal_with_prefix_cache(input_ids=input_ids,attention_mask=attention_mask)
        print(f"generate: attention_mask sum {attention_mask.sum().item()}")
         
        generate_kwargs = {
        "max_new_tokens": 2,
        "use_cache": False,
        "prefix_cache_block_ids_list": prefix_cache_block_ids_list,
        "new_kv_cache_positions": new_kv_cache_positions,
        "prefix_lengths_list": prefix_lengths_list
    }    
        generate_ids = model.generate(input_ids, attention_mask=attention_mask, **generate_kwargs)
        # print("Generated IDs:", generate_ids)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
     
        return result

    def _adjust_attention_mask(
        self,
        attention_mask: torch.Tensor,
        prefix_lengths: List[int],
        new_input_ids_list: List[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = len(prefix_lengths)
        max_new_length = max(len(ids)+prefix_lengths[i] for i, ids in enumerate(new_input_ids_list)) 
        new_attention_mask = torch.zeros(
            (batch_size, max_new_length), # to be determined
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        
        for batch_idx in range(batch_size):
            new_length = len(new_input_ids_list[batch_idx])+prefix_lengths[batch_idx]
            new_attention_mask[batch_idx, :new_length] = 1
            
        return new_attention_mask
