from utils_hh.prefix_cache import SessionKVCache, CacheEngine, CacheConfig
from utils_hh.llama import LlamaForCausalLM
import torch
from typing import Optional, List   
from torch.nn.utils.rnn import pad_sequence

class PrefixCacheScheduler:
    def __init__(self, model: LlamaForCausalLM, cache_config: CacheConfig):
        self.model = model
        self.session_kv_cache = SessionKVCache(cache_config, model.config)
        # self.cache_engine = model.cache_engine
        self.cache_config = cache_config

    def generate(
        self,
        model,
        tokenizer,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        **generate_kwargs
    ):
        # match prefix cache
        batch_size = input_ids.shape[0]
        if self.cache_config and self.cache_config.enable_prefix_caching:
            new_input_ids_list = []
            prefix_cache_block_ids_list = []
            prefix_lengths_list = []

            for batch_idx in range(batch_size):
                sequence = input_ids[batch_idx]  
                if self.session_kv_cache.sessions:
                    matching_session = self.session_kv_cache.find_matching_session(sequence.tolist())
                else:
                    matching_session = self.session_kv_cache.create_new_session(sequence.tolist(), []) 
                
                if matching_session:
                    cached_blocks = matching_session.block_ids
                    prefix_cache_block_ids_list.append(cached_blocks)
                    match_length = matching_session.compute_match_length(sequence.tolist())

                    new_tokens = sequence[match_length:]
                    prefix_lengths_list.append(match_length)
                else:
                    new_tokens = sequence 
                    prefix_cache_block_ids_list.append(None)
                    prefix_lengths_list.append(None)
                new_input_ids_list.append(new_tokens)

            input_ids = pad_sequence(new_input_ids_list, batch_first=True, padding_value=0)
        
        # deal with attention mask
        if attention_mask is not None:
            new_attention_mask = self._adjust_attention_mask(
                attention_mask, 
                prefix_lengths_list,
                new_input_ids_list
            )
        else:
            new_attention_mask = None
        
        # calculated new KV cache positions 
        new_kv_cache_positions = []
        for b in range(batch_size):
            if prefix_cache_block_ids_list[b] is not None:
                block_id = prefix_lengths_list[b] // self.cache_config.block_size
                offset = prefix_lengths_list[b] % self.cache_config.block_size
                new_kv_cache_positions.append((block_id, offset))
            else:
                new_kv_cache_positions.append(None)
        
        generate_kwargs = {
        "max_new_tokens": 1024,
        "use_cache": True,
        "prefix_cache_block_ids_list": prefix_cache_block_ids_list,
        "new_kv_cache_positions": new_kv_cache_positions,
        "prefix_lengths_list": prefix_lengths_list
    }    
        generate_ids = model.generate(input_ids, **generate_kwargs)
        result = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        
     
        return result

    def _adjust_attention_mask(
        self,
        attention_mask: torch.Tensor,
        prefix_lengths: List[int],
        new_input_ids_list: List[torch.Tensor]
    ) -> torch.Tensor:
        batch_size = len(prefix_lengths)
        max_new_length = max(len(ids) for ids in new_input_ids_list)

        new_attention_mask = torch.zeros(
            (batch_size, max_new_length),
            dtype=attention_mask.dtype,
            device=attention_mask.device
        )
        
        for batch_idx in range(batch_size):
            new_length = len(new_input_ids_list[batch_idx])
            new_attention_mask[batch_idx, :new_length] = 1
            
        return new_attention_mask
