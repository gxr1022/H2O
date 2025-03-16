import torch
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import List, Any, Optional, Union, Tuple
import numpy as np
import hashlib
from utils_hh.utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size



class CacheConfig:
    """Configuration for the KV cache.

    Args:
        block_size: Size of a cache block in number of tokens.
        gpu_memory_utilization: Fraction of GPU memory to use for the
            vLLM execution.
        swap_space: Size of the CPU swap space per GPU (in GiB).
        cache_dtype: Data type for kv cache storage.
        num_gpu_blocks_override: Number of GPU blocks to use. This overrides the
            profiled num_gpu_blocks if specified. Does nothing if None.
        enable_prefix_caching: Whether to enable prefix caching.
    """

    def compute_hash(self) -> str:
        """
        WARNING: Whenever a new field is added to this config,
        ensure that it is included in the factors list if
        it affects the computation graph.

        Provide a hash that uniquely identifies all the configs
        that affect the structure of the computation
        graph from input ids/embeddings to the final hidden states,
        excluding anything before input ids/embeddings and after
        the final hidden states.
        """
        factors: List[Any] = []
        factors.append(self.cache_dtype)
        # `cpu_offloadk_gb` does not use `torch.compile` yet.
        hash_str = hashlib.md5(str(factors).encode()).hexdigest()
        return hash_str

    def __init__(
        self,
        block_size: int,
        gpu_memory_utilization: float,
        # swap_space: float,
        cache_dtype: str,
        num_gpu_blocks_override: Optional[int] = None,
        enable_prefix_caching: bool = False,
    ) -> None:
        self.block_size = block_size
        # default 0.9
        self.gpu_memory_utilization = gpu_memory_utilization
        # self.swap_space_bytes = swap_space * 1024 * 1024 * 1024
        self.cache_dtype = cache_dtype
        self.num_gpu_blocks_override = num_gpu_blocks_override
        self.enable_prefix_caching = enable_prefix_caching

        self._verify_args()
        self._verify_cache_dtype()

        # Will be set after profiling.
        self.num_gpu_blocks: Optional[int] = None
        # self.num_cpu_blocks: Optional[int] = None


    def metrics_info(self):
        # convert cache_config to dict(key: str, value: str) for prometheus
        # metrics info
        return {key: str(value) for key, value in self.__dict__.items()}

    def _verify_args(self) -> None:
        if self.gpu_memory_utilization > 1.0:
            raise ValueError(
                "GPU memory utilization must be less than 1.0."
                f"{self.gpu_memory_utilization}.")

    # def _verify_cache_dtype(self) -> None:
    #     if self.cache_dtype == "auto":
    #         pass
    #     elif self.cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
    #         print(
    #             "Using fp8 data type to store kv cache. It reduces the GPU "
    #             "memory footprint and boosts the performance. "
    #             "Meanwhile, it may cause accuracy drop without a proper "
    #             "scaling factor")
    #     else:
    #         raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")

    def _verify_cache_dtype(self):
        valid_dtypes = ["float16", "float32", torch.float16, torch.float32]
        if self.cache_dtype not in valid_dtypes:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}. Must be one of {valid_dtypes}") 
        

class CacheEngine:
    """Manages the KV cache.

    This class is responsible for initializing and managing the GPU and CPU KV
    caches. It also provides methods for performing KV cache operations, such
    as swapping and copying.
    """

    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: LlamaConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config

        self.num_kv_heads = model_config.num_attention_heads
        self.head_size = model_config.hidden_size // self.num_kv_heads

        
        self.num_attention_layers = model_config.num_hidden_layers
        
        self.block_size = cache_config.block_size
        self.num_gpu_blocks = cache_config.num_gpu_blocks
        if self.num_gpu_blocks is None:
            self.num_gpu_blocks = self.determine_num_available_blocks()
        
        # self.num_cpu_blocks = cache_config.num_cpu_blocks

        if cache_config.cache_dtype == "auto":
            self.dtype = Union[str, torch.dtype] 
        else:
            self.dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        # Initialize the cache.
        self.gpu_cache = self._allocate_kv_cache(
            self.num_gpu_blocks, torch.device("cuda:0"))
        # self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    '''
    # 访问第一层的 key cache 中第3个块的第2个token的第1个头
    layer_0_key = kv_cache[0][3, 2, 1, :]
    '''
    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: torch.device,
    ) -> List[torch.Tensor]:

        """Allocates KV cache on the specified device."""
        # kv_cache_shape = (2,num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        kv_cache_shape = (2, num_blocks, self.num_kv_heads, self.block_size, self.head_size)
        print("CacheEngine:_allocate_kv_cache",kv_cache_shape)
        kv_cache: List[torch.Tensor] = []

        alloc_shape = kv_cache_shape

        for _ in range(self.num_attention_layers):
            layer_kv_cache = torch.zeros(alloc_shape,
                                         dtype=self.dtype,
                                         pin_memory=False,
                                         device=device)

            kv_cache.append(layer_kv_cache.view(kv_cache_shape))
            # print("layer_kv_cache",layer_kv_cache.shape)
        return kv_cache

    # def swap_in(self, src_to_dst: torch.Tensor) -> None:
    #     for i in range(self.num_attention_layers):
    #         self.attn_backend.swap_blocks(self.cpu_cache[i], self.gpu_cache[i],
    #                                       src_to_dst)

    # def swap_out(self, src_to_dst: torch.Tensor) -> None:
    #     for i in range(self.num_attention_layers):
    #         self.attn_backend.swap_blocks(self.gpu_cache[i], self.cpu_cache[i],
    #                                       src_to_dst)

    # def copy(self, src_to_dsts: torch.Tensor) -> None:
    #     self.attn_backend.copy_blocks(self.gpu_cache, src_to_dsts)
    
    @torch.inference_mode()
    def determine_num_available_blocks(self) -> Tuple[int, int]:

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        free_memory_pre_profile, total_gpu_memory = torch.cuda.mem_get_info()
        
        
        non_kv_cache_memory = 14 * 1024 * 1024 * 1024 + 5 * 1024 * 1024 * 1024
            
        memory_for_current_instance = total_gpu_memory * self.cache_config.gpu_memory_utilization
        available_kv_cache_memory = (memory_for_current_instance - non_kv_cache_memory)
        
        cache_block_size = self.get_cache_block_size_bytes(self.cache_config, self.model_config)
        self.num_gpu_blocks = int(available_kv_cache_memory // cache_block_size)
        # self.num_cpu_blocks = int(self.cache_config.swap_space_bytes // cache_block_size)
        return self.num_gpu_blocks

    def get_cache_block_size_bytes(
        self,
        cache_config: CacheConfig,
        model_config: LlamaConfig,
    ) -> int:
        head_size = model_config.hidden_size // model_config.num_attention_heads
        num_heads = model_config.num_attention_heads
        num_attention_layers = model_config.num_hidden_layers

        if cache_config.cache_dtype == "auto":
            dtype = model_config.dtype
        else:
            dtype = STR_DTYPE_TO_TORCH_DTYPE[cache_config.cache_dtype]

        key_cache_entry = num_heads * head_size
        value_cache_entry = key_cache_entry 

        # ?? Why num_attention_layers need to be multiplied?
        # Because every layer has a kv cache, and the number of gpu blocks is the same for all layers, so we need to multiply the number of layers
        total = num_attention_layers * cache_config.block_size * (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    
    def append_kv(
    self,
    block_id: int,
    offset: int,
    layer_idx: int,
    key_states: torch.Tensor,   # (num_head, seq_len, head_dim)
    value_states: torch.Tensor  # (num_head, seq_len, head_dim)
) -> Tuple[int, int, int]:
        """
        将 (num_head, seq_len, head_dim) 形状的 key/value states 写入到
        (2, block_id, num_head, block_size, head_dim) 的缓存中, 支持部分填充：

        - 当当前 block 剩余空间不足时，会先填满这一 block
        - 剩下的内容挪到下一个 block 继续写，必要时可连续分配多个 block

        参数:
            block_id (int):        当前可用的 block id
            offset (int):         当前 block 尚未写入的起始位置
            layer_idx (int):      指定写入第几层
            key_states (Tensor):  (num_head, seq_len, head_dim)
            value_states (Tensor):(num_head, seq_len, head_dim)

        返回值:
            (final_block_id, final_offset, blocks_allocated)
                final_block_id:   写完后最终所在的 block
                final_offset:     写完后在该 block 的 offset
                blocks_allocated: 在写入过程中新增分配了多少个 block
        """
        seq_len = key_states.size(1)
        total_tokens_to_place = seq_len
        start_idx = 0

        blocks_allocated = 0
        final_block_id = block_id
        final_offset = offset

        while total_tokens_to_place > 0:
            leftover = self.block_size - final_offset
            tokens_to_fill = min(leftover, total_tokens_to_place)
           
            self.gpu_cache[layer_idx][0, final_block_id, :, final_offset : final_offset + tokens_to_fill, :] = (
                key_states[:, start_idx : start_idx + tokens_to_fill, :]
            )
            self.gpu_cache[layer_idx][1, final_block_id, :, final_offset : final_offset + tokens_to_fill, :] = (
                value_states[:, start_idx : start_idx + tokens_to_fill, :]
            )

            final_offset += tokens_to_fill
            start_idx += tokens_to_fill
            total_tokens_to_place -= tokens_to_fill

            if final_offset == self.block_size and total_tokens_to_place > 0:
                final_block_id += 1
                final_offset = 0
                blocks_allocated += 1

        return final_block_id, final_offset, blocks_allocated

    def get_cached_kv(self, block_ids: List[int], layer_idx: int,block_id: int, offset: int, key: int):

        if not block_ids:
            return None
        # if(layer_idx == 0 and key == 0):
        #     print("block_ids",block_ids)
        #     print("offset",offset)
        start_block = 0
        end_block   = block_id 

        start_pos = 0
        end_pos   = (end_block) * self.block_size + offset  
        length    = end_pos - start_pos                   

        # 1) 先“view”成 [num_kv_heads, total_length, head_size]
        #    注意：要确保 self.gpu_cache[layer_idx][key] 的形状能被这样整除
        reshaped = self.gpu_cache[layer_idx][key][0:end_block+1, :, :, :].permute(1, 0, 2, 3).contiguous().view(
            self.num_kv_heads,
            -1,               
            self.head_size
        )
       

        sliced_view = reshaped.narrow(dim=1, start=start_pos, length=length)
        # if(layer_idx == 0):
        #     print("self.gpu_cache[layer_idx][key][end_block, :, :, :]",self.gpu_cache[layer_idx][key][end_block, :, :, :])
        #     print("reshaped",reshaped)
        #     print("reshaped.shape",reshaped.shape)
        #     print("sliced_view",sliced_view)
        #     print("sliced_view.shape",sliced_view.shape)

        
        final_tensor = sliced_view.unsqueeze(0)
        # if(layer_idx == 0):
        #     print("final_tensor[:, -1, :]",final_tensor[:, -1, :])
        #     print("final_tensor.shape",final_tensor.shape) # 形状: [num_kv_heads, length, head_size]
        return final_tensor

class SessionInfo:
    def __init__(self, session_id: int):
        self.session_id = session_id
        self.block_ids = []  # Sequentially stored block ids
        self.token_ids = []  # Complete token sequence
        self.total_tokens = 0
        self.prefix_hash = None

    def update_session(self, token_ids: List[int], block_ids: List[int]) -> None:
        """Update session with new tokens and blocks"""
        self.token_ids.extend(token_ids)
        self.block_ids.extend(block_ids)
        self.total_tokens += len(token_ids)
        self.prefix_hash = self._compute_prefix_hash()

    def _compute_prefix_hash(self) -> str:
        """Compute hash value for the current token sequence"""
        return hashlib.md5(str(self.token_ids).encode()).hexdigest()

    def compute_match_length(self, other_tokens: List[int]) -> int:
        """Calculate matching length with another token sequence"""
        i = 0
        while i < len(other_tokens) and i < len(self.token_ids) and other_tokens[i] == self.token_ids[i]:
            i += 1
        return i

    def matches_prefix(self, token_ids: List[int]) -> bool:
        """Check if given tokens match this session's prefix"""
        target_hash = hashlib.md5(str(token_ids[:len(self.token_ids)]).encode()).hexdigest()
        return target_hash == self.prefix_hash

class SessionKVCache:
    def __init__(self, cache_config: CacheConfig, model_config: LlamaConfig):
        # self.cache_engine = CacheEngine(cache_config, model_config)
        self.sessions: List[SessionInfo] = [] 

    def find_matching_session(self, token_ids: List[int]) -> Optional[SessionInfo]:
        """Find matching session"""
        candidate_sessions = set()
        if len(self.sessions) == 0:
            return None
        # Filter sessions by hash matching
        for session in self.sessions:
            if session.matches_prefix(token_ids):
                candidate_sessions.add(session)
        
        if len(candidate_sessions) == 0:
            return None
        elif len(candidate_sessions) == 1:
            return candidate_sessions.pop()
        else:
            # Find best match among candidates
            best_match = None
            max_match_length = 0
            for session in candidate_sessions:
                match_length = session.compute_match_length(token_ids)
                if match_length > max_match_length:
                    max_match_length = match_length
                    best_match = session
            return best_match

    def create_new_session(self, token_ids: List[int], block_ids: List[int]) -> SessionInfo:
        """Create new session"""
        session_id = len(self.sessions)
        session = SessionInfo(session_id)
        session.update_session(token_ids, block_ids)
        self.sessions.append(session) 
        return session

    # def get_session_kv_cache(self, session: SessionInfo) -> List[torch.Tensor]:
    #     """Retrieve KV cache for entire session"""
    #     kv_caches = []
    #     for layer_idx in range(self.cache_engine.num_attention_layers):
    #         layer_cache = []
    #         for block_id in session.block_ids:
    #             block_cache = self.cache_engine.gpu_cache[layer_idx][:, block_id, :, :, :]
    #             layer_cache.append(block_cache)
    #         kv_caches.append(torch.cat(layer_cache, dim=1))
    #     return kv_caches


    def cleanup_old_sessions(self, max_sessions: int = 1000):
        """Clean up old sessions (when session count exceeds limit)"""
        if len(self.sessions) <= max_sessions:
            return
            
        # Sort by last used time
        sorted_sessions = sorted(
            self.sessions.items(), 
            key=lambda x: x[1].last_used
        )
        
        # Remove oldest sessions
        sessions_to_remove = sorted_sessions[:-max_sessions]
        for session_id, session in sessions_to_remove:
            self._remove_session(session_id)

    def _remove_session(self, session_id: int):
        """Remove specified session"""
        session = self.sessions[session_id]
        # Clear indices
        for prefix_hash, sessions in self.session_index.items():
            if session_id in sessions:
                sessions.remove(session_id)
        # Delete session
        del self.sessions[session_id]

 