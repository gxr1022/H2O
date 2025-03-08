import torch
from transformers import AutoTokenizer
from transformers.models.llama.configuration_llama import LlamaConfig
from typing import List, Any, Optional, Union, Tuple
import numpy as np
import hashlib
from utils import STR_DTYPE_TO_TORCH_DTYPE, get_dtype_size



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

    def _verify_cache_dtype(self) -> None:
        if self.cache_dtype == "auto":
            pass
        elif self.cache_dtype in ("fp8", "fp8_e4m3", "fp8_e5m2"):
            print(
                "Using fp8 data type to store kv cache. It reduces the GPU "
                "memory footprint and boosts the performance. "
                "Meanwhile, it may cause accuracy drop without a proper "
                "scaling factor")
        else:
            raise ValueError(f"Unknown kv cache dtype: {self.cache_dtype}")
    
      

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
            self.num_gpu_blocks, "cuda:0")
        # self.cpu_cache = self._allocate_kv_cache(self.num_cpu_blocks, "cpu")

    '''
    # 访问第一层的 key cache 中第3个块的第2个token的第1个头
    layer_0_key = kv_cache[0][0, 3, 2, 1, :]
    '''
    def _allocate_kv_cache(
        self,
        num_blocks: int,
        device: str,
    ) -> List[torch.Tensor]:

        """Allocates KV cache on the specified device."""
        kv_cache_shape = (2,num_blocks, self.block_size, self.num_kv_heads, self.head_size)
        kv_cache: List[torch.Tensor] = []

        alloc_shape = kv_cache_shape

        for _ in range(self.num_attention_layers):
            layer_kv_cache = torch.zeros(alloc_shape,
                                         dtype=self.dtype,
                                         pin_memory=False,
                                         device=device)

            kv_cache.append(layer_kv_cache.view(kv_cache_shape))
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


    def get_cache_block_size_bytes(
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
        total = num_attention_layers * cache_config.block_size * (key_cache_entry + value_cache_entry)

        dtype_size = get_dtype_size(dtype)
        return dtype_size * total

    # def get_cached_kvs(self, block_ids: List[List[int]]) -> Tuple[torch.Tensor, torch.Tensor]:

    def append_kv(self, block_id:int, offset:int, layer_idx:int, key_states: torch.Tensor, value_states: torch.Tensor) -> None:
        if offset >= self.block_size:
            block_id += 1
            offset = 0
            new_allocated_blocks = 1
        else:
            new_allocated_blocks = 0
        self.gpu_cache[layer_idx][0, block_id, offset, :, :] = key_states
        self.gpu_cache[layer_idx][1, block_id, offset, :, :] = value_states
        return new_allocated_blocks
        
    def get_cached_kv(self, block_ids: List[int], layer_idx: int, offset: int, key:int):
        if not block_ids:
            return None
        
        total_len = (len(block_ids) - 1) * self.block_size + offset
        
        # 使用index_select一次性选择所需的数据
        indices = torch.arange(
            block_ids[0] * self.block_size,
            (block_ids[-1]) * self.block_size + offset,
            device=self.gpu_cache[layer_idx].device
        )
        return self.gpu_cache[layer_idx][key].reshape(-1, self.num_heads, self.head_dim)[indices].reshape(1, -1, self.num_heads, self.head_dim)
    

# class TokenIdCache:

#     """
#     Args:
#         prev_block (Block): The previous block in the sequence.
#         token_ids (List[int]): The initial token IDs to be stored in the block.
#         block_size (int): The maximum number of token IDs that can be stored in
#             the block.
#         allocator (BlockAllocator): The block allocator associated with this
#             block.
#         block_id (Optional[int], optional): The physical block index
#             of this block. Defaults to None, which means no allocation has been
#             made.
#     """


#     def __init__(self, cache_config: CacheConfig, model_config: LlamaConfig):
#         self.cache_config = cache_config
#         self.model_config = model_config
   
#     def __init__(self,
#                  prev_block: Optional[Block],
#                  token_ids: List[int],
#                  block_size: int,
#                  allocator: BlockAllocator,
#                  block_id: Optional[int] = None,
#                  extra_hash: Optional[int] = None):
#         self._token_ids: List[int] = []
#         self._block_size = block_size
#         self._prev_block = prev_block
#         self._block_id = block_id
#         self._allocator = allocator
#         self._append_token_ids(token_ids)

#     def _append_token_ids(self, token_ids: List[int]) -> None:
#         """Appends the given token IDs to the block
#         Args:
#             token_ids (List[int]): The token IDs to be appended to the block.
#         """
#         if len(token_ids) == 0:
#             return

#         assert len(token_ids) <= self.num_empty_slots

#         self._token_ids.extend(token_ids)

#     @property
#     def block_id(self) -> Optional[int]:
#         return self._block_id

#     def block_id(self, value: Optional[int]) -> None:
#         self._block_id = value

#     @property
#     def is_full(self) -> bool:
#         return self.num_empty_slots == 0

#     @property
#     def num_empty_slots(self) -> int:
#         return self._block_size - len(self.token_ids)

#     @property
#     def token_ids(self) -> List[int]:
#         return self._token_ids

#     @property
#     def num_tokens_total(self) -> int:
#         raise NotImplementedError(
#             "num_tokens_total is not used for naive block")

#     @property
#     def block_size(self) -> int:
#         return self._block_size

#     @property
#     def prev_block(self) -> Optional["Block"]:
#         return self._prev_block



   
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


    # def cleanup_old_sessions(self, max_sessions: int = 1000):
    #     """Clean up old sessions (when session count exceeds limit)"""
    #     if len(self.sessions) <= max_sessions:
    #         return
            
    #     # Sort by last used time
    #     sorted_sessions = sorted(
    #         self.sessions.items(), 
    #         key=lambda x: x[1].last_used
    #     )
        
    #     # Remove oldest sessions
    #     sessions_to_remove = sorted_sessions[:-max_sessions]
    #     for session_id, session in sessions_to_remove:
    #         self._remove_session(session_id)

    # def _remove_session(self, session_id: int):
    #     """Remove specified session"""
    #     session = self.sessions[session_id]
    #     # Clear indices
    #     for prefix_hash, sessions in self.session_index.items():
    #         if session_id in sessions:
    #             sessions.remove(session_id)
    #     # Delete session
    #     del self.sessions[session_id]

 