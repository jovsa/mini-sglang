# KV Cache Breakdown: Complete System Explanation

This document provides a detailed breakdown of how KV cache works in mini-sglang, with references to the actual implementation code.

## Table of Contents

1. [Overview](#overview)
2. [Phase 1: Request Arrival & Cache Matching](#phase-1-request-arrival--cache-matching)
3. [Phase 2: Cache Allocation](#phase-2-cache-allocation)
4. [Phase 3: Forward Pass - KV Cache Usage](#phase-3-forward-pass---kv-cache-usage)
5. [Phase 4: Decode Phase - Incremental Updates](#phase-4-decode-phase---incremental-updates)
6. [Phase 5: Cache Cleanup](#phase-5-cache-cleanup)
7. [Memory Layout](#memory-layout)
8. [Code References](#code-references)

---

## Overview

The KV cache system in mini-sglang enables efficient LLM inference by:
- **Caching computed key/value pairs** from attention layers
- **Sharing prefixes** across requests using a radix tree
- **Managing memory** through eviction of unused cache entries
- **Protecting active requests** from cache eviction

### Key Components

```
┌─────────────────────────────────────────────────────────────────┐
│                        Request Layer                             │
│  UserMsg ──────> Req                                            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ Uses
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Cache Management                             │
│  CacheManager ──> RadixCacheManager ──> RadixTree                │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ Maps to
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                         Storage                                  │
│  PageTable ──────> MHAKVCache (KV Cache)                        │
│  TokenPool                                                       │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     │ Points to
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                        Engine Layer                              │
│  Engine ──────> Attention Backend ──────> Reads/Writes KV Cache │
└─────────────────────────────────────────────────────────────────┘

Flow:
  UserMsg creates Req
  Req uses CacheManager
  CacheManager manages RadixCacheManager
  RadixCacheManager uses RadixTree
  Req maps to PageTable and TokenPool
  PageTable points to MHAKVCache
  Engine uses Attention Backend
  Attention Backend reads/writes MHAKVCache
```

**Key Files:**
- `python/minisgl/kvcache/radix_manager.py` - Radix tree cache manager
- `python/minisgl/kvcache/mha_pool.py` - KV cache storage
- `python/minisgl/scheduler/cache.py` - Cache allocation and management
- `python/minisgl/scheduler/prefill.py` - Request scheduling with cache
- `python/minisgl/engine/engine.py` - Forward pass execution

---

## Phase 1: Request Arrival & Cache Matching

### Flow Diagram

```
UserMsg
  │
  │ add_one_req(msg)
  ▼
PrefillManager
  │
  │ match_req(pending_req)
  ▼
CacheManager
  │
  │ match_prefix(input_ids)
  ▼
RadixManager
  │
  │ Traverse tree
  ▼
RadixTree
  │
  │ Found node + prefix_len
  │
  └─────────────────┐
                    │
                    ▼
            RadixManager
                    │
                    │ CacheHandle + match_indices
                    │
                    └─────────────────┐
                                      │
                                      ▼
                              CacheManager
                                      │
                                      │ CacheHandle (cached_len)
                                      │
                                      └─────────────────┐
                                                        │
                                                        ▼
                                                PrefillManager
                                                        │
                                                        │ lock(handle)
                                                        │
                                                        ▼
                                                CacheManager
                                                        │
                                                        │ lock_handle(handle)
                                                        │
                                                        ▼
                                                RadixManager
                                                        │
                                                        │ Increment ref_count
                                                        │
                                                        ▼
                                                RadixTree
```

### Step-by-Step Explanation

#### 1. Request Arrives

**Code:** `python/minisgl/scheduler/prefill.py:121-122`

```python
def add_one_req(self, req: UserMsg) -> None:
    self.pending_list.append(PendingReq(req.uid, req.input_ids, req.sampling_params))
```

A `UserMsg` arrives with `input_ids` (token sequence). It's converted to a `PendingReq` and added to the pending list.

#### 2. Cache Matching

**Code:** `python/minisgl/scheduler/cache.py:24-27`

```python
def match_req(self, req: PendingReq):
    input_len = req.input_len
    assert input_len > 0, "Input length must be greater than 0."
    return self.manager.match_prefix(req.input_ids[: input_len - 1])
```

The cache manager tries to find a matching prefix in the radix tree. Note: it matches `input_ids[:input_len - 1]` (excluding the last token, which will be processed).

**Code:** `python/minisgl/kvcache/radix_manager.py:116-127`

```python
def match_prefix(self, input_ids: torch.Tensor) -> Tuple[RadixCacheHandle, torch.Tensor]:
    node, prefix_len = self._walk(input_ids)
    if prefix_len == 0:
        return RadixCacheHandle(prefix_len, node), self.empty_tensor
    value_list: List[torch.Tensor] = []
    matched_node = node
    while not node.is_root():
        value_list.append(node.value)
        node = node.parent
    value_list.reverse()
    return RadixCacheHandle(prefix_len, matched_node), torch.cat(value_list)
```

The radix tree is traversed to find the longest matching prefix:
- **New request**: `prefix_len = 0`, returns empty match_indices
- **Prefix match**: `prefix_len > 0`, returns page indices for cached tokens

**Code:** `python/minisgl/kvcache/radix_manager.py:139-164` (the `_walk` method)

```python
def _walk(self, input_ids: torch.Tensor) -> Tuple[RadixTreeNode, int]:
    prefix_len = 0
    indice_len = len(input_ids)
    node = self.root_node
    tic = time.monotonic_ns()

    while prefix_len < indice_len:
        this_id = int(input_ids[prefix_len].item())
        if this_id not in node.children:
            return node, prefix_len

        node = node.children[this_id]
        match_len = node.get_match_len(input_ids[prefix_len:])
        prefix_len += match_len

        if match_len != node.length:
            node = node._split_at(match_len)
            return node, prefix_len

        node.timestamp = tic

    return node, prefix_len
```

#### 3. Handle Locking

**Code:** `python/minisgl/scheduler/cache.py:33-34`

```python
def lock(self, handle: BaseCacheHandle) -> None:
    self.manager.lock_handle(handle, unlock=False)
```

**Code:** `python/minisgl/kvcache/radix_manager.py:97-114`

```python
def lock_handle(self, handle: BaseCacheHandle, unlock: bool = False) -> None:
    assert isinstance(handle, RadixCacheHandle)
    node = handle.node
    if unlock:
        while not node.is_root():
            node.ref_count -= 1
            assert node.ref_count >= 0
            if node.ref_count == 0:
                self.evictable_size += node.length
                self.protected_size -= node.length
            node = node.parent
    else:
        while not node.is_root():
            if node.ref_count == 0:
                self.evictable_size -= node.length
                self.protected_size += node.length
            node.ref_count += 1
            node = node.parent
```

Locking a handle:
- Increments `ref_count` for all nodes in the path
- Moves entries from evictable to protected
- Prevents eviction while request is active

### Key Concepts

**cached_len**: Number of tokens already in cache
- `cached_len = 0`: New request, no cache
- `cached_len > 0`: Prefix match, reuse cache

**Radix Tree**: Efficient prefix matching
- O(n) traversal to find longest match
- Enables sharing of common prefixes
- Reference counting tracks usage

---

## Phase 2: Cache Allocation

### Flow Diagram

```
PrefillManager
  │
  │ allocate()
  ▼
TableManager
  │
  │ table_idx
  │
  └─────────────────┐
                    │
                    ▼
            PrefillManager
                    │
                    │ allocate(needed_len)
                    │
                    ▼
            CacheManager
                    │
        ┌───────────┴───────────┐
        │                       │
        │ Free slots available  │ Cache full
        │                       │
        ▼                       ▼
    FreePool              RadixManager
        │                       │
        │ Take from             │ evict(needed_size)
        │ free_slots            │
        │                       │ evicted_indices
        │                       │
        └───────────┬───────────┘
                    │
                    │ allocated_pages
                    │
                    ▼
            PrefillManager
                    │
                    │ Set page_table[table_idx]
                    │
                    ▼
            PageTable
```

### Step-by-Step Explanation

#### 1. Table Slot Allocation

**Code:** `python/minisgl/scheduler/table.py:15-16`

```python
def allocate(self) -> int:
    return self._free_slots.pop()
```

Each request gets a `table_idx` that maps to:
- `token_pool[table_idx]`: Token IDs for this request
- `page_table[table_idx]`: Page indices mapping tokens to cache pages

#### 2. Cache Page Allocation

**Code:** `python/minisgl/scheduler/cache.py:39-52`

```python
def allocate(self, needed_len: int) -> torch.Tensor:
    if needed_len <= (free_len := len(self._free_slots)):
        allocated = self._free_slots[:needed_len]
        self._free_slots = self._free_slots[needed_len:]
        return allocated

    # NOTE: len(evicted) + free_len >= needed_len
    evicted = self.manager.evict(needed_len - free_len)
    merged = torch.cat([self._free_slots, evicted])
    assert len(merged) >= needed_len, "Eviction did not free enough space."

    allocated = merged[:needed_len]
    self._free_slots = merged[needed_len:]
    return allocated
```

Allocation strategy:
1. **Free slots available**: Take from `_free_slots`
2. **Cache full**: Evict unused entries, then allocate

#### 3. Page Table Setup

**Code:** `python/minisgl/scheduler/prefill.py:54-61`

```python
table_idx = self.table_manager.allocate()
if cached_len > 0:  # NOTE: set the cached part
    device_ids = self.table_manager.token_pool[table_idx][:cached_len]
    page_entry = self.table_manager.page_table[table_idx][:cached_len]
    device_ids.copy_(req.input_ids[:cached_len].pin_memory(), non_blocking=True)
    page_entry.copy_(match_indices)
```

For prefix matches:
- Copy existing page indices for cached tokens
- Allocate new pages only for new tokens

### Memory Layout

```
Page Table Structure:
┌─────────────────────────────────────────┐
│ table_idx │ token_pos │ page_index     │
├───────────┼───────────┼────────────────┤
│     0     │     0     │       5        │  ← Request 1, token 0 → page 5
│     0     │     1     │       6        │  ← Request 1, token 1 → page 6
│     0     │     2     │       7        │
│     1     │     0     │       5        │  ← Request 2, token 0 → page 5 (shared!)
│     1     │     1     │       6        │  ← Request 2, token 1 → page 6 (shared!)
│     1     │     2     │       7        │  ← Request 2, token 2 → page 7 (shared!)
│     1     │     3     │       8        │  ← Request 2, token 3 → page 8 (new)
└─────────────────────────────────────────┘
```

---

## Phase 3: Forward Pass - KV Cache Usage

### Flow Diagram

```
Engine
  │
  │ forward()
  ▼
Model
  │
  │ forward(q, k, v, layer_id, batch)
  ▼
Attention
  │
  │ store_kv(k, v, out_loc, layer_id)
  ▼
KVCache
  │
  │ Write to pages[out_loc]
  │
  └─────────────────┐
                    │
                    ▼
            KVCache (internal)
                    │
                    │ Read from cache for cached tokens
                    │
                    └─────────────────┐
                                      │
                                      ▼
                              Attention
                                      │
                                      │ output
                                      │
                                      └─────────────────┐
                                                        │
                                                        ▼
                                                Model
                                                        │
                                                        │ logits
                                                        │
                                                        └─────────────────┐
                                                                          │
                                                                          ▼
                                                                  Engine
                                                                          │
                                                                          │ complete_one()
                                                                          │
                                                                          ▼
                                                                  Req
                                                                          │
                                                                          │ cached_len = device_len
                                                                          │
                                                                          ▼
                                                                  Req (updated)
```

### Step-by-Step Explanation

#### 1. Batch Preparation

**Code:** `python/minisgl/scheduler/scheduler.py:180-194`

```python
def _prepare_batch(self, batch: Batch) -> ForwardInput:
    needed_size = sum(r.extend_len for r in batch.reqs)
    batch.out_loc = self.cache_manager.allocate(needed_size)
    # ... prepare indices ...
    self.page_table.view(-1)[load_indices] = batch.out_loc
    self.engine.attn_backend.prepare_metadata(batch)
```

- Allocates pages for all new tokens in batch
- Sets up `out_loc` (output locations = page indices)
- Writes page indices to page table

#### 2. Forward Pass

**Code:** `python/minisgl/engine/engine.py:188-203`

```python
def forward_batch(self, batch: Batch, args: BatchSamplingArgs) -> ForwardOutput:
    assert torch.cuda.current_stream() == self.stream
    with self.ctx.forward_batch(batch):
        if self.graph_runner.can_use_cuda_graph(batch):
            logits = self.graph_runner.replay(batch)
        else:
            logits = self.model.forward()

    for req in batch.reqs:
        req.complete_one()

    next_tokens_gpu = self.sampler.sample(logits[: batch.size], args).to(torch.int32)
    # ...
```

The model forward pass processes only new tokens (from `cached_len` to `device_len`).

#### 3. KV Cache Storage

**Code:** `python/minisgl/attention/fa.py:49-65`

```python
def forward(
    self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, layer_id: int, batch: Batch
) -> torch.Tensor:
    metadata = batch.attn_metadata
    assert isinstance(metadata, FAMetadata)
    self.kvcache.store_kv(k, v, batch.out_loc, layer_id)
    return _fa_sgl_impl(
        q=q,
        k_cache=self.kvcache.k_cache(layer_id),
        v_cache=self.kvcache.v_cache(layer_id),
        page_table=metadata.page_table,
        cache_seqlens=metadata.cache_seqlens,
        # ...
    )
```

**Code:** `python/minisgl/kvcache/mha_pool.py:56-67`

```python
def store_kv(
    self, k: torch.Tensor, v: torch.Tensor, out_loc: torch.Tensor, layer_id: int
) -> None:
    from minisgl.kernel import store_cache

    store_cache(
        k_cache=self._k_buffer[layer_id].view(self._storage_shape),
        v_cache=self._v_buffer[layer_id].view(self._storage_shape),
        indices=out_loc,
        k=k,
        v=v,
    )
```

The `store_kv` method writes K and V tensors to the cache pages specified by `out_loc`.

#### 4. Update cached_len

**Code:** `python/minisgl/core.py:51-53`

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1
```

After processing, `cached_len` is updated to reflect that tokens are now cached.

### Key Concepts

**extend_len**: `device_len - cached_len`
- Number of new tokens to process
- Only these tokens need computation

**out_loc**: Page indices for new tokens
- Maps each new token to its cache page
- Used by `store_kv` to write to correct location

---

## Phase 4: Decode Phase - Incremental Updates

### Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│              Loop: Each decode step                         │
│                                                              │
│  DecodeManager                                              │
│    │                                                         │
│    │ schedule_next_batch()                                   │
│    ▼                                                         │
│  Engine                                                      │
│    │                                                         │
│    │ allocate(1)                                            │
│    ▼                                                         │
│  CacheManager                                               │
│    │                                                         │
│    │ new_page                                               │
│    │                                                         │
│    └─────────────────┐                                      │
│                      │                                      │
│                      ▼                                      │
│              Engine                                         │
│                      │                                      │
│                      │ page_table[table_idx][device_len]    │
│                      │   = new_page                         │
│                      ▼                                      │
│              PageTable                                      │
│                      │                                      │
│                      │ forward() (1 token)                  │
│                      ▼                                      │
│              Model                                           │
│                      │                                      │
│                      │ store_kv(k, v, new_page, layer_id)   │
│                      ▼                                      │
│              KVCache                                        │
│                      │                                      │
│                      │ complete_one()                       │
│                      ▼                                      │
│              Req                                            │
│                      │                                      │
│                      │ cached_len++, device_len++           │
│                      ▼                                      │
│              Req (updated)                                  │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Step-by-Step Explanation

#### 1. Decode Scheduling

**Code:** `python/minisgl/scheduler/decode.py:23-26`

```python
def schedule_next_batch(self) -> Batch | None:
    if not self.runnable:
        return None
    return Batch(reqs=list(self.running_reqs), phase="decode")
```

Decode manager schedules all running requests for decode phase.

#### 2. Single Token Processing

**Code:** `python/minisgl/core.py:51-53`

```python
def complete_one(self) -> None:
    self.cached_len = self.device_len
    self.device_len += 1
```

Each decode step:
- Processes one new token
- Allocates one new page
- Updates `cached_len` and `device_len`

#### 3. Incremental Cache Updates

The forward pass for decode:
- Processes only the new token (extend_len = 1)
- Stores KV to the newly allocated page
- Reads from cache for all previous tokens

### Key Concepts

**remain_len**: `max_device_len - device_len`
- Number of tokens remaining to generate
- Request completes when `remain_len == 0`

**Incremental Allocation**: One page per decode step
- Efficient memory usage
- Enables early stopping

---

## Phase 5: Cache Cleanup

### Flow Diagram

```
Scheduler
  │
  │ free_and_cache_finished_req(handle, input_ids, indices)
  ▼
CacheManager
  │
  │ insert_prefix(input_ids, indices)
  ▼
RadixManager
  │
  │ Insert new nodes
  ▼
RadixTree
  │
  │ unlock(handle)
  │
  │
  └─────────────────┐
                    │
                    ▼
            RadixManager
                    │
                    │ Decrement ref_count
                    │
                    ▼
            RadixTree
                    │
        ┌───────────┴───────────┐
        │                       │
        │ ref_count == 0        │ ref_count > 0
        │                       │
        ▼                       │
    Mark as evictable           │
        │                       │
        └───────────┬───────────┘
                    │
                    │ _free(unused_indices)
                    │
                    ▼
            CacheManager
                    │
                    │ free(table_idx)
                    │
                    ▼
            TableManager
```

### Step-by-Step Explanation

#### 1. Insert Prefix into Radix Tree

**Code:** `python/minisgl/scheduler/cache.py:54-62`

```python
def free_and_cache_finished_req(
    self,
    old_handle: BaseCacheHandle,
    input_ids: torch.Tensor,
    indices: torch.Tensor,
) -> None:
    in_cache_len = self.manager.insert_prefix(input_ids, indices)
    self._free(indices[old_handle.cached_len : in_cache_len])
    self.unlock(old_handle)
```

**Code:** `python/minisgl/kvcache/radix_manager.py:129-137`

```python
def insert_prefix(self, input_ids: torch.Tensor, indices: torch.Tensor) -> int:
    node, prefix_len = self._walk(input_ids)
    assert prefix_len <= len(input_ids)
    if prefix_len < len(input_ids):
        new_node = RadixTreeNode()
        new_node.set_key_value(input_ids[prefix_len:], indices[prefix_len:])
        new_node.set_parent(node)
        self.evictable_size += new_node.length
    return prefix_len
```

Inserts the request's prefix into the radix tree for future reuse.

#### 2. Unlock Handle

**Code:** `python/minisgl/kvcache/radix_manager.py:97-114` (unlock path)

```python
if unlock:
    while not node.is_root():
        node.ref_count -= 1
        assert node.ref_count >= 0
        if node.ref_count == 0:
            self.evictable_size += node.length
            self.protected_size -= node.length
        node = node.parent
```

Unlocking:
- Decrements `ref_count` for all nodes
- When `ref_count == 0`, entry becomes evictable
- Moves from protected to evictable

#### 3. Free Unused Pages

**Code:** `python/minisgl/scheduler/cache.py:20-22`

```python
def _free(self, indices: torch.Tensor) -> None:
    if len(indices) > 0:
        self._free_slots = torch.cat([self._free_slots, indices])
```

Unused pages (e.g., allocated but not used) are returned to the free pool.

#### 4. Free Table Slot

**Code:** `python/minisgl/scheduler/table.py:18-19`

```python
def free(self, slot: int) -> None:
    self._free_slots.append(slot)
```

The table slot is freed for reuse.

### Key Concepts

**Protected vs Evictable**:
- **Protected**: `ref_count > 0`, cannot be evicted
- **Evictable**: `ref_count == 0`, can be evicted when cache is full

**Prefix Insertion**: Makes cache available for future requests
- Enables prefix sharing
- Improves cache hit rate

---

## Memory Layout

### KV Cache Structure

**Code:** `python/minisgl/kvcache/mha_pool.py:16-48`

```python
# Shape: (2, num_layers, num_pages, 1, local_kv_heads, head_dim)
# [0] = K cache, [1] = V cache
self._kv_buffer = kv_buffer.view(2, num_layers, num_pages, 1, local_kv_heads, head_dim)
```

```
KV Cache Memory Layout:
┌─────────────────────────────────────────────────────────┐
│ Layer 0                                                 │
│ ┌───────────────────────────────────────────────────┐  │
│ │ Page 0 │ Page 1 │ Page 2 │ ... │ Page N-1        │  │  │
│ │  K/V   │  K/V   │  K/V   │     │   K/V          │  │  │
│ └───────────────────────────────────────────────────┘  │
│ Layer 1                                                 │
│ ┌───────────────────────────────────────────────────┐  │
│ │ Page 0 │ Page 1 │ Page 2 │ ... │ Page N-1        │  │  │
│ │  K/V   │  K/V   │  K/V   │     │   K/V          │  │  │
│ └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Page Table Structure

```
Page Table: [max_running_reqs, max_seq_len]
┌──────────┬──────────────────────────────────────────────┐
│ table_idx│ page_index for each token position         │
├──────────┼──────────────────────────────────────────────┤
│    0     │ [5, 6, 7, 8, 9, 0, 0, ...]                │
│    1     │ [5, 6, 7, 10, 11, 0, 0, ...]              │
│    2     │ [12, 13, 14, 0, 0, ...]                   │
└──────────┴──────────────────────────────────────────────┘
```

### Radix Tree Structure

```
Radix Tree (simplified):
                    Root
                     │
            ┌────────┴────────┐
            │                 │
        [1,2,3]          [10,11,12]
         │                    │
    ┌────┴────┐          [13]
    │         │
  [4,5]    [6,7,8]
   │         │
  Req1     Req2
```

---

## Code References

### Core Components

#### KV Cache Storage
- **Base Interface**: `python/minisgl/kvcache/base.py:11-38` - `BaseKVCache`
- **Implementation**: `python/minisgl/kvcache/mha_pool.py:10-80` - `MHAKVCache`
- **Creation**: `python/minisgl/kvcache/__init__.py:27-44` - `create_kvcache()`

#### Cache Management
- **Cache Manager**: `python/minisgl/scheduler/cache.py:12-72` - `CacheManager`
- **Radix Manager**: `python/minisgl/kvcache/radix_manager.py:87-222` - `RadixCacheManager`
- **Radix Tree Node**: `python/minisgl/kvcache/radix_manager.py:13-80` - `RadixTreeNode`

#### Request Scheduling
- **Prefill Manager**: `python/minisgl/scheduler/prefill.py:114-154` - `PrefillManager`
- **Decode Manager**: `python/minisgl/scheduler/decode.py:10-31` - `DecodeManager`
- **Table Manager**: `python/minisgl/scheduler/table.py:4-19` - `TableManager`

#### Core Structures
- **Request**: `python/minisgl/core.py:27-66` - `Req`
- **Batch**: `python/minisgl/core.py:69-95` - `Batch`
- **Cache Handle**: `python/minisgl/kvcache/base.py:46-49` - `BaseCacheHandle`

### Key Functions

#### Cache Matching
- `python/minisgl/scheduler/cache.py:24-27` - `CacheManager.match_req()`
- `python/minisgl/kvcache/radix_manager.py:116-127` - `RadixCacheManager.match_prefix()`
- `python/minisgl/kvcache/radix_manager.py:139-164` - `RadixCacheManager._walk()`

#### Cache Allocation
- `python/minisgl/scheduler/cache.py:39-52` - `CacheManager.allocate()`
- `python/minisgl/kvcache/radix_manager.py:166-193` - `RadixCacheManager.evict()`

#### Forward Pass
- `python/minisgl/engine/engine.py:188-203` - `Engine.forward_batch()`
- `python/minisgl/attention/fa.py:49-65` - `FlashAttentionBackend.forward()`
- `python/minisgl/kvcache/mha_pool.py:56-67` - `MHAKVCache.store_kv()`
- `python/minisgl/core.py:51-53` - `Req.complete_one()`

#### Cache Cleanup
- `python/minisgl/scheduler/cache.py:54-62` - `CacheManager.free_and_cache_finished_req()`
- `python/minisgl/kvcache/radix_manager.py:129-137` - `RadixCacheManager.insert_prefix()`
- `python/minisgl/kvcache/radix_manager.py:97-114` - `RadixCacheManager.lock_handle()`

### Data Structures

#### SizeInfo
- **Definition**: `python/minisgl/kvcache/base.py:51-57`
- **Usage**: Tracks evictable and protected cache sizes

#### RadixCacheHandle
- **Definition**: `python/minisgl/kvcache/radix_manager.py:82-84`
- **Fields**: `cached_len`, `node`
- **Usage**: Reference to cached prefix in radix tree

#### Req
- **Definition**: `python/minisgl/core.py:27-66`
- **Key Fields**: `cached_len`, `device_len`, `table_idx`, `cache_handle`
- **Usage**: Represents an active request

---

## Summary

The KV cache system in mini-sglang provides:

1. **Efficient Prefix Sharing**: Radix tree enables O(n) prefix matching
2. **Memory Management**: Protected/evictable separation ensures safety
3. **Incremental Updates**: Only new tokens are processed
4. **Cache Reuse**: Completed requests make cache available for future requests

Key design decisions:
- **Radix tree** for efficient prefix matching (vs. simple comparison)
- **Protected/evictable** separation for safe eviction
- **Page-based allocation** for flexible memory management
- **Reference counting** for shared cache entries

For a runnable example, see: `learning/puzzles/04_kvcache/kvcache_example.py`
