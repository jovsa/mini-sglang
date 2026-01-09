#!/usr/bin/env python3
"""Learning Prototype 2: KV Cache and Radix Tree - Uses REAL Mini-SGLang code."""

import torch
import time
from minisgl.kvcache.radix_manager import RadixCacheManager


def visualize_tree(manager, max_depth=10):
    def _vis(node, prefix, is_last, depth):
        if depth > max_depth:
            return
        conn = "└── " if is_last else "├── "
        if node.is_root():
            print(f"ROOT (ref={node.ref_count})")
        else:
            lock = "🔒" if node.ref_count > 0 else "📦"
            print(f"{prefix}{conn}{lock} key={node._key.tolist()} -> pages={node._value.tolist()} (ref={node.ref_count})")
        children = list(node.children.values())
        for i, child in enumerate(children):
            new_prefix = prefix + ("    " if is_last else "│   ")
            _vis(child, new_prefix, i == len(children) - 1, depth + 1)
    _vis(manager.root_node, "", True, 0)


def demo_basic():
    print("=" * 60)
    print("1. Basic RadixCacheManager Operations")
    print("=" * 60)

    device = torch.device("cuda:0")
    manager = RadixCacheManager(device)

    print("\n--- First request ---")
    input1 = torch.tensor([10, 20, 30, 40, 50, 100], dtype=torch.int32)
    pages1 = torch.arange(6, dtype=torch.int32, device=device)
    h1, idx1 = manager.match_prefix(input1)
    print(f"match_prefix: cached_len={h1.cached_len}")
    manager.insert_prefix(input1, pages1)
    visualize_tree(manager)

    print("\n--- Second request (shared prefix) ---")
    input2 = torch.tensor([10, 20, 30, 40, 50, 200], dtype=torch.int32)
    h2, idx2 = manager.match_prefix(input2)
    print(f"match_prefix: cached_len={h2.cached_len}, reusing {idx2.tolist()}")
    manager.lock_handle(h2)
    manager.insert_prefix(input2, torch.cat([idx2, torch.tensor([6], dtype=torch.int32, device=device)]))
    print("\nAfter insert (split occurred):")
    visualize_tree(manager)
    manager.lock_handle(h2, unlock=True)


def demo_split():
    print("\n" + "=" * 60)
    print("2. Node Splitting Demo")
    print("=" * 60)

    device = torch.device("cuda:0")
    manager = RadixCacheManager(device)

    print("\n--- Insert [1,2,3,4,5] ---")
    manager.insert_prefix(
        torch.tensor([1, 2, 3, 4, 5], dtype=torch.int32),
        torch.tensor([0, 1, 2, 3, 4], dtype=torch.int32, device=device)
    )
    visualize_tree(manager)

    print("\n--- Insert [1,2,3,6,7] (split at 3) ---")
    h, idx = manager.match_prefix(torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32))
    print(f"Match: {h.cached_len} tokens")
    manager.insert_prefix(
        torch.tensor([1, 2, 3, 6, 7], dtype=torch.int32),
        torch.cat([idx, torch.tensor([5, 6], dtype=torch.int32, device=device)])
    )
    print("\nAfter split:")
    visualize_tree(manager)


def demo_multi_turn():
    print("\n" + "=" * 60)
    print("3. Multi-Turn Chat Efficiency")
    print("=" * 60)

    device = torch.device("cuda:0")
    manager = RadixCacheManager(device)

    system = list(range(100, 150))
    user1 = list(range(200, 220))
    asst1 = list(range(400, 500))
    user2 = list(range(300, 330))

    t1 = torch.tensor(system + user1, dtype=torch.int32)
    manager.insert_prefix(t1, torch.arange(len(t1), dtype=torch.int32, device=device))

    t1_full = torch.tensor(system + user1 + asst1, dtype=torch.int32)
    manager.insert_prefix(t1_full, torch.arange(len(t1_full), dtype=torch.int32, device=device))

    t2 = torch.tensor(system + user1 + asst1 + user2, dtype=torch.int32)
    h2, _ = manager.match_prefix(t2)

    print(f"Turn 2 total: {len(t2)} tokens")
    print(f"Cache hit: {h2.cached_len} tokens!")
    print(f"Efficiency: {h2.cached_len / len(t2) * 100:.1f}% reused!")


if __name__ == "__main__":
    if not torch.cuda.is_available():
        print("ERROR: Requires CUDA GPU")
        exit(1)

    print("Mini-SGLang RadixCache Demo (REAL code)")
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")

    demo_basic()
    demo_split()
    demo_multi_turn()

    print("\nKey: RadixTree enables efficient prefix sharing!")
