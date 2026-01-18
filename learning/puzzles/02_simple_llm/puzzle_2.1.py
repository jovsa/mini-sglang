"""
PUZZLE 2.1: Predict Sampling Behavior - [Easy]

TASK:
Predict and understand how different sampling parameters affect output.

GIVEN:
- Same prompt with different sampling parameters
- Understanding of temperature, top_k, top_p

================================================================================
DATA FLOW: From Engine Forward to Sampling to Next Token
================================================================================

Sampling happens after the model generates logits. Here's the complete flow:

```mermaid
flowchart TD
    A[Engine.forward_batch] -->|"Model forward pass"| B[Logits Tensor]
    B -->|"Shape: [batch_size, vocab_size]"| C[Sampler.prepare]
    C -->|"Check is_greedy"| D{All greedy?}
    D -->|"Yes"| E[BatchSamplingArgs: temperatures=None]
    D -->|"No"| F[BatchSamplingArgs: temperatures, top_k, top_p]
    E -->|"Sampler.sample"| G[torch.argmax: Greedy]
    F -->|"Sampler.sample"| H[sample_impl: Non-greedy]
    G -->|"next_tokens"| I[Return to Scheduler]
    H -->|"next_tokens"| I
```

Step-by-step flow:
1. **Engine.forward_batch()** (`python/minisgl/engine/engine.py`):
   - Forward pass through model generates logits
   - Logits shape: `[batch_size, vocab_size]` (one probability distribution per request)

2. **Sampler.prepare()** (`python/minisgl/engine/sample.py:53-68`):
   - Collects `sampling_params` from all Reqs in batch
   - Checks if all are greedy: `all(p.is_greedy for p in params)`
   - If greedy: Returns `BatchSamplingArgs(temperatures=None)` (fast path)
   - If not: Prepares tensors for temperature, top_k, top_p

3. **Sampler.sample()** (`python/minisgl/engine/sample.py:71-75`):
   - **Greedy path**: `torch.argmax(logits, dim=-1)` - just take most likely token
   - **Non-greedy path**: `sample_impl()` - applies temperature, top_k, top_p, then samples

4. **Return to Scheduler**: `next_tokens` tensor (one token per request)

**Key Optimization**: When `is_greedy=True`, the entire sampling pipeline is skipped in favor
   of a simple `argmax`, providing significant performance improvement.

================================================================================
COMPONENT CONNECTIONS: How Sampling Connects to Request Parameters
================================================================================

```mermaid
graph TB
    subgraph "Request Layer"
        Req[Req]
        SP[Req.sampling_params]
    end
    
    subgraph "Batch Layer"
        Batch[Batch]
        Reqs[Batch.reqs]
    end
    
    subgraph "Engine Layer"
        Engine[Engine]
        Model[Model]
        Sampler[Sampler]
    end
    
    subgraph "Sampling"
        Logits[Logits Tensor]
        Prepare[Sampler.prepare]
        Sample[Sampler.sample]
        NextTokens[next_tokens]
    end
    
    Req -->|"Contains"| SP
    Req -->|"In"| Reqs
    Reqs -->|"Part of"| Batch
    Batch -->|"Passed to"| Engine
    Engine -->|"Calls"| Model
    Model -->|"Generates"| Logits
    Batch -->|"Extract params"| Prepare
    Prepare -->|"Reads"| SP
    Prepare -->|"Creates"| BatchSamplingArgs
    Logits -->|"Input"| Sample
    BatchSamplingArgs -->|"Input"| Sample
    Sample -->|"Generates"| NextTokens
```

Key Connections:
- **Req.sampling_params**: Source of truth for sampling behavior
- **Sampler.prepare()**: Extracts params from batch and prepares GPU tensors
- **is_greedy optimization**: Fast path when all requests are greedy

================================================================================
TECHNICAL DECISIONS: Why Greedy Sampling Optimization
================================================================================

**Decision**: Special fast path for greedy sampling (`is_greedy=True`).

**Why?**
1. **Performance**: Greedy sampling is just `argmax(logits)` - no need for:
   - Softmax computation
   - Temperature scaling
   - Top-k/top-p filtering
   - Random sampling
   This is orders of magnitude faster!

2. **Deterministic Behavior**: Greedy sampling always produces the same output for the same input.
   This is important for reproducibility and debugging.

3. **Common Use Case**: Many applications use greedy sampling (temperature=0.0), so optimizing this
   path improves overall system performance.

**Implementation** (`python/minisgl/engine/sample.py:53-56, 73-74`):
```python
# In prepare(): Check if all greedy
if all(p.is_greedy for p in params):
    return BatchSamplingArgs(temperatures=None)  # Fast path marker

# In sample(): Fast path
if args.temperatures is None:  # greedy sampling
    return torch.argmax(logits, dim=-1)  # No softmax, no sampling!
```

**Alternative Considered**: Always use full sampling pipeline.
   - **Problem**: Wastes computation for greedy case (softmax + sampling when argmax suffices)
   - **Problem**: Slower performance for common use case
   - **Chosen**: Fast path optimization for better performance

**Real Impact**: This optimization can provide 2-3x speedup for greedy generation, which is
   commonly used in production systems.

**Parameter Interactions**:
- **Temperature**: Scales logits before softmax. Lower = more deterministic.
- **Top-k**: Limits sampling to top k tokens by probability. Smaller k = less diverse.
- **Top-p (nucleus)**: Samples from tokens with cumulative probability >= p. Lower p = less diverse.
- **Combined**: All parameters work together to control the sampling distribution.

================================================================================
CHALLENGE
================================================================================

- Predict which configuration will be most deterministic
- Predict which will be most diverse
- Understand the interaction between parameters

HINT:
- temperature=0.0 → deterministic (greedy)
- temperature>0.0 → more random
- top_k limits vocabulary size
- top_p uses cumulative probability

QUESTIONS:
1. Which will be most deterministic: temp=0.0 or temp=0.7?
2. What's the difference between top_k=1 and top_k=10?
3. How does top_p interact with temperature?
4. **NEW**: Trace through the code: What happens in `Sampler.sample()` when `temperatures=None`?
   (Hint: Read `python/minisgl/engine/sample.py:73-74`)
5. **NEW**: Why does `Sampler.prepare()` check if ALL requests are greedy? What if some are greedy?
   (Hint: The batch must use the same sampling path for all requests)

TEST:
Run: pytest learning/puzzles/02_simple_llm/test_2.1.py -v
"""

import sys
import importlib.util
from pathlib import Path

# Import puzzle_1.1 using importlib (module name has dot)
puzzle_dir = Path(__file__).parent.parent / "01_core_structures"
puzzle_1_1_file = puzzle_dir / "puzzle_1.1.py"
spec = importlib.util.spec_from_file_location("puzzle_1_1", puzzle_1_1_file)
puzzle_1_1 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(puzzle_1_1)
create_sampling_params = puzzle_1_1.create_sampling_params
SamplingParams = puzzle_1_1.SamplingParams


def predict_sampling_behavior(params: SamplingParams) -> dict:
    """
    Predict the behavior of sampling parameters.

    Returns a dictionary with predictions:
    - "deterministic": bool - Will output be deterministic?
    - "diversity": str - "low", "medium", "high"
    - "vocab_size": str - "full", "limited", "very_limited"

    TODO: Implement this function to analyze sampling parameters
    """
    result = {
        "deterministic": False,  # TODO: Determine if params.is_greedy
        "diversity": "medium",    # TODO: Predict based on temperature
        "vocab_size": "full",    # TODO: Predict based on top_k and top_p
    }

    result["deterministic"] = params.is_greedy
    result["diversity"] = "low" if params.is_greedy or params.temperature <= 0.3 else "high" if params.temperature >= 0.8 else "medium"
    result["vocab_size"] = "full" if (params.top_k == -1 and params.top_p == 1.0) else "very_limited" if params.top_k <= 3 or params.top_p < 0.5 else "limited"

    return result


def compare_sampling_configs() -> dict:
    """
    Compare different sampling configurations.

    Returns a dictionary mapping config name to behavior prediction.
    """
    configs = {
        "greedy": create_sampling_params(temperature=0.0, top_k=1, top_p=1.0),
        "creative": create_sampling_params(temperature=1.0, top_k=50, top_p=0.9),
        "balanced": create_sampling_params(temperature=0.7, top_k=10, top_p=0.95),
        "focused": create_sampling_params(temperature=0.3, top_k=5, top_p=0.8),
    }

    results = {}
    for name, params in configs.items():
        results[name] = predict_sampling_behavior(params)

    return results


# Test your predictions
if __name__ == "__main__":
    results = compare_sampling_configs()
    for name, behavior in results.items():
        print(f"{name}: {behavior}")
