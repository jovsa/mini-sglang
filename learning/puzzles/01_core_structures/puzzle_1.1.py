"""
PUZZLE 1.1: Complete the SamplingParams - [Easy]

TASK:
Fill in the missing properties to create a SamplingParams object.
You need to implement the create_sampling_params function and the is_greedy property.

GIVEN:
- The SamplingParams dataclass structure from core.py
- Partial implementation below

================================================================================
DATA FLOW: How SamplingParams Moves Through the System
================================================================================

SamplingParams flows through the entire system, from user request to token generation:

```mermaid
flowchart LR
    A[HTTP Request] -->|"temperature, top_k, top_p"| B[API Server]
    B -->|"Create SamplingParams"| C[TokenizeMsg]
    C -->|"ZMQ message"| D[Tokenizer Worker]
    D -->|"Forward params"| E[UserMsg]
    E -->|"ZMQ message"| F[Scheduler]
    F -->|"Store in Req"| G[Req.sampling_params]
    G -->|"Batch.reqs"| H[Engine.forward]
    H -->|"Access params"| I[Sampler.sample]
    I -->|"Use is_greedy"| J[Token Generation]
```

Step-by-step flow:
1. **API Server** (`python/minisgl/server/api_server.py:260-266`): Receives HTTP request with 
   sampling parameters (temperature, top_k, top_p, max_tokens) and creates SamplingParams
2. **TokenizeMsg** (`python/minisgl/message/tokenizer.py:34-38`): SamplingParams embedded in message
3. **UserMsg** (`python/minisgl/message/backend.py:33-36`): SamplingParams forwarded from TokenizeMsg
4. **Req** (`python/minisgl/core.py:34`): SamplingParams stored as part of request state
5. **Engine.sample()** (`python/minisgl/engine/sample.py:53-55`): Accesses `req.sampling_params.is_greedy`
   to optimize sampling path (greedy vs non-greedy)

================================================================================
COMPONENT CONNECTIONS: How SamplingParams Connects Components
================================================================================

```mermaid
graph TB
    subgraph "API Layer"
        API[API Server]
    end
    
    subgraph "Message Layer"
        TMsg[TokenizeMsg]
        UMsg[UserMsg]
    end
    
    subgraph "Scheduler Layer"
        Req[Req]
        Batch[Batch]
    end
    
    subgraph "Engine Layer"
        Engine[Engine]
        Sampler[Sampler]
    end
    
    API -->|"Creates"| TMsg
    TMsg -->|"Contains"| SP1[SamplingParams]
    TMsg -->|"ZMQ"| UMsg
    UMsg -->|"Contains"| SP2[SamplingParams]
    UMsg -->|"Creates"| Req
    Req -->|"Stores"| SP3[SamplingParams]
    Req -->|"In"| Batch
    Batch -->|"Passes to"| Engine
    Engine -->|"Reads"| SP3
    Engine -->|"Uses"| Sampler
    Sampler -->|"Checks is_greedy"| SP3
```

Key Connections:
- **Message Boundaries**: SamplingParams crosses process boundaries via ZMQ serialization
- **Request Lifecycle**: Once in Req, SamplingParams stays with the request until completion
- **Sampling Optimization**: `is_greedy` property enables fast path in `Sampler.sample()` 
  (see `python/minisgl/engine/sample.py:73-74`)

================================================================================
TECHNICAL DECISIONS: Why is_greedy is a Property
================================================================================

**Decision**: `is_greedy` is a computed property, not stored state.

**Why?**
1. **Single Source of Truth**: The greedy condition is derived from temperature, top_k, and top_p.
   Storing it separately risks inconsistency if parameters change (though they shouldn't in practice).

2. **Performance**: The computation is trivial (simple boolean logic), so caching provides minimal benefit.
   The property is only accessed once per batch during `Sampler.prepare()`.

3. **Code Clarity**: Making it a property makes the relationship explicit: "greedy is determined by these parameters."

**Alternative Considered**: Store `_is_greedy` in `__post_init__` (as done here for simplicity).
   - **Trade-off**: Slightly faster access, but requires maintaining consistency
   - **Chosen**: Property approach for clarity and correctness

**Real Impact**: In `Sampler.sample()` (`python/minisgl/engine/sample.py:73-74`), when `is_greedy=True`,
   the code uses `torch.argmax()` instead of the full sampling pipeline, providing significant
   performance improvement for deterministic generation.

**Why top_p must equal 1.0 for greedy?**
- Nucleus sampling (top_p < 1.0) introduces randomness by sampling from a subset of tokens
- Even with temperature=0.0, if top_p < 1.0, we're still sampling from a filtered distribution
- True greedy sampling requires taking the single most likely token (top_k=1) from the full distribution (top_p=1.0)

================================================================================
CHALLENGE
================================================================================

- Understand what each parameter means
- Implement the is_greedy property correctly
- Handle edge cases

HINT:
- Reference: `python/minisgl/core.py:22-24`
- is_greedy should return True when (temperature <= 0.0 OR top_k == 1) AND top_p == 1.0
- Think about why top_p must equal 1.0 for greedy sampling

QUESTIONS:
1. What happens if temperature=0.0 and top_p=0.5? Is it greedy?
2. What's the difference between top_k=-1 and top_k=1?
3. When would ignore_eos be useful?
4. **NEW**: Trace through the code: How does `is_greedy` affect the execution path in `Sampler.sample()`?
   (Hint: Read `python/minisgl/engine/sample.py:53-55` and `71-74`)
5. **NEW**: Why does SamplingParams need to be serializable? What happens if we add a non-serializable field?

TEST:
Run: pytest learning/puzzles/01_core_structures/test_1.1.py -v
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SamplingParams:
    """Sampling parameters for text generation."""
    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    ignore_eos: bool = False
    max_tokens: int = 1024
    _is_greedy: bool = False

    def __post_init__(self):
        self._is_greedy = self.temperature <= 0.0 or (self.top_k == 1 and self.top_p == 1.0)

    @property
    def is_greedy(self) -> bool:
        """
        Returns True if the sampling parameters result in greedy (deterministic) sampling.

        Greedy sampling occurs when:
        - temperature <= 0.0 (always greedy, regardless of top_p)
        - OR (top_k == 1 AND top_p == 1.0)

        Think: Why must top_p == 1.0? Because top_p < 1.0 means nucleus sampling,
        which introduces randomness even with temperature=0.
        """
        return self._is_greedy


def create_sampling_params(
    temperature: float = 0.0,
    top_k: int = -1,
    top_p: float = 1.0,
    ignore_eos: bool = False,
    max_tokens: int = 1024,
) -> SamplingParams:
    """
    Create a SamplingParams object with the given parameters.

    Args:
        temperature: Sampling temperature. Lower = more deterministic.
        top_k: Sample from top k tokens. -1 means no limit.
        top_p: Nucleus sampling. Sample from tokens with cumulative prob >= top_p.
        ignore_eos: Whether to ignore end-of-sequence token.
        max_tokens: Maximum tokens to generate.

    Returns:
        SamplingParams object
    """
    res = SamplingParams(temperature=temperature, top_k=top_k, top_p=top_p, ignore_eos=ignore_eos, max_tokens=max_tokens)

    return res



# Test your implementation
if __name__ == "__main__":
    # Test 1: Greedy sampling
    params1 = create_sampling_params(temperature=0.0, top_k=1, top_p=1.0)
    print(f"Test 1 - Greedy: is_greedy={params1.is_greedy}, expected=True")

    # Test 2: Non-greedy sampling
    params2 = create_sampling_params(temperature=0.7, top_k=10, top_p=0.9)
    print(f"Test 2 - Sampling: is_greedy={params2.is_greedy}, expected=False")

    # Test 3: Edge case - temperature 0 but top_p < 1
    params3 = create_sampling_params(temperature=0.0, top_p=0.5)
    print(f"Test 3 - Edge case: is_greedy={params3.is_greedy}")
