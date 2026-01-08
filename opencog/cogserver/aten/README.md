# ATen/PyTorch Integration for CogServer

This module provides integration between OpenCog's AtomSpace and PyTorch's
ATen tensor library, enabling neuro-symbolic AI by bridging symbolic
knowledge graphs with neural network computations.

## Overview

The ATenSpace extension provides:

- **TensorValue**: A new Value type that wraps ATen tensors for storage in AtomSpace
- **ATenSpace**: High-level API for tensor operations on the knowledge graph
- **MCP Plugin**: JSON-based tools for LLM interaction with tensors
- **Scheme Bindings**: Guile/Scheme API for tensor operations

## Building with ATen Support

### Prerequisites

Install PyTorch (CPU or CUDA version):

```bash
# CPU only
pip install torch

# With CUDA support
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

Or download LibTorch from https://pytorch.org/get-started/locally/

### CMake Configuration

```bash
mkdir build && cd build
cmake -DWITH_ATEN=ON ..
make -j
sudo make install
```

If Torch is not found automatically, specify its location:

```bash
cmake -DWITH_ATEN=ON -DTorch_DIR=/path/to/libtorch/share/cmake/Torch ..
```

## Usage

### Scheme API

```scheme
(use-modules (opencog aten))

;; Create a tensor
(define emb (cog-new-tensor '(128) #:init 'randn))

;; Store on an atom
(cog-set-tensor! (Concept "cat") (Predicate "embedding") emb)

;; Retrieve
(define cat-emb (cog-get-tensor (Concept "cat") (Predicate "embedding")))

;; Matrix operations
(define w (cog-new-tensor '(64 128)))
(define result (cog-tensor-matmul w cat-emb))

;; Similarity
(define dog-emb (cog-get-tensor (Concept "dog") (Predicate "embedding")))
(cog-tensor-cosine-similarity cat-emb dog-emb)  ; Returns float

;; Find nearest neighbors
(cog-nearest-by-embedding
  (Concept "cat")
  (Predicate "embedding")
  (list (Concept "dog") (Concept "fish") (Concept "bird"))
  3)  ; Returns top 3 similar atoms
```

### MCP Tools

When using an LLM via MCP, the following tools are available:

- `tensorCreate`: Create new tensors with shape, dtype, initialization
- `tensorGet`: Retrieve tensor from an atom
- `tensorSet`: Store tensor data on an atom
- `tensorOp`: Matrix operations (matmul, transpose, reshape)
- `tensorMath`: Element-wise operations (add, sub, mul, div)
- `tensorActivation`: Apply activation functions (relu, sigmoid, softmax)
- `tensorSimilarity`: Compute similarity between tensors
- `tensorNearest`: Find k nearest neighbors by embedding
- `tensorInfo`: Get CUDA/system information

Example MCP request:

```json
{
  "tool": "tensorCreate",
  "params": {
    "shape": [128],
    "dtype": "float32",
    "init": "randn",
    "atom": "(Concept \"word-cat\")",
    "key": "(Predicate \"embedding\")"
  }
}
```

### C++ API

```cpp
#include <opencog/cogserver/aten/ATenSpace.h>
#include <opencog/cogserver/aten/TensorValue.h>

using namespace opencog;

// Create ATenSpace attached to AtomSpace
ATenSpace aten(&atomspace);

// Create and store tensor
Handle word = atomspace.add_node(CONCEPT_NODE, "cat");
Handle key = atomspace.add_node(PREDICATE_NODE, "embedding");
aten.set_tensor(word, key, {128}, at::kFloat, "randn");

// Retrieve tensor
TensorValuePtr emb = aten.get_tensor(word, key);

// Compute similarity
TensorValuePtr other = aten.get_tensor(dog, key);
double sim = aten.cosine_similarity(emb, other);

// Neural network operations
TensorValuePtr activated = aten.activate(emb, "relu");
TensorValuePtr output = aten.linear(input, weights, bias);
```

## Architecture

```
                    ┌─────────────────────┐
                    │     AtomSpace       │
                    │  (Knowledge Graph)  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    TensorValue      │
                    │  (Value wrapper)    │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   ATenSpace     │  │   McpPlugATen   │  │   Scheme API    │
│  (C++ API)      │  │  (MCP Tools)    │  │   (Guile)       │
└────────┬────────┘  └────────┬────────┘  └────────┬────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              ▼
                    ┌─────────────────────┐
                    │    ATen/PyTorch     │
                    │  (Tensor Library)   │
                    └─────────────────────┘
```

## Key Concepts

### TensorValue

TensorValue extends AtomSpace's Value system to hold ATen tensors:

- Tensors are stored as Values attached to Atoms with keys
- Supports GPU/CPU tensors transparently
- Provides conversion to/from FloatValue for compatibility
- Thread-safe for multi-user access

### Neuro-Symbolic Integration

This module enables hybrid AI approaches:

1. **Embeddings in Knowledge Graphs**: Store neural embeddings directly on concept nodes
2. **Similarity-Based Reasoning**: Find related concepts by embedding similarity
3. **Neural Network Layers**: Apply transformations using stored weight matrices
4. **Attention Mechanisms**: Compute attention over AtomSpace structures

## Files

- `TensorValue.h/cc`: TensorValue class implementation
- `ATenSpace.h/cc`: High-level tensor operations API
- `McpPlugATen.h/cc`: MCP plugin for JSON-based tool access
- `ATenSCM.cc`: Guile/Scheme bindings
- `aten.scm`: Scheme module definition
- `CMakeLists.txt`: Build configuration

## Dependencies

- PyTorch/LibTorch (ATen): Core tensor library
- jsoncpp: JSON parsing for MCP plugin
- Guile 3.0+: Scheme bindings (optional)

## See Also

- [AtomSpace Documentation](https://wiki.opencog.org/w/AtomSpace)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MCP Protocol](https://www.anthropic.com/news/model-context-protocol)
