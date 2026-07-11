"""Retrieval eval dataset: a small in-memory corpus + queries with ground truth.

CORPUS: the documents the retriever searches over; each has a stable resource_id.
CASES:  a learner-style query plus the resource_id(s) that should come back.

The production suite runs the same metrics over a live vector index; this
standalone corpus lets the harness run anywhere with just an embeddings key.
"""

CORPUS = [
    {
        "resource_id": "rag",
        "title": "Retrieval-Augmented Generation",
        "body": (
            "Retrieval-Augmented Generation (RAG) pairs a retriever with a generator. "
            "The retriever pulls relevant documents from an external corpus at query "
            "time, and the generator conditions on them to produce grounded answers. "
            "This lets a model cite sources and use up-to-date or private knowledge "
            "without retraining, which reduces hallucination."
        ),
    },
    {
        "resource_id": "attention",
        "title": "Self-Attention",
        "body": (
            "Self-attention lets every token in a sequence attend to every other token. "
            "Each token produces a query, key, and value vector; attention weights come "
            "from scaled dot products between queries and keys, and the output is a "
            "weighted sum of values. This captures long-range dependencies in parallel, "
            "unlike recurrent models that process tokens one at a time."
        ),
    },
    {
        "resource_id": "bpe",
        "title": "Byte-Pair Encoding",
        "body": (
            "Byte-pair encoding (BPE) is a subword tokenization algorithm. It starts from "
            "characters and repeatedly merges the most frequent adjacent pair into a new "
            "token. This balances vocabulary size against sequence length and lets the "
            "model handle rare or unseen words by composing them from subword pieces."
        ),
    },
    {
        "resource_id": "backprop",
        "title": "Backpropagation",
        "body": (
            "Backpropagation computes the gradient of the loss with respect to every "
            "weight by applying the chain rule backward through the network. Starting "
            "from the output error, it propagates gradients layer by layer, so an "
            "optimizer like gradient descent can update the weights to reduce the loss."
        ),
    },
    {
        "resource_id": "dropout",
        "title": "Dropout",
        "body": (
            "Dropout is a regularization technique that randomly zeroes a fraction of "
            "activations during training. This stops neurons from co-adapting and acts "
            "like training an ensemble of sub-networks, which improves generalization. "
            "At test time all neurons are active and outputs are scaled accordingly."
        ),
    },
    {
        "resource_id": "batchnorm",
        "title": "Batch Normalization",
        "body": (
            "Batch normalization normalizes the inputs to a layer using the mean and "
            "variance of the current mini-batch, then applies a learnable scale and "
            "shift. It stabilizes and speeds up training by reducing internal covariate "
            "shift and lets you use higher learning rates."
        ),
    },
]

CASES = [
    {
        "name": "rag_grounding",
        "query": "how can a language model answer using sources without being retrained?",
        "expected_resource_ids": ["rag"],
    },
    {
        "name": "attention_dependencies",
        "query": "how do transformers let every word look at every other word at once?",
        "expected_resource_ids": ["attention"],
    },
    {
        "name": "bpe_subwords",
        "query": "how do models split rare words into smaller reusable pieces?",
        "expected_resource_ids": ["bpe"],
    },
    {
        "name": "backprop_gradients",
        "query": "how are gradients computed backward through a neural network?",
        "expected_resource_ids": ["backprop"],
    },
    {
        "name": "dropout_regularization",
        "query": "what randomly turns off neurons during training to reduce overfitting?",
        "expected_resource_ids": ["dropout"],
    },
    {
        "name": "batchnorm_stability",
        "query": "how does normalizing layer inputs per mini-batch speed up training?",
        "expected_resource_ids": ["batchnorm"],
    },
]
