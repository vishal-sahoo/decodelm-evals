"""
System prompt eval dataset.

Tests the tutor agent's behavior: tool-calling decisions, response quality,
and adherence to teaching philosophy.

Each case has:
  - name: short identifier
  - description: what this tests
  - learner_profile: simulated learner context (or "")
  - current_content: simulated content metadata (or "")
  - user_message: what the user says
  - assertions: list of checks
      {"type": "calls_tool", "tool": "tool_name"}         — must call this tool
      {"type": "no_tool_calls"}                            — must NOT call any tools
      {"type": "calls_any_tool"}                           — must call at least one tool
      {"type": "does_not_call", "tool": "tool_name"}       — must NOT call this tool
      {"type": "response_quality", "criteria": "..."}      — LLM-as-judge on response text
"""

# Minimal learner profiles for testing
BEGINNER_PROFILE = (
    "<learner_profile>\n"
    "Name: Alex\n"
    "Background: Computer science student, 2nd year\n"
    "Learning style: Prefers step-by-step explanations with analogies\n"
    "Knowledge level: Beginner — knows Python basics, no ML experience\n"
    "Learnings: 2 notes (Python lists are mutable, functions can return multiple values)\n"
    "</learner_profile>"
)

ADVANCED_PROFILE = (
    "<learner_profile>\n"
    "Name: Jordan\n"
    "Background: ML engineer at a startup, 5 years experience\n"
    "Learning style: Concise, technical, skip the basics\n"
    "Knowledge level: Advanced — deep experience with PyTorch, transformers, fine-tuning\n"
    "Learnings: 47 notes spanning attention, RLHF, quantization, distributed training\n"
    "</learner_profile>"
)

INTERMEDIATE_PROFILE = (
    "<learner_profile>\n"
    "Name: Sam\n"
    "Background: Data analyst transitioning into ML, completed an online deep learning course\n"
    "Learning style: Likes building intuition before formulas\n"
    "Knowledge level: Intermediate — understands gradient descent, loss functions, basic neural nets\n"
    "Learnings: 14 notes spanning backpropagation, overfitting, regularization, basic CNNs\n"
    "</learner_profile>"
)

CONTENT_BLOCK = (
    "<current_content>\n"
    "- Understanding Transformers (310 lines) [content_id: personal:abc123]\n"
    "</current_content>"
)

CASES = [
    # ── Tool-calling decisions ────────────────────────────────────────
    {
        "name": "greeting_no_tools",
        "description": "Greeting should not trigger any tool calls",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "Hey! How's it going?",
        "assertions": [
            {"type": "no_tool_calls"},
            {
                "type": "response_quality",
                "criteria": "Response is conversational and friendly, not a lecture",
            },
        ],
    },
    {
        "name": "meta_question_no_tools",
        "description": "Questions about the app itself should not trigger tools",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "What can you help me with?",
        "assertions": [
            {"type": "no_tool_calls"},
        ],
    },
    {
        "name": "content_question_reads_content",
        "description": "Question about available content should use read_content",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": CONTENT_BLOCK,
        "user_message": "Can you explain what self-attention is based on the document?",
        "assertions": [
            {"type": "calls_tool", "tool": "read_content"},
        ],
    },
    {
        "name": "general_question_searches",
        "description": "General concept question without content should search knowledge base",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "What is gradient descent and why is it important?",
        "assertions": [
            {"type": "calls_tool", "tool": "search_all_knowledge"},
        ],
    },
    {
        "name": "quiz_request_uses_learnings",
        "description": "Quiz request should start with get_spaced_learnings and never use search_all_knowledge or the web",
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": "Quiz me on something I've been studying",
        "assertions": [
            {"type": "calls_tool", "tool": "get_spaced_learnings"},
            {"type": "does_not_call", "tool": "search_all_knowledge"},
            {"type": "does_not_call", "tool": "read_content"},
            {"type": "does_not_call", "tool": "duckduckgo_search"},
        ],
    },
    {
        "name": "quiz_concrete_wrong_answer_verdict_first",
        "description": "Grading a concrete wrong quiz answer must state it's wrong first, not affirm-then-contradict",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": (
            "We're doing a quiz. You asked me: "
            '"What does `7 % 3` evaluate to in Python?" My answer: 2.'
        ),
        "assertions": [
            {
                "type": "response_quality",
                "criteria": (
                    "The user gave a concrete WRONG answer (2) to '7 % 3' "
                    "(correct value: 1). The response MUST plainly indicate the "
                    "answer is incorrect and give the correct value 1. It MUST "
                    "NOT open with affirmation such as 'Correct', \"You're "
                    "right\", 'Yes', or 'Exactly' before correcting — no "
                    "praise-then-contradiction. Stating it's wrong first (e.g. "
                    "'Not quite') and then explaining is the correct behavior."
                ),
            },
        ],
    },
    {
        "name": "selection_reads_content",
        "description": "User selection block should trigger read_content for surrounding context",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": CONTENT_BLOCK,
        "user_message": (
            '<user_selection source="Understanding Transformers" content_id="personal:abc123">\n'
            "the attention mechanism computes weighted sums of values\n"
            "</user_selection>\n\n"
            "Why weighted sums specifically?"
        ),
        "assertions": [
            {"type": "calls_tool", "tool": "read_content"},
        ],
    },
    {
        "name": "pdf_diagram_question_reads_content",
        "description": "A visual/diagram question on an uploaded PDF should read_content (which shows the actual file), not answer from memory",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": (
            "<current_content>\n"
            "- Circuits Lecture 4 (210 lines) [content_id: personal:pdf789]\n"
            "</current_content>"
        ),
        "user_message": (
            "Walk me through the circuit diagram in this lecture PDF, "
            "specifically the supermesh — I can't follow it from the figure."
        ),
        "assertions": [
            {"type": "calls_tool", "tool": "read_content"},
        ],
    },
    # ── Response quality / teaching philosophy ────────────────────────
    {
        "name": "beginner_gets_analogy",
        "description": "Beginner learner should get accessible language, not jargon-heavy response",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "What is a neural network?",
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="Intro to Neural Networks"'
                ' content_id="shared:nn-001">\n'
                "A neural network is a computational model "
                "inspired by biological neurons. It consists "
                "of layers of interconnected nodes that learn "
                "to map inputs to outputs by adjusting weights "
                "during training via backpropagation.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "Response uses simple language appropriate "
                    "for a beginner. It includes an analogy or "
                    "intuitive explanation, not just technical "
                    "definitions. It does NOT assume knowledge "
                    "of calculus, linear algebra, or ML terminology."
                ),
            },
        ],
    },
    {
        "name": "advanced_gets_depth",
        "description": "Advanced learner should not get basic definitions they already know",
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": (
            "How does flash attention differ from standard "
            "attention in terms of memory access patterns?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="Flash Attention Deep Dive"'
                ' content_id="shared:flash-001">\n'
                "Flash Attention restructures the attention "
                "computation to minimize HBM (high bandwidth "
                "memory) reads/writes by tiling the Q, K, V "
                "matrices into blocks that fit in SRAM. Standard "
                "attention materializes the full N×N attention "
                "matrix in HBM, causing IO-bound performance. "
                "Flash Attention achieves O(N) HBM access vs "
                "O(N²) for standard attention.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "Response does NOT start with 'attention is a "
                    "mechanism that...' or other basic definitions. "
                    "It engages at an advanced level, discussing "
                    "memory hierarchy, tiling, or IO complexity. "
                    "It treats the learner as a peer who already "
                    "understands standard attention deeply."
                ),
            },
        ],
    },
    {
        "name": "misconception_corrected_gently",
        "description": "Factual error should be corrected without being dismissive",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": (
            "CNNs are mainly used for sequential data " "like text, right?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="CNN Fundamentals"'
                ' content_id="shared:cnn-001">\n'
                "Convolutional Neural Networks (CNNs) are "
                "designed for spatial data, particularly images. "
                "They use sliding filters (kernels) to detect "
                "local patterns like edges, textures, and shapes. "
                "For sequential data like text, RNNs, LSTMs, or "
                "Transformers are more commonly used.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "Response corrects the misconception: CNNs are "
                    "primarily for spatial data like images, not "
                    "sequential data. It does NOT start with "
                    "'No, that's wrong' or 'Actually, you're wrong'. "
                    "It acknowledges what the user might be thinking "
                    "of (RNNs/LSTMs for sequential data) before "
                    "correcting."
                ),
            },
        ],
    },
    {
        "name": "thanks_no_tools",
        "description": "'Thanks' should not trigger unnecessary tool calls",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "Thanks, that makes sense!",
        "assertions": [
            {"type": "no_tool_calls"},
        ],
    },
    # ── Socratic teaching ─────────────────────────────────────────────
    {
        "name": "socratic_probe_advanced_why",
        "description": "Advanced learner asks 'why' about topic they know — tutor should probe, not lecture",
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": "Why do most LLMs use decoder-only architectures instead of encoder-decoder?",
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="LLM Architectures Overview" content_id="shared:arch-001">\n'
                "Decoder-only architectures (GPT-style) have become dominant for LLMs. "
                "They simplify training by using a single forward pass with causal masking, "
                "avoiding the need for separate encoder/decoder stacks. This reduces "
                "complexity and enables more efficient scaling. Encoder-decoder models "
                "(like T5) still excel at tasks with clear input-output mappings like "
                "translation.\n</result>"
            ),
            "search_user_knowledge": (
                "<user_learnings>\n"
                "Jordan has 47 learnings including: transformers use self-attention for "
                "parallel processing; encoder-decoder models have cross-attention between "
                "stacks; causal masking prevents attending to future tokens.\n"
                "</user_learnings>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "Response asks the learner a guiding question or prompts them to reason "
                    "before (or instead of) giving the full answer. For example: 'Given your "
                    "experience with transformers, what tradeoffs do you see?' or 'What would "
                    "encoder-decoder give you that decoder-only doesn't?'. "
                    "It does NOT just lecture the full answer without any Socratic element."
                ),
            },
        ],
    },
    {
        "name": "revise_mode_interview_skips_probe",
        "description": "Advanced learner signals interview/time pressure — tutor should answer directly, not probe (revise mode)",
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": (
            "I have an interview in an hour and no time to think it through, so just "
            "explain it directly: why do most LLMs use decoder-only architectures "
            "instead of encoder-decoder?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="LLM Architectures Overview" content_id="shared:arch-001">\n'
                "Decoder-only architectures (GPT-style) have become dominant for LLMs. "
                "They simplify training by using a single forward pass with causal masking, "
                "avoiding the need for separate encoder/decoder stacks. This reduces "
                "complexity and enables more efficient scaling. Encoder-decoder models "
                "(like T5) still excel at tasks with clear input-output mappings like "
                "translation.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "The learner signaled time pressure (interview in an hour) and explicitly "
                    "asked for a direct explanation. PASS if the response leads with a direct "
                    "explanation of why decoder-only architectures dominate. FAIL if it opens "
                    "with a Socratic guiding question or asks the learner to reason it out first "
                    "before explaining, since the urgency signal should override the usual "
                    "probe-the-advanced-learner behavior."
                ),
            },
        ],
    },
    {
        "name": "socratic_no_probe_beginner",
        "description": "Beginner asks conceptual question — should get scaffolding, not a probe",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "Why do we need to normalize data before training a model?",
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="Data Preprocessing Basics" content_id="shared:prep-001">\n'
                "Normalization scales features to a similar range (e.g., 0 to 1 or "
                "mean 0, std 1). Without it, features with larger magnitudes dominate "
                "the learning process, causing slow or unstable training. Common methods "
                "include min-max scaling and z-score standardization.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "Response explains the concept directly with scaffolding appropriate for a beginner. "
                    "It does NOT ask the learner 'what do you think?' or 'what's your intuition?' as the primary response — "
                    "a beginner with 2 learnings about Python basics has no foundation to reason about normalization. "
                    "It SHOULD use an analogy or simple example to build understanding."
                ),
            },
        ],
    },
    {
        "name": "socratic_probe_claim",
        "description": "Advanced learner makes a claim — tutor should ask them to predict consequences",
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": "I think you could just use a larger context window instead of bothering with RAG.",
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="RAG vs Long Context" content_id="shared:rag-001">\n'
                "Long context windows (100k+ tokens) can reduce the need for RAG in "
                "some cases, but come with quadratic attention cost, higher latency, "
                "and the 'lost in the middle' problem where models struggle with "
                "information in the middle of long contexts. RAG remains more cost-effective "
                "for large, frequently updated knowledge bases.\n</result>"
            ),
            "search_user_knowledge": (
                "<user_learnings>\n"
                "Jordan has learnings on RAG retrieval quality being a bottleneck, "
                "and on how attention scales quadratically with sequence length.\n"
                "</user_learnings>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "Response challenges the learner to think through consequences or edge cases "
                    "of their claim rather than immediately agreeing or disagreeing. For example: "
                    "'What happens to cost/latency as context grows?' or 'How would the model "
                    "handle information that changes daily?'. The response treats the claim as a "
                    "starting point for exploration, not something to just confirm or refute."
                ),
            },
        ],
    },
    {
        "name": "socratic_no_probe_factual_lookup",
        "description": "Factual lookup question should be answered directly, not turned into a probe",
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": "What does LoRA stand for?",
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="Parameter-Efficient Fine-Tuning" content_id="shared:peft-001">\n'
                "LoRA (Low-Rank Adaptation) freezes the pretrained model weights and injects "
                "trainable low-rank decomposition matrices into each transformer layer, "
                "drastically reducing the number of trainable parameters.\n</result>"
            ),
        },
        "assertions": [
            {
                "type": "response_quality",
                "criteria": (
                    "Response directly answers that LoRA stands for Low-Rank Adaptation. "
                    "It does NOT ask the learner to guess or reason about the "
                    "acronym. A factual lookup should be answered immediately."
                ),
            },
        ],
    },
    {
        "name": "socratic_no_probe_content_clarification",
        "description": "Clarification about study content should explain directly, not probe",
        "learner_profile": ADVANCED_PROFILE,
        "current_content": CONTENT_BLOCK,
        "user_message": "What does 'causal masking' mean in this context?",
        "mock_tool_results": {
            "read_content": (
                "Line 42: In the decoder, causal masking ensures "
                "that position i can only attend to positions "
                "<= i. This prevents information leakage from "
                "future tokens during training.\n"
                "Line 43: The mask is typically implemented as an "
                "upper-triangular matrix of -inf values added to "
                "the attention scores before softmax."
            ),
        },
        "assertions": [
            {"type": "calls_tool", "tool": "read_content"},
            {
                "type": "response_quality",
                "criteria": (
                    "Response explains what causal masking means, "
                    "grounded in the content. It does NOT ask "
                    "'what do you think causal masking means?' — "
                    "the learner asked for clarification on specific "
                    "text, which should be answered directly."
                ),
            },
        ],
    },
    # ── Knowledge sourcing ────────────────────────────────────────────
    {
        "name": "substantive_question_must_search",
        "description": "Substantive conceptual question must trigger a search, not rely on model memory",
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "What is overfitting and how do you prevent it?",
        "assertions": [
            {"type": "calls_tool", "tool": "search_all_knowledge"},
        ],
    },
    # ── Retrieval activation ───────────────────────────────────────────
    {
        "name": "retrieval_activation_intermediate",
        "description": (
            "Intermediate learner with related knowledge "
            "should be prompted to recall before being taught"
        ),
        "learner_profile": INTERMEDIATE_PROFILE,
        "current_content": "",
        "user_message": "What is dropout and why does it help?",
        "mock_tool_results": {
            "search_all_knowledge": (
                "<user_learnings>\n"
                "<result>Sam understands that overfitting "
                "happens when a model memorizes training data "
                "instead of learning general patterns</result>\n"
                "<result>Sam knows that regularization "
                "techniques like L2 penalty constrain weights "
                "to prevent overfitting</result>\n"
                "</user_learnings>\n\n"
                "<other_articles>\n"
                '<result title="Regularization Techniques"'
                ' content_id="shared:reg-001">\n'
                "Dropout randomly zeroes activations during "
                "training with probability p, forcing the "
                "network to learn redundant representations. "
                "It acts as an implicit ensemble of sub-networks "
                "and reduces co-adaptation of neurons. Typically "
                "p=0.5 for hidden layers and p=0.2 for input "
                "layers.\n</result>\n"
                "</other_articles>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "The learner already knows about overfitting "
                    "and L2 regularization. The response should "
                    "prompt them to connect their existing knowledge "
                    "before explaining dropout — e.g., asking how "
                    "they think dropout might relate to the "
                    "regularization they already know, or what "
                    "problem it's solving given what they know "
                    "about overfitting. It should NOT launch "
                    "straight into a full explanation of dropout "
                    "without engaging their prior knowledge first."
                ),
            },
        ],
    },
    {
        "name": "no_retrieval_activation_beginner",
        "description": (
            "Beginner with no related knowledge should get "
            "direct teaching, not recall prompt"
        ),
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "What is regularization?",
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="Regularization Basics"'
                ' content_id="shared:reg-002">\n'
                "Regularization is a set of techniques that "
                "prevent overfitting by constraining the model. "
                "Common methods include L1 (lasso), L2 (ridge), "
                "dropout, and early stopping. The goal is to "
                "improve generalization to unseen data."
                "\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "The learner is a beginner with only Python "
                    "basics. The response should explain "
                    "regularization directly — it should NOT ask "
                    "'what do you already know about "
                    "regularization?' or 'how do you think this "
                    "relates to what you've learned?' or otherwise "
                    "prompt the learner to recall prior knowledge "
                    "before explaining. Beginners have no "
                    "foundation to recall from."
                ),
            },
        ],
    },
    # ── One step at a time ─────────────────────────────────────────────
    {
        "name": "one_step_at_a_time",
        "description": (
            "Multi-step explanation should pause after "
            "first concept, not dump everything"
        ),
        "learner_profile": INTERMEDIATE_PROFILE,
        "current_content": "",
        "user_message": (
            "How do transformers process sequences in parallel "
            "instead of one token at a time like RNNs?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="Transformer Architecture"'
                ' content_id="shared:tx-001">\n'
                "Unlike RNNs which process tokens sequentially, "
                "transformers use self-attention to process all "
                "tokens simultaneously. Self-attention computes "
                "pairwise relationships between all positions. "
                "To preserve order information lost by parallel "
                "processing, positional encodings are added to "
                "input embeddings. The original paper used "
                "sinusoidal encodings, but learned encodings and "
                "rotary position embeddings (RoPE) are now "
                "common. The full architecture also includes "
                "multi-head attention, layer normalization, and "
                "feed-forward layers.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "This question spans multiple concepts: "
                    "self-attention for parallelism, positional "
                    "encodings for order, multi-head attention, "
                    "etc. The response should focus on ONE main "
                    "concept (likely: self-attention enables "
                    "parallel processing by computing all pairwise "
                    "relationships at once) and then pause or "
                    "invite the learner to continue to the next "
                    "concept. It should NOT explain self-attention "
                    "AND positional encodings AND multi-head "
                    "attention AND feed-forward layers all in one "
                    "response. A bridge question like 'Want me to "
                    "explain how it handles word order?' or 'Make "
                    "sense so far?' signals appropriate pacing."
                ),
            },
        ],
    },
    {
        "name": "simple_question_full_answer",
        "description": (
            "A simple one-step question should be answered "
            "fully, not artificially broken up"
        ),
        "learner_profile": INTERMEDIATE_PROFILE,
        "current_content": "",
        "user_message": (
            "What's the difference between a parameter " "and a hyperparameter?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="ML Fundamentals"'
                ' content_id="shared:ml-001">\n'
                "Parameters are values learned during training "
                "(weights, biases). Hyperparameters are set "
                "before training and control the learning "
                "process (learning rate, batch size, number of "
                "layers). Parameters are optimized by the "
                "algorithm; hyperparameters are chosen by the "
                "practitioner.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "This is a straightforward one-step "
                    "distinction. The response should answer it "
                    "fully and clearly — explaining both "
                    "parameters and hyperparameters and the "
                    "difference. It should NOT artificially break "
                    "this into multiple turns (e.g., 'Let me first "
                    "explain parameters... want me to explain "
                    "hyperparameters next?'). Simple questions "
                    "deserve complete answers."
                ),
            },
        ],
    },
    # ── Productive failure ─────────────────────────────────────────────
    {
        "name": "productive_failure_wrong_approach",
        "description": (
            "When learner proposes a wrong reasoning path, "
            "tutor should explore it with them, not correct "
            "immediately"
        ),
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": (
            "I think we could just increase the model size to "
            "fix hallucinations — more parameters means more "
            "knowledge, right?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="LLM Hallucination"'
                ' content_id="shared:hall-001">\n'
                "Hallucination in LLMs is not primarily a "
                "knowledge capacity issue. Larger models can "
                "still hallucinate because the issue stems from "
                "the training objective (next-token prediction "
                "rewards fluency over factuality), lack of "
                "grounding mechanisms, and distribution shift at "
                "inference time. Approaches like RLHF, retrieval "
                "augmentation, and constrained decoding address "
                "hallucination more directly than scaling alone."
                "\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "The learner proposes a wrong reasoning path "
                    "(scaling fixes hallucination). The response "
                    "should guide them to explore the consequences "
                    "of their reasoning rather than immediately "
                    "correcting them. For example: 'If that were "
                    "true, what would you expect to see as models "
                    "got bigger?' or 'Let's test that — do the "
                    "largest models hallucinate less?' The goal is "
                    "for the learner to discover why scaling alone "
                    "doesn't solve hallucination. The response "
                    "should NOT immediately say 'That's not how it "
                    "works, hallucination is caused by...' or "
                    "lecture the correct answer without letting the "
                    "learner reason through it first."
                ),
            },
        ],
    },
    {
        "name": "factual_error_correct_directly",
        "description": (
            "Factual errors should be corrected directly, "
            "not explored as reasoning paths"
        ),
        "learner_profile": INTERMEDIATE_PROFILE,
        "current_content": "",
        "user_message": (
            "So batch normalization was introduced in the "
            "original transformer paper, right?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": (
                '<result title="Normalization in Deep Learning"'
                ' content_id="shared:norm-001">\n'
                "Batch normalization was introduced by Ioffe & "
                "Szegedy (2015) for CNNs. The original "
                "transformer paper (Vaswani et al., 2017) used "
                "layer normalization, not batch normalization, "
                "because layer norm works better with "
                "variable-length sequences and doesn't depend "
                "on batch statistics.\n</result>"
            ),
        },
        "assertions": [
            {"type": "calls_any_tool"},
            {
                "type": "response_quality",
                "criteria": (
                    "This is a factual error. The response MUST "
                    "directly state that the transformer paper "
                    "used layer normalization, not batch "
                    "normalization. It should NOT guide the "
                    "learner to explore or reason through the "
                    "claim (e.g., 'Let's think about what would "
                    "happen if...'). A direct factual correction "
                    "is the right approach here — not productive "
                    "failure or Socratic questioning."
                ),
            },
        ],
    },
    # ── New-tool coverage: spaced review, content listing, web fallback ──
    {
        "name": "spaced_review_uses_spaced_tool",
        "description": (
            "A request to review what's due should call "
            "get_spaced_learnings, not general search"
        ),
        "learner_profile": INTERMEDIATE_PROFILE,
        "current_content": "",
        "user_message": (
            "What should I review today? Anything due for spaced repetition?"
        ),
        "assertions": [
            {"type": "calls_tool", "tool": "get_spaced_learnings"},
            {"type": "does_not_call", "tool": "search_all_knowledge"},
        ],
    },
    {
        "name": "list_uploads_uses_list_tool",
        "description": (
            "Asking what files the user has uploaded should "
            "call list_user_contents"
        ),
        "learner_profile": BEGINNER_PROFILE,
        "current_content": "",
        "user_message": "What documents have I uploaded so far?",
        "assertions": [
            {"type": "calls_tool", "tool": "list_user_contents"},
        ],
    },
    {
        "name": "web_search_last_resort_after_empty_search",
        "description": (
            "When knowledge search returns nothing, the tutor may fall "
            "back to the web (duckduckgo_search) as a last resort"
        ),
        "learner_profile": ADVANCED_PROFILE,
        "current_content": "",
        "user_message": (
            "What did the DecodeLM v2.3 release notes change about the "
            "merge pipeline?"
        ),
        "mock_tool_results": {
            "search_all_knowledge": "No results found.",
        },
        "assertions": [
            {"type": "calls_tool", "tool": "search_all_knowledge"},
            {"type": "calls_tool", "tool": "duckduckgo_search"},
        ],
    },
    {
        "name": "quiz_question_emits_item_id_tag",
        "description": (
            "A quiz question must end with the required <quiz_item_id> tag "
            "carrying the card id from get_spaced_learnings"
        ),
        "learner_profile": INTERMEDIATE_PROFILE,
        "current_content": "",
        "user_message": "Quiz me.",
        "mock_tool_results": {
            "get_spaced_learnings": (
                "<spaced_learnings>\n"
                "- [id:card_backprop_01] (learning) Backpropagation applies the "
                "chain rule backward through the network to compute gradients.\n"
                "- [id:card_dropout_02] (learning) Dropout randomly zeroes "
                "activations during training to reduce overfitting.\n"
                "</spaced_learnings>"
            ),
            "search_user_knowledge": "No additional user knowledge found.",
        },
        "assertions": [
            {"type": "calls_tool", "tool": "get_spaced_learnings"},
            {"type": "preserves_voice", "phrase": "<quiz_item_id>"},
        ],
    },
]
