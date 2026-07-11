"""
Extraction eval dataset.

IMPORTANT: Eval cases must NOT reuse conversations or topics from the prompt's
examples. We are testing generalization, not memorization.

Each case has:
  - name: short identifier
  - conversation: XML-tagged conversation (same format the prompt expects)
  - expected: the correct extraction output (list of strings)
  - assertions: list of checks to run on the output
      {"type": "exact_empty"}                          — output must be []
      {"type": "exact_count"}                          — output must have exactly len(expected) items
      {"type": "semantic"}                             — each expected item has a semantic match in output
      {"type": "preserves_voice", "phrase": "..."}     — user's exact phrasing is preserved verbatim
      {"type": "count_range", "min": N, "max": N}     — output has N-M items
      {"type": "does_not_contain", "idea": "..."}      — output must NOT semantically contain this idea

Add new cases as you find failure modes in production.
"""

CASES = [
    # ── Should extract nothing ────────────────────────────────────────
    {
        "name": "meta_questions_about_app",
        "description": "User asks about the app itself — not learnings",
        "conversation": (
            "<user>\nwhat can you help me with?\n</user>\n"
            "<tutor>\nI can explain concepts, quiz you, help debug your understanding...\n</tutor>\n"
            "<user>\ncan you see what I'm reading?\n</user>\n"
            "<tutor>\nYes, I can see you have a Python tutorial attached.\n</tutor>"
        ),
        "expected": [],
        "assertions": [{"type": "exact_empty"}],
    },
    {
        "name": "tutor_recites_profile_data",
        "description": "Tutor says 'you like X' — that's profile data, not user speech",
        "conversation": (
            "<user>\nwhat do you know about me?\n</user>\n"
            "<tutor>\nYou're a data scientist at Google. You prefer visual explanations "
            "and learn best with worked examples. You like to explore edge cases.\n</tutor>\n"
            "<user>\ncool\n</user>"
        ),
        "expected": [],
        "assertions": [{"type": "exact_empty"}],
    },
    {
        "name": "all_cold_questions_docker",
        "description": "User asks multiple unrelated cold questions about Docker, never restates",
        "conversation": (
            "<user>\nwhat's a Docker container?\n</user>\n"
            "<tutor>\nA Docker container is a lightweight, isolated environment that "
            "packages an application and its dependencies together.\n</tutor>\n"
            "<user>\nwhat about volumes?\n</user>\n"
            "<tutor>\nVolumes let you persist data outside the container lifecycle. "
            "When a container is removed, volume data survives.\n</tutor>\n"
            "<user>\nand Docker Compose?\n</user>\n"
            "<tutor>\nDocker Compose is a tool for defining and running multi-container "
            "applications using a YAML file.\n</tutor>"
        ),
        "expected": [],
        "assertions": [{"type": "exact_empty"}],
    },
    {
        "name": "logistics_and_process_no_concept",
        "description": "User talks scheduling/feelings about studying — no concept grasped",
        "conversation": (
            "<user>\ncan you help me prep for my biology final?\n</user>\n"
            "<tutor>\nHappy to. Which topics are you most worried about?\n</tutor>\n"
            "<user>\nthe exam is in 3 days, let's just go fast and keep it simple — "
            "i keep running out of time on these\n</user>"
        ),
        "expected": [],
        "assertions": [{"type": "exact_empty"}],
    },
    {
        "name": "bare_quiz_answer_letter",
        "description": "Tutor asks multiple-choice; user replies with a bare letter",
        "conversation": (
            "<user>\nquiz me on the OSI model\n</user>\n"
            "<tutor>\nWhich layer is responsible for routing between networks? "
            "A) Data Link  B) Network  C) Transport\n</tutor>\n"
            "<user>\nB\n</user>"
        ),
        "expected": [],
        "assertions": [{"type": "exact_empty"}],
    },
    {
        "name": "relays_pasted_source_text",
        "description": "User pastes a textbook definition and asks for notes — relaying, not reasoning",
        "conversation": (
            "<user>\nhere's from my textbook: \"Osmosis is the movement of solvent "
            "across a semipermeable membrane from a region of low solute concentration "
            "to one of high solute concentration.\" make me notes on this\n</user>\n"
            "<tutor>\nSure — osmosis is solvent moving across a semipermeable membrane "
            "toward higher solute concentration.\n</tutor>\n"
            "<user>\nthanks\n</user>"
        ),
        "expected": [],
        "assertions": [{"type": "exact_empty"}],
    },
    {
        "name": "concept_amid_logistics",
        "description": "Real restatement mixed with an exam-logistics aside — extract the concept, drop the logistics",
        "conversation": (
            "<user>\nWhat's a deadlock?\n</user>\n"
            "<tutor>\nA deadlock is when two or more threads each wait on a resource "
            "another holds, so none can proceed.\n</tutor>\n"
            "<user>\noh so it's a circular wait — each thread is stuck holding one lock "
            "and waiting for the lock the other already has. btw my OS exam is tomorrow "
            "so let's move fast\n</user>"
        ),
        "expected": [
            "A deadlock is a circular wait where each thread holds one lock and waits "
            "for the lock another thread already holds"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "does_not_contain", "idea": "the user has an OS exam tomorrow"},
        ],
    },
    # ── Should extract learnings + preserve voice ─────────────────────
    {
        "name": "user_paraphrases_concept",
        "description": "User restates batch norm in their own words",
        "conversation": (
            "<user>\nWhy is batch normalization used?\n</user>\n"
            "<tutor>\nBatch normalization normalizes the inputs to each layer during "
            "training, which reduces internal covariate shift and allows higher "
            "learning rates.\n</tutor>\n"
            "<user>\nSo it's like resetting the playing field each layer so the "
            "next layer always sees data in the same range\n</user>"
        ),
        "expected": [
            "Batch normalization resets the playing field each layer so the next layer sees data in the same range"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "preserves_voice", "phrase": "resetting the playing field"},
        ],
    },
    {
        "name": "user_acknowledges_specific_concept",
        "description": "User says 'makes sense' after explanation of LSTM gates",
        "conversation": (
            "<user>\nHow do LSTMs handle long sequences?\n</user>\n"
            "<tutor>\nLSTMs use a cell state that runs through the entire sequence "
            "like a conveyor belt. Three gates — forget, input, and output — "
            "control what information is removed, added, or passed along.\n</tutor>\n"
            "<user>\nAh so the forget gate is like a filter that decides what "
            "old info to throw away\n</user>\n"
            "<tutor>\nExactly! And the input gate decides what new information "
            "to write to the cell state.\n</tutor>\n"
            "<user>\nMakes sense — so the cell state is the memory and the gates "
            "control what gets remembered\n</user>"
        ),
        "expected": [
            "The forget gate is like a filter that decides what old info to throw away",
            "The cell state is the memory and the gates control what gets remembered",
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "preserves_voice", "phrase": "filter"},
            {"type": "preserves_voice", "phrase": "gates control what gets remembered"},
        ],
    },
    {
        "name": "user_builds_on_concept",
        "description": "User extends the concept of gradient descent with their own framing",
        "conversation": (
            "<user>\nWhat's the intuition behind gradient descent?\n</user>\n"
            "<tutor>\nImagine you're on a hilly landscape in fog and want to reach "
            "the lowest point. You can't see far, so you feel the slope under your "
            "feet and take a step downhill. Gradient descent does this iteratively.\n</tutor>\n"
            "<user>\nSo the learning rate is basically how big a step you take — "
            "too big and you overshoot the valley, too small and you take forever\n</user>"
        ),
        "expected": [
            "The learning rate controls step size in gradient descent — too big overshoots, too small is slow"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "preserves_voice", "phrase": "overshoot the valley"},
        ],
    },
    {
        "name": "mixed_questions_and_understanding",
        "description": "Some questions (no extract) and some demonstrated understanding",
        "conversation": (
            "<user>\nWhat's a GAN?\n</user>\n"
            "<tutor>\nA GAN has two neural networks — a generator that creates fake "
            "data and a discriminator that tries to tell real from fake. They train "
            "adversarially, each making the other better.\n</tutor>\n"
            "<user>\nWhat's mode collapse?\n</user>\n"
            "<tutor>\nMode collapse is when the generator finds one output that fools "
            "the discriminator and keeps producing only that, ignoring the diversity "
            "of the real data distribution.\n</tutor>\n"
            "<user>\nOh so it's like a student who finds one essay template that "
            "always gets a B+ and never tries anything else\n</user>"
        ),
        "expected": [
            "Mode collapse is like a student finding one essay template that always gets a B+ and never trying anything else"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "count_range", "min": 1, "max": 2},
            {"type": "preserves_voice", "phrase": "essay template"},
        ],
    },
    # ── Follow-up questions that imply understanding ──────────────────
    {
        "name": "followup_question_builds_on_concept",
        "description": "User asks a follow-up that only makes sense if they grasped the prior concept (HTTP cookies) — extract the prior concept, not the follow-up's answer",
        "conversation": (
            "<user>\nWhat is an HTTP cookie?\n</user>\n"
            "<tutor>\nA cookie is a small piece of data the server sends in a response "
            "header. The browser stores it and automatically includes it on later "
            "requests to that same domain, which lets the server recognize a "
            "returning client.\n</tutor>\n"
            "<user>\nSo if the cookie lives in the browser, what stops a user from "
            "just editing it to become someone else?\n</user>\n"
            "<tutor>\nThe server signs the cookie — if the client edits the value, the "
            "signature no longer verifies and the server rejects it.\n</tutor>"
        ),
        "expected": [
            "An HTTP cookie is data the server sends in a response header that the "
            "browser stores and sends back on later requests to the same domain, "
            "letting the server recognize a returning client"
        ],
        "assertions": [
            {"type": "semantic"},
        ],
    },
    {
        "name": "progressive_deepening_questions",
        "description": "User asks a chain of increasingly specific questions about HTTP",
        "conversation": (
            "<user>\nWhat's the difference between HTTP and HTTPS?\n</user>\n"
            "<tutor>\nHTTPS adds a TLS/SSL encryption layer on top of HTTP. The data "
            "traveling between client and server is encrypted, so intermediaries "
            "can't read or tamper with it.\n</tutor>\n"
            "<user>\nHow does TLS actually establish the encryption?\n</user>\n"
            "<tutor>\nThrough a handshake: the client and server exchange certificates, "
            "agree on a cipher suite, and generate shared session keys using "
            "asymmetric cryptography. After that, they switch to faster symmetric "
            "encryption for the actual data.\n</tutor>\n"
            "<user>\nWhy not just use asymmetric the whole time?\n</user>\n"
            "<tutor>\nAsymmetric encryption is computationally expensive — roughly "
            "1000x slower than symmetric. It's great for securely exchanging a "
            "small key, but too slow for encrypting bulk data.\n</tutor>"
        ),
        "expected": [
            "HTTPS adds TLS/SSL encryption on top of HTTP to prevent intermediaries from reading or tampering with data",
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "count_range", "min": 2, "max": 3},
        ],
    },
    {
        "name": "followup_vs_cold_question",
        "description": "Follow-up implies learning, but a cold question at the end should not be extracted",
        "conversation": (
            "<user>\nWhat is a hash table?\n</user>\n"
            "<tutor>\nA hash table maps keys to values using a hash function that "
            "converts keys into array indices. This gives O(1) average-case "
            "lookups.\n</tutor>\n"
            "<user>\nWhat happens when two keys hash to the same index?\n</user>\n"
            "<tutor>\nThat's a collision. Common solutions are chaining (linked list "
            "at each slot) or open addressing (probe for the next open slot).\n</tutor>\n"
            "<user>\nWhat's a binary search tree?\n</user>\n"
            "<tutor>\nA BST is a tree where each node's left children are smaller "
            "and right children are larger. This gives O(log n) search if balanced.\n</tutor>"
        ),
        "expected": [
            "A hash table maps keys to values using a hash function for O(1) average lookups",
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "count_range", "min": 1, "max": 2},
            {
                "type": "does_not_contain",
                "idea": "a BST is a tree where left children are smaller and right children are larger",
            },
        ],
    },
    {
        "name": "user_vivid_analogy_preserved",
        "description": "User creates a vivid analogy — extraction must not flatten it",
        "conversation": (
            "<user>\nHow does a load balancer work?\n</user>\n"
            "<tutor>\nA load balancer distributes incoming requests across multiple "
            "servers so no single server gets overwhelmed. It can use round-robin, "
            "least connections, or weighted algorithms.\n</tutor>\n"
            "<user>\nSo it's basically a restaurant host — seats customers at "
            "different tables so no waiter gets slammed\n</user>"
        ),
        "expected": [
            "A load balancer is like a restaurant host — seats requests at different servers so none gets slammed"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "preserves_voice", "phrase": "restaurant host"},
        ],
    },
    # ── Self-corrections (rule 8) ─────────────────────────────────────
    {
        "name": "self_correction_cnn_pooling",
        "description": "User says pooling adds parameters, gets corrected, restates correctly",
        "conversation": (
            "<user>\nPooling layers in CNNs learn filters just like conv layers right?\n</user>\n"
            "<tutor>\nActually pooling layers have no learnable parameters at all. "
            "They just apply a fixed operation — like taking the max or average over "
            "a window. That's part of why they're useful: they reduce spatial dimensions "
            "without adding parameters.\n</tutor>\n"
            "<user>\nOh right, so pooling is purely mechanical — just shrinks the "
            "feature map by taking max or average, no weights involved\n</user>"
        ),
        "expected": [
            "Pooling layers have no learnable parameters — they just shrink the feature map by taking max or average over a window"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "does_not_contain", "idea": "pooling layers learn filters like conv layers"},
        ],
    },
    {
        "name": "self_correction_precision_recall",
        "description": "User confuses precision and recall, corrects after explanation",
        "conversation": (
            "<user>\nPrecision is about not missing any positives right? Like catching every spam email\n</user>\n"
            "<tutor>\nYou've got it flipped — catching every positive is actually recall. "
            "Precision is about accuracy when you do predict positive: of all emails you "
            "flagged as spam, how many actually were spam? Recall is: of all actual spam, "
            "how many did you catch?\n</tutor>\n"
            "<user>\nGot it — precision is about not crying wolf, recall is about "
            "not missing the wolf\n</user>"
        ),
        "expected": [
            "Precision is about not crying wolf (accuracy of positive predictions), recall is about not missing the wolf (catching all actual positives)"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "preserves_voice", "phrase": "crying wolf"},
            {"type": "preserves_voice", "phrase": "missing the wolf"},
            {"type": "does_not_contain", "idea": "precision is about not missing any positives or catching every spam email"},
        ],
    },
    {
        "name": "self_correction_no_restatement",
        "description": "User gets corrected but never restates — nothing to extract",
        "conversation": (
            "<user>\nDropout randomly removes neurons permanently during training right?\n</user>\n"
            "<tutor>\nNot permanently — dropout only masks neurons temporarily during "
            "each forward pass. At test time all neurons are active. The key is that "
            "it's stochastic and temporary, which forces redundancy.\n</tutor>\n"
            "<user>\nWhat about batch size, does that matter?\n</user>\n"
            "<tutor>\nYes, batch size affects the noise in gradient estimates...\n</tutor>"
        ),
        "expected": [],
        "assertions": [
            {"type": "exact_empty"},
        ],
    },
    # ── Socratic responses (rule 9) ───────────────────────────────────
    {
        "name": "socratic_user_reasons_through_probe",
        "description": "Tutor asks guiding question, user reasons through it — extract the reasoning",
        "conversation": (
            "<tutor>\nYou understand how word2vec maps words to vectors. So why do "
            "you think we can't just use one-hot encoding instead of embeddings?\n</tutor>\n"
            "<user>\nI think because one-hot vectors are all orthogonal — there's no "
            "notion of similarity. 'cat' and 'dog' would be just as far apart as "
            "'cat' and 'refrigerator'. Embeddings fix that by putting similar words "
            "close together\n</user>\n"
            "<tutor>\nExactly. And there's a practical issue too — with a 50k word "
            "vocabulary, one-hot vectors are 50k-dimensional and extremely sparse.\n</tutor>\n"
            "<user>\nRight, so it's also a dimensionality problem — embeddings compress "
            "that into a dense vector of maybe 300 dimensions\n</user>"
        ),
        "expected": [
            "One-hot encoding makes all words equidistant with no notion of similarity — 'cat' and 'dog' are as far apart as 'cat' and 'refrigerator'. Embeddings put similar words close together.",
            "Embeddings also solve a dimensionality problem — compressing a sparse 50k-dimensional one-hot vector into a dense vector of maybe 300 dimensions",
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "preserves_voice", "phrase": "refrigerator"},
            {"type": "count_range", "min": 1, "max": 3},
        ],
    },
    {
        "name": "socratic_user_says_idk",
        "description": "Tutor probes but user says they don't know — nothing to extract",
        "conversation": (
            "<tutor>\nYou've worked with SQL joins before. What do you think happens "
            "when you join two tables and there's no matching row in the right table?\n</tutor>\n"
            "<user>\nHmm I'm not sure actually\n</user>\n"
            "<tutor>\nNo worries. With a LEFT JOIN, you'd get the left table's row "
            "with NULLs for all the right table's columns. With an INNER JOIN, "
            "that row would be excluded entirely.\n</tutor>"
        ),
        "expected": [],
        "assertions": [
            {"type": "exact_empty"},
        ],
    },
    {
        "name": "socratic_user_partially_right",
        "description": "User reasons through probe, gets part right — extract only what they articulated",
        "conversation": (
            "<tutor>\nYou know that neural networks stack layers. Why do you think "
            "we need activation functions between layers?\n</tutor>\n"
            "<user>\nI think without them, stacking layers would be pointless? Like "
            "two linear transformations in a row is just another linear transformation, "
            "so you'd never learn anything more complex\n</user>\n"
            "<tutor>\nSpot on — that's the key insight. Without non-linearity, a "
            "100-layer network collapses to a single linear transformation. The "
            "activation function is what lets depth actually matter. ReLU is popular "
            "because it's cheap to compute and avoids the vanishing gradient problem "
            "that sigmoid has.\n</tutor>"
        ),
        "expected": [
            "Without activation functions, stacking layers is pointless because two linear transformations collapse into one — the network can never learn anything more complex"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "does_not_contain", "idea": "ReLU is popular because it avoids vanishing gradients"},
            {"type": "count_range", "min": 1, "max": 2},
        ],
    },
    # ── Mixed Socratic + self-correction ──────────────────────────────
    {
        "name": "socratic_probe_leads_to_correction",
        "description": "Tutor probes, user reasons incorrectly, gets corrected, then restates",
        "conversation": (
            "<tutor>\nYou understand TCP basics. What do you think happens when a "
            "packet gets lost in transit?\n</tutor>\n"
            "<user>\nThe receiver sends a request back asking for the specific packet "
            "to be resent?\n</user>\n"
            "<tutor>\nClose, but it's actually the other way around. The receiver doesn't "
            "explicitly request the missing packet. Instead, TCP uses acknowledgments — "
            "the sender notices it never got an ACK for that packet and retransmits "
            "after a timeout. The receiver just ACKs what it has received.\n</tutor>\n"
            "<user>\nAh so it's sender-driven — the sender tracks which ACKs came back "
            "and retransmits anything that timed out, rather than the receiver asking "
            "for specific packets\n</user>"
        ),
        "expected": [
            "TCP retransmission is sender-driven — the sender tracks which ACKs came back and retransmits anything that timed out, rather than the receiver requesting specific packets"
        ],
        "assertions": [
            {"type": "semantic"},
            {"type": "preserves_voice", "phrase": "sender-driven"},
            {"type": "does_not_contain", "idea": "the receiver sends a request asking for the specific packet to be resent"},
        ],
    },
    # ── Regressions from real DDPM-paper tutoring session ─────────────
    # Each case captures a failure mode observed in production where the
    # extractor saved tutor-spoken content or raw user questions as if the
    # user had demonstrated understanding.
    {
        "name": "ddpm_dangling_deictic_pinned_passage",
        "description": "User pins to a passage and asks 'what is this' — cold question with pinned context; should not extract the tutor's definition",
        "conversation": (
            '<pinned source="2006.11239 p.4">the variational bound on the negative log likelihood</pinned>\n'
            "<user>\nremind me what this is measuring\n</user>\n"
            "<tutor>\nThat variational bound is a tractable upper bound on the negative "
            "log-likelihood that DDPM training minimizes — pushing it down makes the "
            "model assign higher likelihood to the real data.\n</tutor>"
        ),
        "expected": [],
        "assertions": [
            {"type": "exact_empty"},
            {
                "type": "does_not_contain",
                "idea": "the variational bound is a tractable upper bound on the negative log-likelihood that training minimizes",
            },
        ],
    },
    {
        "name": "ddpm_cold_chain_markov_and_transition",
        "description": "User asks multiple cold probability questions; tutor explains Markov chain and Gaussian transition; user never restates",
        "conversation": (
            "<user>\nslow down. I don't quite remember the probability and the "
            "equations are confusing me.\n\nwhat is joint distribution? what is "
            "markov chain? what is a Gaussian transition?\n</user>\n"
            "<tutor>\nA Markov chain is a process where the next state depends only "
            "on the current state — not the whole history. Once you know x_t, the "
            "way you go to x_{t-1} doesn't care about earlier x_{t+1}, x_{t+2}, … "
            "A transition p_theta(x_{t-1}|x_t) is the rule for sampling x_{t-1} "
            "given x_t.\n</tutor>"
        ),
        "expected": [],
        "assertions": [
            {"type": "exact_empty"},
            {
                "type": "does_not_contain",
                "idea": "once you know x_t, the way you go to x_{t-1} doesn't depend on earlier states",
            },
            {
                "type": "does_not_contain",
                "idea": "a transition is the rule for sampling x_{t-1} given x_t",
            },
        ],
    },
    {
        "name": "ddpm_numbered_question_list_not_a_learning",
        "description": "User asks a numbered list of cold questions; the raw question text must never be saved as if it were a learning",
        "conversation": (
            "<user>\n1. what is the forward noising process?\n2. what does the reverse "
            "process learn?\n3. what is the reparameterization trick doing here?\n</user>\n"
            "<tutor>\nThe forward process gradually adds Gaussian noise over many steps "
            "until the data is pure noise. The reverse process learns to denoise one "
            "step at a time. The reparameterization trick lets you sample x_t directly "
            "and backpropagate through the sampling.\n</tutor>"
        ),
        "expected": [],
        "assertions": [
            {"type": "exact_empty"},
            {
                "type": "does_not_contain",
                "idea": "what the forward noising process is and what the reverse process learns",
            },
        ],
    },
    {
        "name": "ddpm_covariance_vs_std_no_restatement",
        "description": "User asks 'why covariance and not std deviation' cold across multiple turns; tutor explains; user never restates",
        "conversation": (
            "<user>\nwhy is it covariance and not std deviation? what is the "
            "difference between the two?\n</user>\n"
            "<tutor>\nCovariance and standard deviation are related — mostly about "
            "dimension. In 1D, std deviation is sqrt(variance) and covariance "
            "reduces to variance. In multi-D, the covariance matrix captures "
            "correlations between dimensions — diagonal entries are variances, "
            "off-diagonals are how dimensions co-vary. Standard deviation only "
            "gives per-dimension width without correlations.\n</tutor>\n"
            "<user>\nwhat is the formula for variance? covariance?\n</user>\n"
            "<tutor>\nFor a scalar Y, Var(Y) = E[(Y - E[Y])^2]. For two scalars, "
            "Cov(Y,Z) = E[(Y - E[Y])(Z - E[Z])].\n</tutor>"
        ),
        "expected": [],
        "assertions": [
            {"type": "exact_empty"},
            {
                "type": "does_not_contain",
                "idea": "covariance is used because in multi-dimensional cases it captures correlations between dimensions, while std deviation only gives per-dimension spread",
            },
        ],
    },
    {
        "name": "action_log_outcome_not_a_learning",
        "description": "User narrates the step-by-step actions and outcome of a security lab; only the principle they actually state may be captured, never the play-by-play",
        "conversation": (
            "<user>\nfor this lab how do I get at another user's invoices?\n</user>\n"
            "<tutor>\nIf an endpoint trusts a user-supplied object id without checking "
            "who owns it, anyone can read another user's records just by changing the "
            "id in the URL.\n</tutor>\n"
            "<user>\nright, so if the endpoint never checks who owns the invoice, "
            "changing the id in /invoices/1043 to 1044 just loads someone else's — I "
            "tried it and scraped a few hundred with a quick script\n</user>"
        ),
        "expected": [
            "If an endpoint does not verify ownership of a user-supplied object id, "
            "changing that id loads another user's records"
        ],
        "assertions": [
            {"type": "count_range", "min": 0, "max": 1},
            {
                "type": "does_not_contain",
                "idea": "I tried it and scraped a few hundred invoices with a quick script",
            },
        ],
    },
]
