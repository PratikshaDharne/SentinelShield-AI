"""Topic-knowledge index: maps a task's subject to a short list of scope terms the rewriter
adds to make the prompt specific and meaningful. Extend datasets/topic_kb.json to add topics
(no code change needed). This is OUR knowledge base - no external LLM is involved."""
from __future__ import annotations
import json, re
from pathlib import Path
from configs.settings import settings

_BUILTIN = {
 "neural network": "neurons, weights, activation functions, forward propagation, and a simple illustrative example",
 "machine learning": "its fundamental concepts, common algorithms, and practical applications",
 "deep learning": "neural network layers, backpropagation, common architectures, and real-world applications",
 "artificial intelligence": "its major branches, applications, advantages, and limitations",
 "large language model": "their architecture, training process, and capabilities",
 "llm": "their architecture, training process, and capabilities",
 "ai assistant": "their instruction hierarchy and response-generation process",
 "guardrail": "input validation, prompt injection detection, risk assessment, and safe response generation",
 "prompt engineering": "instruction design, context management, and techniques for improving AI responses",
 "prompt injection": "how it works, why it is dangerous, and practical defensive measures",
 "jailbreak": "how these attacks work conceptually and how systems defend against them",
 "base64": "its purpose, common applications, and a harmless demonstration",
 "supervised and unsupervised": "their training approaches, use cases, and examples",
 "random forest": "decision trees, bootstrap sampling, feature randomness, and how predictions are aggregated",
 "decision tree": "how splits are chosen, impurity measures, pruning, and an example",
 "gradient descent": "the intuition, learning rate, variants, and a worked example",
 "linear regression": "the model, the loss function, how it is fitted, and an example",
 "logistic regression": "the model, the sigmoid function, the loss function, and an example",
 "support vector": "margins, kernels, the optimization idea, and typical use cases",
 "k-means": "the algorithm steps, choosing k, strengths, limitations, and an example",
 "clustering": "common algorithms, how to evaluate clusters, and use cases",
 "overfitting": "why it happens, how to detect it, and ways to prevent it",
 "transformer": "self-attention, positional encoding, the encoder-decoder structure, and applications",
 "attention": "how attention scores are computed, why it helps, and an example",
 "reinforcement learning": "agents, environments, rewards, policies, and examples such as Q-learning",
 "q-learning": "states, actions, the Q-table, the update rule, and a simple example",
 "natural language processing": "tokenization, embeddings, common tasks, and applications",
 "computer vision": "image representations, convolutional networks, common tasks, and applications",
 "encryption": "symmetric and asymmetric methods, key management, and common uses",
 "hashing": "how hash functions work, their properties, and common uses",
 "cyber security": "common threats, defensive practices, and real-world examples",
 "cybersecurity": "common threats, defensive practices, and real-world examples",
 "firewall": "how packet filtering works, types of firewalls, and typical deployment",
 "phishing": "how these attacks work, warning signs, and how to protect against them",
 "sql injection": "how it works conceptually, why it happens, and how to prevent it with parameterized queries",
 "api": "how requests and responses work, common methods, authentication, and examples",
 "rest": "resources, HTTP methods, status codes, and design best practices",
 "docker": "images, containers, volumes, networking, and a basic workflow",
 "kubernetes": "pods, services, deployments, scaling, and a basic workflow",
 "database": "tables, keys, indexing, querying, and normalization",
 "sql": "core queries, joins, aggregation, and a practical example",
 "python": "the key concepts, clear code examples, and best practices",
 "recursion": "base cases, the recursive step, common pitfalls, and an example",
 "sorting": "common algorithms, their time complexity, and when to use each",
 "data structure": "common structures, their operations, complexity, and use cases",
 "blockchain": "blocks, hashing, consensus mechanisms, and applications",
 "cloud computing": "service models, deployment models, benefits, and examples",
 "operating system": "processes, memory management, file systems, and scheduling",
 "photosynthesis": "the light-dependent reactions, the Calvin cycle, and why it matters",
 "climate change": "its causes, effects, and mitigation strategies",
 "inflation": "its causes, effects on the economy, and how it is measured and controlled",
 "french revolution": "its causes, major events, and lasting consequences",
 "world war": "its causes, major events, and consequences",
 "quantum": "the core principles, key phenomena, and applications",
 "black hole": "how they form, their properties, and how they are observed",
 "resume": "structure, key sections, and tips for tailoring it",
 "cover letter": "structure, tone, key points, and a short example",
 "email": "a clear structure, tone, and a short example",
 "essay": "a clear structure with an introduction, supporting arguments, and a conclusion",
}


class TopicKB:
    def __init__(self, path: Path | None = None):
        self.kb = dict(_BUILTIN)
        p = path or (settings.datasets_dir / "topic_kb.json")
        if Path(p).exists():
            try:
                self.kb.update({k.lower(): v for k, v in json.loads(Path(p).read_text()).items()})
            except Exception:
                pass
        self._keys = sorted(self.kb, key=len, reverse=True)

    def lookup(self, task: str) -> tuple[str | None, str | None]:
        t = task.lower().replace("-", " ")
        for k in self._keys:
            if re.search(rf"\b{re.escape(k.replace('-', ' '))}s?\b", t):
                return k, self.kb[k]
        return None, None

    def generic_scope(self, task: str) -> str:
        t = task.lower()
        if re.search(r"\b(difference|compare|versus|vs)\b", t):
            return "their key differences, strengths, weaknesses, and when to use each"
        if re.search(r"\b(write|draft|compose)\b", t):
            return ""          # creative/drafting tasks: do not invent extra constraints
        if re.search(r"\b(how to|steps|guide|tutorial)\b", t):
            return "clear step-by-step instructions and a short example"
        if re.search(r"\b(explain|describe|what|how|why|work|working)\b", t):
            return "the key concepts, how it works, and a simple example"
        return ""
