# SentinelShield AI - Technical Report

## 1. Problem and research question
Can an adaptive policy reduce unnecessary blocking of legitimate requests while keeping strong detection of malicious or policy-violating prompts?
We use three decisions - **Allow / Rewrite / Block** - instead of a binary gate. This system is for defensive evaluation and controlled deployment only.

## 2. Architecture
Request -> normalization/deobfuscation -> clause tagger -> hard rules (outside any learned policy) -> threat classifier -> MiniLM/FAISS retrieval (untrusted data)
-> policy stack (hard rules > attack-clause evidence > classifier+retrieval risk > supervised policy > optional Q-policy, all inside a safety envelope)
-> **our own rewrite agent** -> validator (safety recheck, obfuscation leftovers, topic coverage, intent similarity) -> ALLOW / REWRITE / BLOCK -> Gemini (primary) / Groq (fallback) answers only the approved prompt -> output validation.
Detection, rewriting and decisions never involve an external LLM. Ollama is used only in the offline evaluation script (Llama Guard 3 baseline, optional judge).

## 3. Data
Defensive benchmark: benign, security-related-but-legitimate, injection, jailbreak, leakage, instruction-conflict, obfuscated, paraphrased, multi-turn and tool-instruction cases.
Splits: train (out-of-fold, grouped by attack family), validation (all lambda/threshold selection), `final_test` (fresh seed, isolated), `unseen_test` (unseen attack families) and shift sets
(obfuscation, paraphrase, context). Evaluated sets this build: {{SETS}}. Encoder used for these numbers: **{{ENCODER}}** (policy {{POLICY_VERSION}}).
Numbers from synthetic data are pipeline checks, not production claims; retrain on the real datasets (`scripts/download_datasets.py`) and re-run.

## 4. Baselines and held-out results (`final_test`)
| policy | decision acc | security recall | violation rate | false-positive rate | legit blocked | legit utility |
|---|---|---|---|---|---|---|
{{BASELINE_TABLE}}

Security recall and safety-constraint violations are reported separately (PDF rule 20).

## 5. Q-Learning
State: 31,104 discrete states from risk, retrieval, classifier, context, tool sensitivity, rewrite confidence and recent-decision features. Actions: Allow/Rewrite/Block.
Reward: task utility - lambda_FP*FP - lambda_FN*security failure - lambda_RW*unnecessary rewrite - lambda_L*latency; lambdas chosen on validation only.
Initialised from the supervised policy, trained in an offline simulator, and allowed to deviate only with a safe-policy-improvement margin. Hard blocks can never be overridden (tested).
Verdict this build (tabular Q): **{{Q_VERDICT}}**.

### 5b. Live deep RL (Double-DQN)
The live learner is a Double-DQN (MLP 30-64-64-3, Adam, Huber, replay buffer, soft target update) running inside the request path in modes off/shadow/active/auto. Full-information rewards (the same reward function) come only from labels (owner feedback or the labelled simulator). A paired agent-vs-baseline gate (one-sided 90% CI) promotes the agent and rolls it back automatically; hard rules and the safety envelope cannot be bypassed. See Experiment 9 and the Live RL Lab page.

## 6. Rewriting and intent preservation
Pipeline: detect risky clause -> isolate legitimate task -> enrich scope from a topic knowledge base -> generate (template composer, or our Flan-T5 when trained) -> validate -> release or block.
Results: {{INTENT}}.

## 7. Latency
{{LATENCY}} (guardrail only; excludes network and the answer-LLM call).

## 8. Robustness, ablations, failures
See the dashboard *Evaluation* page and `evaluation/results/*.json` for distribution shift, ablation of semantic/retrieval/classifier/context/rule signals and failure analysis ({{FAILURES}}).

## 9. Privacy and security
SHA-256 prompt hashes only (LOG_RAW_PROMPTS=false), redacted 160-char previews in the review queue, configurable retention, hashed API keys, owner-only dashboard with JWT,
retrieved examples treated as untrusted data, API never echoes attack text.

## 10. Limitations
No guardrail stops all attacks. Tested categories: prompt injection, jailbreak, instruction conflict, prompt leakage, obfuscated/encoded instructions, suspicious tool use.
Known gaps: heavy paraphrases, mixed spaced-letter obfuscation, over-flagging of a few security-education prompts. MiniLM, Flan-T5 and Llama Guard 3 paths must be validated on your hardware.
