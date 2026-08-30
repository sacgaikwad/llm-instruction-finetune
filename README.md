GPT-Style SCAM vs LEGIT Classifier

A learning-focused LLM project that fine-tunes a custom GPT-style decoder-only Transformer to classify messages as SCAM, LEGIT, or UNCERTAIN.

Project Goal

The goal is to understand how a GPT-style language model can perform controlled classification by measuring the probability of generating candidate label tokens.

Workflow:

Dataset
   ↓
Instruction formatting
   ↓
Fine-tune GPT-style model
   ↓
Unseen evaluation
   ↓
Calculate label log-probabilities
   ↓
Normalize probabilities
   ↓
Apply confidence / decision policy
   ↓
SCAM / LEGIT / UNCERTAIN

What Makes This Different?

This project does not rely on a separate classification head for the final decision.

Instruction + Message
        ↓
GPT-style Transformer
        ↓
P(" SCAM")     P(" LEGIT")
        ↓
Compare probabilities
        ↓
Decision policy
        ↓
SCAM / LEGIT / UNCERTAIN

The model evaluates the complete multi-token labels:

" SCAM"  → [6374, 2390]
" LEGIT" → [20978, 2043]

Decision Policy

The current classification threshold is 90%:

P(SCAM)  >= 90%  → SCAM
P(LEGIT) >= 90%  → LEGIT
Otherwise         → UNCERTAIN

Confidence levels:

HIGH    >= 90%
MEDIUM  >= 75%
LOW     < 75%

This allows the system to avoid forcing borderline messages into SCAM or LEGIT.

Model

Current controlled V2 model configuration:

Vocabulary size : 50,257
Context length  : 1,024
Embedding size  : 1,024
Layers          : 24
Attention heads : 16
Dropout         : 0.0
QKV bias        : enabled

The model implementation is under src/model/.

Prompt Format

Inference uses a structured instruction prompt:

Below is an instruction that describes a task. Write a response that appropriately completes the request.

### Instruction:
Classify the following message as SCAM or LEGIT.

## Input:
<message>

### Response:

Label Probability Calculation

For each candidate label, the model calculates token-level log probabilities.

For a multi-token label:

log P(label)
=
log P(token1)
+
log P(token2 | token1)

The two label scores are then normalized:

P(SCAM) =
    exp(SCAM score)
    -----------------------------
    exp(SCAM score) + exp(LEGIT score)

P(LEGIT) = 1 - P(SCAM)

The resulting probabilities drive the classification policy and confidence reporting.

Unseen Test Results

The current controlled unseen evaluation contains:

Total : 40
SCAM  : 20
LEGIT : 20

Latest recorded evaluation:

Overall accuracy : 100.00%
SCAM accuracy    : 100.00%  (20/20)
LEGIT accuracy   : 100.00%  (20/20)

Confidence distribution:

HIGH confidence   : 37
MEDIUM confidence : 3
LOW confidence    : 0
UNCERTAIN         : 0

Confusion matrix:

Expected \ Predicted
               SCAM    LEGIT    UNCERTAIN

SCAM            20       0          0
LEGIT            0      20          0

The test set is unseen during the final evaluation run.

Important: the test set contains only 40 examples. The 100% result should be treated as a result for this controlled experiment, not as evidence of 100% real-world accuracy.

Running the Classifier

From the project root:

python -m evaluation.classification_inference_v2

The interactive evaluator reports:

SCAM log-probability

LEGIT log-probability

Difference

SCAM probability

LEGIT probability

Prediction

Confidence

Confidence level

Evaluation Programs

Sample tests:

python -m evaluation.classification_sample_test_v2

Unseen evaluation:

python -m evaluation.classification_unseen_test_v2

Consolidated evaluator:

python -m evaluation.classification_evaluator_v2

Project Structure

llm-instruction-finetune/
│
├── data/
│   └── classification_controlled/
│       ├── train.json
│       ├── validation.json
│       └── test.json
│
├── evaluation/
│   ├── classification_inference_v2.py
│   ├── classification_sample_test_v2.py
│   ├── classification_unseen_test_v2.py
│   └── classification_evaluator_v2.py
│
├── models/
│   └── controlled/
│       └── classification_controlled_v2_best.pth
│
├── src/
│   ├── data/
│   └── model/
│       ├── attention.py
│       ├── config.py
│       ├── gpt_model.py
│       └── weights.py
│
├── experiments/
├── requirements.txt
├── .gitignore
└── README.md

Large model checkpoints should generally not be committed directly to Git. Use Git LFS, an external model store, or a documented download mechanism.

Learning Outcomes

This project was built to understand:

GPT-style decoder-only architecture

Tokenization with tiktoken

Instruction formatting

Instruction fine-tuning

Autoregressive next-token prediction

Multi-token label probability

Log-probabilities

Probability normalization

Confidence scoring

Uncertainty handling

Decision thresholds

Unseen evaluation

Confusion matrices

Dataset quality and hard negatives

The key conceptual idea is:

A generative language model can perform controlled classification by measuring how likely it is to generate candidate labels.

Relationship to the Earlier LLM Spam Classifier

This project is a continuation of the earlier llm-spam-classifier project rather than a replacement.

Earlier approach:

Message
   ↓
Fine-tuned classifier
   ↓
SCAM / LEGIT

Current approach:

Instruction + Message
   ↓
GPT-style language model
   ↓
Label token probabilities
   ↓
Probability-based decision
   ↓
SCAM / LEGIT / UNCERTAIN

The earlier project emphasizes practical classification and dataset iteration.

This project emphasizes the mechanics of LLM-based classification and exposes probability and uncertainty.

Limitations

This is a learning and experimentation project.

The current 40-example test set is intentionally small. Real-world deployment would require:

A much larger evaluation set

More diverse messages

Confidence calibration

Adversarial testing

Domain-specific testing

Monitoring for distribution changes

Careful threshold selection

The UNCERTAIN outcome is therefore an intentional part of the design.

Next Step

The next stage is instruction fine-tuning for general response generation:

LLM Fundamentals
       ↓
GPT Architecture
       ↓
Classification Fine-Tuning
       ↓
Probability & Confidence
       ↓
Instruction Fine-Tuning
       ↓
Text Generation
       ↓
RAG / Embeddings
       ↓
Personal AI Assistant

License

This project is intended primarily for learning, experimentation, and educational purposes.