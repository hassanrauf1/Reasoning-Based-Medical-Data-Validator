# 🧠 Reasoning-Based Medical Data Validator 

This project uses Large Language Models (LLMs) to **generate**, **validate**, and **reason** over synthetic medical data using structured prompts and OpenRouter API. It reproduces the logic from Roy Ziv’s reasoning pipeline built with OpenAI, adapted for public use via OpenRouter.

> 🧬 Build intelligent agents that detect medical data inconsistencies like allergy contradictions, lab vs diagnosis mismatches, or incorrect treatments — with explainable outputs.

---

## 🚀 Features

- ✅ Synthetic generation of medical records (with intentional mistakes)
- 🧠 LLM-based reasoning validation per row (using JSON format)
- 🧪 Precision/Recall/F1/Accuracy evaluation
- 🧵 Fast multi-threaded inference
- 🌍 Powered by **OpenRouter** (supports GPT-4o, Mixtral, Claude, etc.)

---

## 🔄 Pipeline Overview

graph TD
    A[Start] --> B[Generate Synthetic Data (CSV rows)]
    B --> C[Iterate Over Rows]
    C --> D[Validate Each Row via LLM Reasoning]
    D --> E[Parse JSON Validity & Issue]
    E --> F[Compare to Ground Truth]
    F --> G[Compute Metrics]
    G --> H[Print Metrics & Mismatches]
    
📦 your-project-folder/
 ┣ 📜 data_validator_agent.py
 ┣ 📄 requirements.txt
 ┣ 📄 README.md           # ← You are here
 ┗ 📁 images/             # Optional screenshots

## 🧠 Prompt Logic Used by the Agent
You are a helpful assistant designed to validate the quality of medical datasets.

- Carefully analyze the row for inconsistencies, contradictions, missing values, or implausible information.
- Focus on relationships between fields: treatments vs diagnoses, allergies vs medications, lab values vs diagnosis.

Return only a JSON object with:

{
  "is_valid": true | false,
  "issue": "brief explanation or null"
}

---

## 📊 Sample Output

📊 Evaluation Metrics:
Precision: 0.92
Recall:    0.88
F1 Score:  0.90
Accuracy:  0.91

❌ Mismatches Found:
Patient ID: P004
Truth: Invalid
Prediction: Valid
Issue: Prescribed Amoxicillin despite Penicillin allergy



