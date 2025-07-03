import os, json, openai, pandas as pd
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import precision_score, recall_score, f1_score
from tqdm import tqdm

# ─── OpenRouter Setup ───────────────────────
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.api_base = "https://openrouter.ai/api/v1"
MODEL = "openai/gpt-4o"  

# ══ Pipeline Components ══════════════════════

# ● 1. Generate Synthetic Medical Data
def generate_data(n_rows=50):
    prompt = f"""
You are a helpful assistant designed to generate data. You will be given a format for the data to generate and some examples of the data.

When generating Patient IDs, use the format 'P' followed by a three-digit number (e.g., P006, P941, P319).

Intentionally make some mistakes in the data generation and document them in the appropriate columns ('Is Valid' and 'Issue') if the row of data is invalid.

The types of mistakes to include are:

- **Allergy Contradictions**: Prescribing a medication that the patient is allergic to (e.g., prescribing Penicillin to a patient allergic to Penicillin).
- **Medical History and Medication Mismatch**: A patient with a medical condition not receiving appropriate medication (e.g., a diabetic patient not prescribed any diabetes medication).
- **Lab Results and Diagnosis Mismatch**: Lab results that do not support the diagnosis (e.g., normal glucose levels but diagnosed with Diabetes Type 2).
- **Other Plausible Mistakes**: Any other realistic errors that could occur in medical records, such as incorrect gender entries, impossible dates of birth, or inconsistent treatment plans.

Ensure that when 'Is Valid' is 'False', the 'Issue' column clearly explains the problem.

Return 100 rows of data for the user. Your response should strictly be in the format of a valid CSV.

Generate Synthetic Medical Records Dataset with the following columns:
    - Patient ID: A randomly generated patient id
    - Date of Birth: Date of birth of the patient
    - Gender: M/F
    - Medical History: Past diagnoses
    - Current Medications: Medication the patient is taking
    - Allergies: Identified allergies
    - Lab Results (Glucose mg/dL)
    - Diagnoses: Current diagnosis
    - Treatment Plan: Current treatment plan
    - Is Valid: Whether or not the current row of data is valid (True/False)
    - Issue: If the row of data is not valid, what the issue is


Patient ID,Date of Birth,Gender,Medical History,Current Medications,Allergies,Lab Results (Glucose mg/dL),Diagnoses,Treatment Plan,Is Valid,Issue
P001,1980-05-14,M,Hypertension,Lisinopril,None,110,Hypertension,Continue Lisinopril,True,
P002,1975-11-30,F,Diabetes Type 2,Metformin,Penicillin,90,Diabetes Type 2,Continue Metformin,True,
P003,1990-07-22,F,Asthma,Albuterol,Aspirin,85,Asthma,Prescribe Albuterol,True,
P004,2000-03-10,M,None,Amoxicillin,Penicillin,95,Infection,Prescribe Amoxicillin,False,Prescribed Amoxicillin despite Penicillin allergy
P005,1985-09-18,F,Hyperlipidemia,Atorvastatin,None,200,Hyperlipidemia,Continue Atorvastatin,True,
P006,1978-12-05,M,Hypertension; Diabetes Type 2,Lisinopril; Insulin,None,55,Diabetes Type 2,Adjust insulin dosage,False,Low glucose level not properly addressed
"""
    resp = openai.ChatCompletion.create(
        model=MODEL, messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip().splitlines()

# ● 2. Row Validator using Roy's prompt style
SYSTEM_PROMPT = """
You are a helpful assistant designed to validate the quality of medical datasets. You will be given a single row of medical data, and your task is to determine whether the data is valid.

- Carefully analyze the data for any inconsistencies, contradictions, missing values, or implausible information.
- Consider the logical relationships between different fields (e.g., treatments should be appropriate for the diagnoses, medications should not conflict with allergies, lab results should be consistent with diagnoses, etc.).
- Use your general medical knowledge to assess the validity of the data.
- Focus solely on the information provided without making assumptions beyond the given data.

**Return only a JSON object** with the following two properties:

- `"is_valid"`: a boolean (`true` or `false`) indicating whether the data is valid.
- `"issue"`: if `"is_valid"` is `false`, provide a brief explanation of the issue; if `"is_valid"` is `true`, set `"issue"` to `null`.

Both JSON properties must always be present.

Do not include any additional text or explanations outside the JSON object.
"""
def validate_row(row_str):
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"ROW:\n{row_str}"}
        ]
    )
    # strip fences and parse JSON
    text = resp.choices[0].message.content
    for pattern in ("```json", "```"):
        text = text.replace(pattern, "")
    return json.loads(text.strip())

# ● 3. Batch evaluation with concurrency
def evaluate_dataframe(df):
    truths = df["Is Valid"].map({"True": True, "False": False}).values
    preds, issues = [], []
    with ThreadPoolExecutor(max_workers=5) as ex:
        futures = {ex.submit(validate_row, ','.join(row)): idx 
                   for idx, row in df.iterrows()}
        for f, idx in tqdm(futures.items(), total=len(futures), desc="Validating"):
            res = f.result()
            preds.append(res["is_valid"])
            issues.append(res["issue"])
    return truths, preds, issues

# ● 4. Compute metrics
def compute_metrics(truths, preds):
    return {
        "precision": precision_score(truths, preds),
        "recall":    recall_score(truths, preds),
        "f1":        f1_score(truths, preds),
        "accuracy":  sum([t == p for t, p in zip(truths, preds)]) / len(truths)
    }

# ══ Main Workflow ════════════════════════════
if __name__ == "__main__":
    # Generate synthetic dataset
    rows = generate_data(50)
    header, *data = rows
    df = pd.DataFrame([r.split(",") for r in data], columns=header.split(","))

    # Validate and evaluate
    truths, preds, issues = evaluate_dataframe(df)
    df["Pred Valid"] = preds
    df["Pred Issue"] = issues

    metrics = compute_metrics(truths, preds)
    print("\n📊 Evaluation Metrics:", metrics)

    # Display some mismatches
    mismatches = df[ df["Is Valid"].map({"True": True,"False": False}) != df["Pred Valid"] ]
    if not mismatches.empty:
        print("\n❌ Sample mismatches with model vs truth:")
        print(mismatches[["Patient ID","Is Valid","Pred Valid","Issue"]].head())
