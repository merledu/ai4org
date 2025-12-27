# ==============================================================
#  CGSMF Q&A Generator (100+ high-quality, logically correct)
# ==============================================================

# !pip install -q -U transformers accelerate bitsandbytes sentencepiece tqdm

import re, json, hashlib, random, time
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, pipeline
import torch

# ------------------- 1. Load quantized model -------------------
MODEL_NAME = "TinyLLaMA/TinyLLaMA-1.1B-intermediate-step-1431k-3T"

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
)

print("Loading model & tokenizer...")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=bnb_config, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device_map="auto")

# ------------------- 2. Input passage -------------------------
passage = """
Allocation of Guarantee Limits to the Participating Financial Institutions (PFIs)
All banks/MFBs/specialized institutions are eligible to participate under the CGSMF. Credit
Guarantee Limits (CGLs) will be assigned to all financial institutions involved in agri. financing
based on their exposure and potential in agricultural credit disbursements. All those banks
which are not currently involved in agri. financing may also apply to Agricultural Credit &
Microfinance Department (ACMFD) expressing their willingness to participate in the Scheme.
Upon receiving the request, guarantee limits may be allocated to the interested financial
institutions.
2. Eligibility of Borrowers
The Scheme will target small farmers/tenants across the country cultivating land, however,
irrespective of land ownership or lease/ tenancy;
a) 5 acres for irrigated land
b) 10 acres for rain-fed land
The borrowers under the Scheme should be those who do not have required collateral to secure
the bank loans and may avail one loan up to the maximum amount of Rs.100,000 at one time,
renewable upon maturity.
PFIs shall take into account the following factors while determining eligibility of the borrowers;
a) Verification of cultivation by the bank/ revenue authorities.
b) Fresh borrowers having no collateral.
c) Cash flows of the borrower/cropping pattern.
d) Obtain e-CIB record of the borrower.
e) In line with the credit policy of the Bank.
f) Be in conformity with the relevant rules and regulations.
g) Shall have valid CNIC.
Annexure - 2
CGSMF SOPs Page 2
3. Tenor of Loan
The loan tenor shall be based on cropping cycle up to a maximum period of one year. However,
for sugarcane crop with 18 months cropping cycle, tenor may be fixed in accordance to its
cropping cycle.
4. Treatment as Unsecured Financing
For the purpose of this Scheme all financing done under CGSMF shall be treated as secured to
the extent of guaranteed amount and the remaining amount shall be treated as Clean Exposure
(collateral free).
5. Provisioning
For the loans extended under CGSMF and classified as ‘Substandard’ the provision process
would be as below:
For Agricultural Loans extended by Commercial/Specialized Banks:
Classification Determinant Treatment of
Income
Provisions to be
Made
(1) (2) (3) (4)
OAEM
(Other Assets
Especially
Mentioned)
Where mark- up /interest or
principal is overdue (past
due) by 90 days from the
due date
Unrealized mark up/
interest to be put in
Memorandum
Account and not to
be credited to Income
Account except when
realized in cash
No provision is
required.
Substandard Where mark- up /interest or
principal is overdue (past
due) by one year or more
from the due date
As Above 20% of the
outstanding principal
net of guaranteed
amount under
CGSMF
Doubtful Where mark- up /interest or
principal is overdue (past
due) by one and a half year
or more from the due date
As Above 50% of the
outstanding principal
net of guaranteed
amount under
CGSMF
Loss Where mark- up /interest or
principal is overdue (past
due) beyond two years or
more from the due date
As Above 100 % of the
outstanding principal
net of guaranteed
amount under
CGSMF
Annexure - 2
CGSMF SOPs Page 3
For Microfinance Banks
Category Determinant Treatment of Income Provisioning
Requirement
OAEM
(Other Assets
Especially
Mentioned)
Loans (principal/mark-up)
is overdue for 30 days or
more but less than 60 days
The unrealized
interest / profit / mark-
up / service charges
on NPLs shall be
suspended and
credited to interest
suspense account.
No provision is
required
Substandard Loans (principal/mark-up)
is overdue for 60 days or
more but less than 90 days
As Above 25% of outstanding
principal net of
guaranteed amount
under CGSMF
Doubtful Loans (principal/mark-up)
is overdue for 90 days or
more but less than 180
days
As Above 50% of outstanding
principal net of
guaranteed amount
under CGSMF
Loss Loans (principal/mark-up)
is overdue for 180 days or
more
As Above 100% of
outstanding
principal net of
guaranteed amount
under CGSMF
6. Monitoring of Portfolio
a) Banks will submit their fresh borrowers’ data, sanctioned during the quarter, on CF-1
Form (Annexure 3) to CGO on quarterly basis.
b) The responsibility of due diligence rest with PFIs which should follow all relevant
guidelines of SBP and their internal SOPs for approving credit under the Scheme.
c) Since the system places primary responsibility on the PFIs and does not require
evaluation on part of CGO prior to disbursement of the loan amount, the credit
guarantee shall automatically stand issued, for a customer which the bank has evaluated
to be eligible and the PFI shall extend lending facility to the borrower treating him/her as
a guaranteed customer under the Scheme.
d) This, in principle, consent for a partial guarantee of maximum 50% to borrowers
evaluated as eligible by the PFIs subject to compliance with borrower’s
evaluation/eligibility criteria. Each PFI shall ensure that the outstanding guarantee
amount does not exceed its allocated guarantee limit at any time.
e) Each PFI shall report its outstanding position of all previously guaranteed loans to CGO
on quarterly basis as per the format CF-2 (Annexure 3).
Annexure - 2
CGSMF SOPs Page 4
7. Payment of Claims:
a) The bank shall have the right to lodge a claim to the extent of 50% of the outstanding
amount (Principal only), as soon as a particular borrower, is classified as
‘Substandard’, as per the respective classification criteria given under Prudential
Regulations for both Agri. Financing and Prudential Regulations for Microfinance Banks.
b) The PFIs shall lodge claims on biannual basis on prescribed formats. Claims against
loans classified as substandard during 1st January to 30th June should be filed by 20th
July and from 1st July to 31st December by 20th January to CGO.
c) All claims lodged by a PFI must be complete in terms of relevant annexure and required
information. The claims shall be verified by the Internal Audit Department of the claimant
bank under its seal and stamp. Besides, e-CIB reports of the delinquent borrowers under
the Scheme shall also be submitted along with the claims to CGO.
d) The claims will be scrutinized and verified by the CGO against the borrowers’ data within
15 days.
e) Claims shall be paid by CGO to the respective PFI within 05 working days under
intimation to ACMFD.
f) In case of multiple borrowings, from one or multiple banks, only single claim will be
reimbursed for a loan which is senior in terms of date of sanctioning among all such
sanctioned loans to single borrower.
g) SBP and or its subsidiary BSC reserves the right to conduct a special inspection of all
the claims reimbursed to a PFI through external auditors at any time.
h) If at any point in time, claims are found to be incorrect or without basis during inspection
by the Banking Inspection Department of SBP or otherwise, the same will be recovered
from PFI’s account maintained with the SBP: BSC. In such case, the PFI will be liable for
penalty under relevant provisions of the Banking Companies Ordinance 1962, and
Microfinance Institutions Ordinance 2001.
8. Recoveries
a) The payment under the CGSMF shall not obviate the PFI from its right of recovery of the
defaulted amount.
b) The PFIs shall continue with their regular procedure for recovery of loan and take all
necessary steps in this regard. The PFIs shall update their recovery status to the
CGO on bi-annual basis.
Annexure - 2
CGSMF SOPs Page 5
c) In the event of recoveries from delinquent borrowers, all such recoveries shall be
treated as the recovery of principal and the PFI shall return the proportionate amount to
CGO on bi-annual basis and will be reported to CGO on CF-4 format. CGSMF’s
proportionate share against recovery from defaulted borrowers during 1st January to 30th
June should be filed /reimbursed by 20th July and from 1st July to 31st December by 20th
January to CGO. Delayed reimbursement of recovered amount shall attract penal
action from State Bank.
d) The costs incurred on recovery efforts shall be borne by the concerned PFI and shall not
be passed on to the Guarantee Fund.
9. Reporting
The PFIs shall report to CGO on prescribed formats (Annexure 3) on quarterly, bi-annual and
annual basis. PFIs shall ensure that all the reports are duly verified and submitted within the
prescribed time limits as given below.
S. No Name of Report Frequency Time Line
1 CF-1 (Fresh Guarantee Loan Report) Quarterly 20 working days
2 CF-2 (Existing Guaranteed Loans Report) Quarterly 20 working days
3 CF-3 (Claim on Guarantee Fund Report) Bi-annually 20 calendar days
4 CF-4 (Recoveries from Delinquent
Borrowers Report)
Bi-annually 20 calendar days
10. Termination of the Scheme
The Scheme shall be terminated as and when announced by the Government. SBP will formally
issue circular in this regard.
In case of termination/discontinuation of the scheme, claims pertaining to the loans guaranteed
by CGO shall remain eligible for coverage under the scheme.
Annexure - 2
CGSMF SOPs Page 6
PART B: Roles and Responsibilities
Roles and Responsibilities of PFI:
a) Each PFI shall nominate at least two senior officials (at least VP/Equivalent) as authorized
officials/ signatories (along with their duly verified signatures, names, designations and
email / contact address) to communicate with the CGO, SBP:BSC, Karachi and ACMFD,
SBP, Karachi for matters pertaining to CGSMF.
b) Only those claims which are signed by designated authorized officials shall be
entertained by CGO.
c) The PFIs shall ensure that a separate record of all the borrowers and portfolio is
maintained as advised by CGO while observing their limits allocated by ACMFD. The
PFI shall submit reports on prescribed formats along with its soft copies to CGO.
d) The PFIs shall ensure implementation of internal controls and monitoring mechanism
for addressing the issue of adverse selection under the Scheme.
e) In addition, PFIs shall take all necessary measures to avoid multiple borrowings under the
Scheme. The PFI shall obtain an undertaking from the borrower that he /she has not
obtained any loan facility under CGSMF from any other bank.
f) PFIs shall ensure verification of all the claims along with annexure (e-CIB reports etc)
through internal audit before submission to CGO.
g) PFIs shall ensure to separately review their guaranteed portfolio and claims through
external audit or as a part of annual audit process of their financial statements at the end
of each year and obtain an independent audit certificate in this regard. Relevant extract
will be forwarded to CGO.
h) PFIs shall arrange regular training and skill development programs for its officials
working in this area particularly under this Scheme to ensure smooth processing and
management of CGSMF
"""   # ← keep the triple-quoted block exactly as you gave it

# ------------------- 3. Settings ------------------------------
OUTPUT_FILE = "cgsmf_qas.jsonl"
TARGET_QA   = 120               # we aim for >100, script stops when reached
BATCH_SIZE  = 5                 # questions per LLM call
SLEEP       = 0.5               # polite pause

# ------------------- 4. Prompt templates ----------------------
PROMPT_TEMPLATES = [
    "Generate {n} **basic** Q&A pairs for a newcomer. Keep answers short and verbatim from the text.",
    "Create {n} **explanatory** Q&A pairs (why/how). Quote the exact rule that justifies the answer.",
    "Formulate {n} **scenario-based** Q&A pairs that a branch manager might face. Answer must be provable.",
    "Produce {n} **definition / eligibility** Q&A pairs. Use only the wording in the passage.",
    "Make {n} **provisioning / claim** Q&A pairs with exact percentages and timelines."
]

def build_prompt(text: str, n: int) -> str:
    tmpl = random.choice(PROMPT_TEMPLATES)
    return f"""You are a precise document-QA expert.
Read the passage below and output **exactly {n}** Q&A pairs in the format:

Q: <question>
A: <answer>

Rules:
- Answer must be a direct quote or a provable derivation from the passage.
- No hallucination, no extra explanation.
- Keep each answer ≤ 2 sentences.

{tmpl}

-


{text}
"""

# ------------------- 5. LLM generation -----------------------
def generate_batch(text: str, n: int = BATCH_SIZE):
    prompt = build_prompt(text, n)
    out = pipe(prompt, max_new_tokens=4096, temperature=0.7, do_sample=True, top_p=0.92)[0]["generated_text"]
    # extract Q: … A: … blocks
    pairs = re.findall(r"Q:\s*(.+?)\s*A:\s*(.+?)(?=Q:|$)", out, re.DOTALL)
    return [(q.strip(), a.strip()) for q, a in pairs]

# ------------------- 6. Deduplication & saving ---------------
seen_hashes = set()

with open(OUTPUT_FILE, "a", encoding="utf-8") as f:
    generated = 0
    pbar = tqdm(total=TARGET_QA, desc="Generating Q&A")

    while generated < TARGET_QA:
        raw = generate_batch(passage, BATCH_SIZE)
        for q, a in raw:
            q_hash = hashlib.sha1(q.encode("utf-8")).hexdigest()[:12]
            if q_hash in seen_hashes:
                continue
            seen_hashes.add(q_hash)

            record = {
                "id": q_hash,
                "question": q,
                "answer": a,
                "source_passage": passage.strip()[:500] + "…"   # short preview
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
            generated += 1
            pbar.update(1)
            if generated >= TARGET_QA:
                break
        time.sleep(SLEEP)

    pbar.close()

print(f"\nFinished! {generated} unique Q&A saved to **{OUTPUT_FILE}**")