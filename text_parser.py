import fitz  # Package Name-PyMuPDF

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def extract_resume_sections(pdf_path):

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    if text=="" :
        return None

    text_lower = text.lower()

    keywords=["education","projects","professional experience","technical skills","skills", "internships","work experience","experience","accomplishments"]

    lines = text_lower.splitlines()
    sections = {}
    current_key = None
    buffer = []

    for line in lines:
        stripped = line.strip()
        if stripped in keywords:
            if current_key:
                sections[current_key] = "\n".join(buffer).strip()
            current_key = stripped
            buffer = []

        elif current_key:
            buffer.append(line)

    if current_key: #adding the final section
        sections[current_key] = "\n".join(buffer).strip()

    return sections

def find_summary(text):

    tokenizer = AutoTokenizer.from_pretrained("t5-small")
    model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")
    
    input_text="summarize:"+text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)

    summary_ids = model.generate(
    inputs["input_ids"],
    max_length=150,       # controls summary length
    min_length=30,        # optional: minimum length for output
    length_penalty=2.0,   # higher = shorter summary
    num_beams=4,          # beam search for better quality
    early_stopping=True   # stop when model is confident
    )

    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return str(summary)

resume_info = extract_resume_sections(r"D:\Codes\csi_25_internal_hackathon\trial_resume2.pdf")
if resume_info==None:
    print("Could not fetch data")

summarised={}
for key, value in resume_info.items():
    summarised[key]=find_summary(value)
    print(key,"\n",summarised[key])

model = SentenceTransformer("all-miniLM-L6-v2")
query = model.encode("Seeking a Human Resources Director with experience in HRIS development, recruiting, FMLA, benefit administration, and policy development. Candidate must have worked in a healthcare environment and be skilled in web page development, OSHA compliance, employee handbooks, budget management, and strategic planning. Experience with database systems and managing full-cycle recruitment is essential. Master's degree in Information Management Systems is preferred.")

#similarity = model.similarity(query_embedding, passage_embeddings)
similarity,j=0,0
for i in summarised:
    similarity+=cosine_similarity([model.encode(summarised[i])],[query])[0][0]
    print(i,similarity)

print("\n\n",similarity/3) 