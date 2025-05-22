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
overall_summary=""
for key, value in resume_info.items():
    summarised[key]=find_summary(value)
    overall_summary+=summarised[key]

model = SentenceTransformer("all-mpnet-base-v2")
query = model.encode("Human Resources professional with over two decades of experience leading recruitment efforts, shaping HR policy, and driving organizational development. Skilled in managing complex HR systems, streamlining administrative workflows, and ensuring compliance with employment regulations such as OSHA, FMLA, and workers' compensation. Demonstrated success in reducing staffing costs, improving hiring efficiency, and implementing innovative HRIS and database solutions to support talent acquisition and retention. Experienced in both public and healthcare sectors, with a strong foundation in employee relations, benefits administration, and strategic planning. Recognized for initiating change management initiatives, developing training programs, and enhancing internal communications through digital tools and web platforms.")

similarity_values=[]
for i in summarised:
    similarity_values.append(cosine_similarity([model.encode(summarised[i])],[query])[0][0] )
    
print(max(similarity_values)) 
