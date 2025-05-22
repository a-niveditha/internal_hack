import requests
import fitz  # Package Name-PyMuPDF

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def extract_resume_sections():

    file_id = "64c3f2a4c1a3fabc12345678"  #replace with actual ID
    url = f"http://<backend-ip>:8000/pdf/{file_id}" #replace this too
    response = requests.get(url)
    with open("temp.pdf", "wb") as f:
        f.write(response.content)

    doc = fitz.open("temp.pdf")
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

def calculate_similarity(summarised_data):

    model = SentenceTransformer("all-mpnet-base-v2")
    query = model.encode("")

    similarity_values=[]
    for i in summarised_data:
        similarity_values.append(cosine_similarity([model.encode(summarised_data[i])],[query])[0][0] )
        
    return(max(similarity_values))

def main_block():
    resume_info = extract_resume_sections()

    summarised={}
    overall_summary=""
    for key, value in resume_info.items():
        summarised[key]=find_summary(value)
        overall_summary+=summarised[key]

    score=calculate_similarity(summarised)*100
    result={"Score":float(score),"Summary":overall_summary}

    return result 

final_result=main_block() 


