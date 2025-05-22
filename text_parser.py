import fitz  # Package Name-PyMuPDF

import spacy
import pytextrank

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

def find_summary(ext_data):

    nlp = spacy.load("en_core_web_lg")
    nlp.add_pipe("textrank")
    print("original size=",len(ext_data))
    doc = nlp(ext_data)

    for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
        #sent- is a token object not raw string
        print('Summary Length:',len(sent))
        return str(sent)

resume_info = extract_resume_sections(r"D:\Codes\csi_25_internal_hackathon\trial_resume2.pdf")
if resume_info==None:
    print("Could not fetch data")

summarised={}
for key, value in resume_info.items():
    #print(f"\n--- {key.upper()} ---\n{value}")
    summarised[key]=find_summary(value)

for key,value in summarised.items():
    print(key.upper(),"\n",value) 
    

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-miniLM-L6-v2")

query = model.encode("Seeking a Human Resources Director with experience in HRIS development, recruiting, FMLA, benefit administration, and policy development. Candidate must have worked in a healthcare environment and be skilled in web page development, OSHA compliance, employee handbooks, budget management, and strategic planning. Experience with database systems and managing full-cycle recruitment is essential. Master's degree in Information Management Systems is preferred.")

#passage_embeddings = model.encode(str(sent))
#similarity = model.similarity(query_embedding, passage_embeddings)
similarity,j=0,0
for i in summarised:
    similarity+=cosine_similarity([model.encode(summarised[i])],[query])[0][0]
    print(i,similarity)

print("\n\n",similarity/3) 