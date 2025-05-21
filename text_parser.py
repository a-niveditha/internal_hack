import fitz  # Package Name-PyMuPDF

def extract_resume_sections(pdf_path):

    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()

    if text=="" :
        return None

    text_lower = text.lower()

    keywords=["education","projects","professional experience","technical skills","skills", "internships","work experience","experience"]

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

final_data=""
resume_info = extract_resume_sections(r"D:\Codes\csi_25_internal_hackathon\trial_resume.pdf")

for key, value in resume_info.items():
    #print(f"\n--- {key.upper()} ---\n{value}")
    final_data+=value

if resume_info==None:
    print("Could not fetch data")

#print(final_data)

import spacy
import pytextrank

nlp = spacy.load("en_core_web_lg")
nlp.add_pipe("textrank")
print("original size=",len(final_data))
doc = nlp(final_data)

for sent in doc._.textrank.summary(limit_phrases=2, limit_sentences=2):
    print(sent) #sent- is a token object not raw string
    print('Summary Length:',len(sent)) 


from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

query_embedding = model.encode("We are seeking a public health consultant with expertise in patient education, preventive medicine, and lifestyle intervention programs. The ideal candidate will have experience developing health promotion initiatives, managing grants, and collaborating with universities or healthcare institutions. Responsibilities include supporting physicians and medical residents, contributing to wellness program development, and providing strategic input on healthcare training, nutrition, and disease prevention. A background in family medicine education, community health, and corporate wellness is preferred")
passage_embeddings = model.encode(str(sent))


similarity = model.similarity(query_embedding, passage_embeddings)
print("\n\n",similarity) 