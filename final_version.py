import requests
import fitz  # Package Name-PyMuPDF
import json 

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

def extract_resume_sections():

    doc = fitz.open(r"D:\Codes\csi_25_internal_hackathon\trial_resume3.pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    if text=="" :
        return None

    text_lower = text.lower()
    keywords=["education","projects","professional experience","technical skills","skills", "internships","work experience","experience","work history","accomplishments"]
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

    GROQ_API_KEY = "" #need to add the api key.

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json"
    }

    url = "https://api.groq.com/openai/v1/chat/completions"

    data = {
        "model": "llama3-8b-8192",  
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that summarizes text into coherent, understandable sentences."
            },
            {
                "role": "user",
                "content": "Rules to follow while summarising: when the text is about education just take the course content and when it is based on experience extract the skills and duration from the data and give it as generalised terms like decade/several couple of years.Also just give me the summary, DONT GIVE ANY EXTRA TEXT BEFORE SUMMARY, THE OUTPUT SHOULD JUST BE SUMMARY. The text which is to be summarised:"+text
            }
        ],
        "temperature": 0.5, #creativity.
        "max_tokens": 500
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code == 200:
        summary = response.json()["choices"][0]["message"]["content"]
        #print("📄 Summary:\n", summary)
        return(str(summary))
    else:
        print("🚨 Error:", response.status_code, response.text)

def calculate_similarity(summarised_data):

    model = SentenceTransformer("all-mpnet-base-v2")
    #query = model.encode("")
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

    score=(calculate_similarity(summarised)+1)*50
    result={"Score":float(score),"Summary":summarised}

    return result 

final_result=main_block() 
print(final_result["Score"])
for i,j in final_result["Summary"].items():
    print(i,"\n",j)


