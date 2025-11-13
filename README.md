# Recruitify

Recruitify is an intelligent resume-to-job description matching tool.
It takes in job requirements provided by the recruiter, compares them with candidate resumes, and outputs a match score based on semantic similarity. This helps recruiters quickly filter out the most relevant candidates.

## ✨ Features

Smart Resume Parsing: Extracts text and structured information (experience, education, projects, etc.) from PDF resumes using PyMuPDF
.

Human-like Summarization: Generates coherent summaries of candidate profiles using Llama3-8B via the Groq API.

Accurate Matching: Uses Sentence Transformers
 to compute semantic similarity between job descriptions and resumes.

Optimized Model Choice: Balances speed and accuracy by leveraging the all-mpnet-base-v2 model for better performance on longer text.

## 🛠️ Technical Approach
1. Resume Summarization

Traditional models like TextRank rely on extractive summarization and fail to generate coherent sentences.

Lightweight transformer models (e.g., T5) still lacked human-like fluency.

Heavy models (e.g., Pegasus) provided high-quality summaries but were computationally infeasible to run locally.

Solution: Integrated the Groq API with Llama3-8B-8192, achieving high-quality, human-like summaries with manageable performance overhead.

2. Extracting Resume Sections

Resumes vary greatly in formatting. Instead of regex, Recruitfy uses a keyword-driven parser:

A curated list of possible section headers (e.g., Experience, Education, Projects) is scanned.

On encountering a keyword, parsing begins until the next keyword is found.

The extracted data is stored as a dictionary with {section_title: content}.

This approach avoids regex limitations and handles varying resume structures more effectively.

3. Computing Match Score

Uses Sentence Transformers to encode job requirements and resume sections into embeddings.

Computes cosine similarity between embeddings to determine alignment.

After testing multiple models like:

all-MiniLM-L6-v2: very fast but weaker on long texts.

all-mpnet-base-v2: slower but significantly better for paragraph-level semantic similarity.

Final choice: all-mpnet-base-v2 for accurate scoring.

## 🚀 Tech Stack

Python

PyMuPDF → Resume parsing

Groq API (Llama3-8B) → Summarization

Sentence Transformers (HuggingFace) → Embeddings & similarity scoring

