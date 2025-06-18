
SUMMARY : 

Models like TextRank  don't use deep learning models, they use extractive summarisation  algorithm which just extracts keywords based on frequency and doesn't give coherent sentences, giving only keywords. 

The lightweight models like T5 which use deep learning models like transformers still didn't give human like coherent sentences. 

The models which gave coherent sentences as summary(ex- pegasus) were too heavy to run on the local machine. 

Therefore had to use groq api to run model llama3-8b-8192 which gave human-like summary.

EXTRACTING USEFUL DATA FROM RESUME PDF:

Used package PyMuPDF to parse through the pdf and get data in the form of string. 

To get specific data like experience, projects, education I created a major keywords list which has combinations of words which could  be the title given in the resume. So a loop runs through the text, as soon as it encounters one topic from the list it starts storing the data as a string and when the iterator reaches the next keyword from the list, the string is terminated and stored in a dictionary where key is the title and value is the text. 

I didn't use regex because even though we can search for  the title line, we can't definitely say the no. of characters in each and every sub topic of different different resume. Thus we cant extract data properly without repetition. 

GETTING A MATCH SCORE:

Used package sentence transformer which supports a lot of models and has in built functions to encode sentences, find cosine similarity between 2 vectors etc. So basically this creates embeddings of sentences which gets mapped into a vector and then these vectors' cosine similarity is found. Tried with 2 different models, found out All-minilm-l6-v2 works better on sentences and not paragraphs even though it is very fast and efficient. So switched the model to All-mpnet-base-v2 which is a bit slower because of higher parameters and higher accuracy. It works very well for semantic comparison of long texts like paragraphs.
