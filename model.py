from sentence_transformers import SentenceTransformer

model = SentenceTransformer("multi-qa-mpnet-base-cos-v1")

query_embedding = model.encode("We are seeking a public health consultant with expertise in patient education, preventive medicine, and lifestyle intervention programs. The ideal candidate will have experience developing health promotion initiatives, managing grants, and collaborating with universities or healthcare institutions. Responsibilities include supporting physicians and medical residents, contributing to wellness program development, and providing strategic input on healthcare training, nutrition, and disease prevention. A background in family medicine education, community health, and corporate wellness is preferred")
passage_embeddings = model.encode()


similarity = model.similarity(query_embedding, passage_embeddings)
print(similarity)




