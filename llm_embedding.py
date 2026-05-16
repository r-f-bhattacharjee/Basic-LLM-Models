
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import numpy as np

load_dotenv()

query = "What is Paytm?"
docs = [ 
        "It offers a wide range of services including mobile payments, e-commerce, and financial products.", 
        "Paytm allows users to make payments, transfer money, and access various financial services through its app and website.", 
        "Paytm is a digital payments and financial services company based in India.",
        "It has become one of the leading digital payment platforms in India, serving millions of users and businesses."]

model_query = OpenAIEmbeddings(model='text-embedding-3-small')
model_docs = OpenAIEmbeddings(model='text-embedding-3-small')

query_result = model_query.embed_query(query)
docs_result = model_docs.embed_documents(docs)


V1=np.array(query_result)
V2=np.array(docs_result)
print('Query vectors:', V1)
print('Document vectors:', V2)

N1=np.linalg.norm(V1)
N2=np.linalg.norm(V2)
print('Norm of query vector:', N1)
print('Norm of document vectors:', N2)

Scores=np.dot(V1, V2.T) / (N1 * N2)
print('Cosine similarity scores:', Scores)

index=np.argsort(Scores)[-1]
print('Most relevant document:', docs[index])


