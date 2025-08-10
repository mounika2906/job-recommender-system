import chromadb
from langchain.embeddings import OpenAIEmbeddings

def create_chroma_collection(collection_name="jobs"):
    client = chromadb.Client()
    try:
        collection = client.get_collection(collection_name)
    except:
        collection = client.create_collection(name=collection_name)
    return collection

def add_jobs_to_collection(collection, jobs, embeddings_model):
    job_texts = [job["text"] for job in jobs]
    embeddings = embeddings_model.embed_documents(job_texts)
    ids = [job["id"] for job in jobs]
    collection.add(ids=ids, documents=job_texts, embeddings=embeddings)

def query_collection(collection, query_text, embeddings_model, top_k=5):
    query_embedding = embeddings_model.embed_query(query_text)
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    return results["documents"]
