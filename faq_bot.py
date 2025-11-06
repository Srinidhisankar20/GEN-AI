# app.py
import os
import uuid
import numpy as np
import streamlit as st
from sentence_transformers import SentenceTransformer
import chromadb
from langchain_community.document_loaders import CSVLoader
from langchain_groq import ChatGroq
from dotenv import load_dotenv

# --------------------------
# Load environment variables
# --------------------------
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY", ".........")

# --------------------------
# Streamlit caching functions
# --------------------------
@st.cache_resource
def load_embedding_model(model_name='all-MiniLM-L6-v2'):
    return SentenceTransformer(model_name)

@st.cache_data
def generate_embeddings(texts, _model):
    return _model.encode(texts, show_progress_bar=True)

@st.cache_resource
def initialize_vector_store(documents, embeddings, persist_dir="vector_store"):
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(name="insurance_faq_collection")

    # Only add documents if collection is empty
    if len(collection.get()) == 0:
        ids, docs_texts, embeddings_list, metadatas = [], [], [], []
        for doc_obj, embedding in zip(documents, embeddings):
            unique_id = str(uuid.uuid4())
            ids.append(unique_id)
            docs_texts.append(doc_obj.page_content)
            embeddings_list.append(embedding.tolist())
            metadatas.append(doc_obj.metadata)
        collection.add(ids=ids, documents=docs_texts, embeddings=embeddings_list, metadatas=metadatas)
    return collection

# --------------------------
# Load CSV
# --------------------------
loader = CSVLoader(file_path="D:/InsuranceRag/data/insurance_faq.csv")
documents = loader.load()
texts = [doc.page_content for doc in documents]

# --------------------------
# Initialize Embedding Model & Vector Store
# --------------------------
embedding_model = load_embedding_model()
embeddings_file = "faq_embeddings.npy"

if os.path.exists(embeddings_file):
    embeddings = np.load(embeddings_file)
else:
    embeddings = generate_embeddings(texts, embedding_model)
    np.save(embeddings_file, embeddings)

vector_collection = initialize_vector_store(documents, embeddings)

# --------------------------
# RAG Retriever
# --------------------------
class RAGRetriever:
    def __init__(self, collection, embedding_model):
        self.collection = collection
        self.embedding_model = embedding_model

    def retrieve(self, query: str, top_k: int = 5):
        query_embedding = self.embedding_model.encode([query])[0]
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)
        retrieved_docs = []
        if results['documents'] and results['documents'][0]:
            documents = results['documents'][0]
            distances = results['distances'][0]
            for doc, distance in zip(documents, distances):
                retrieved_docs.append(doc)
        return retrieved_docs

rag_retriever = RAGRetriever(vector_collection, embedding_model)

# --------------------------
# ChatGroq LLM
# --------------------------
llm = ChatGroq(
    api_key=groq_api_key,
    model="llama-3.1-8b-instant",
    temperature=0.1,
    max_tokens=1024
)

# --------------------------
# RAG Query Function
# --------------------------
def rag_simple(query, retriever, llm, top_k=3):
    results = retriever.retrieve(query, top_k=top_k)
    context = "\n\n".join(results) if results else ""
    if not context:
        return "No relevant documents found."
    prompt = f"""Use the following context to answer the question:

Context:
{context}

Question: {query}

Answer:"""
    response = llm.invoke(prompt)
    return response.content

# --------------------------
# Streamlit UI
# --------------------------
st.set_page_config(page_title="Insurance FAQ RAG Assistant", page_icon="💡")
st.title("💡 Insurance Assistant")

top_k = st.sidebar.slider("Top K retrieved documents", 1, 10, 3)
query = st.text_input("Ask your insurance question:")

if query:
    with st.spinner("Retrieving answer..."):
        answer = rag_simple(query, rag_retriever, llm, top_k=top_k)
        st.subheader("Answer")
        st.write(answer)  # ONLY prints the answer
