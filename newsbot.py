import os
import streamlit as st
import pickle
import time
import langchain
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import WebBaseLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv() 

st.title("News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls=[]
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")
file_path = "D:/InsuranceRag/Equity/faiss_index/index.pkl"

main_placeholder = st.empty()


groq_api_key = os.getenv("GROQ_API_KEY", ".........")
llm = llm = ChatGroq(model="llama-3.1-8b-instant", groq_api_key=groq_api_key)

if process_url_clicked:
    #load data
    docs=[]
    for url in urls:
        if url.strip():
            loader = WebBaseLoader(url)
            main_placeholder.text("Data Loading...Started...✅✅✅")
            docs.extend(loader.load())
   
  

    #split data
    text_splitter = RecursiveCharacterTextSplitter(separators=['\n\n', '\n', '.', ','],chunk_size=800,chunk_overlap=200)
    main_placeholder.text("Text Splitting...Started...✅✅✅")
    data= text_splitter.split_documents(docs)

    #create embeddings and store in vector db
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore_faiss = FAISS.from_documents(data, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_faiss, f)

query = main_placeholder.text_input("Enter your query here: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore_faiss = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore_faiss.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)