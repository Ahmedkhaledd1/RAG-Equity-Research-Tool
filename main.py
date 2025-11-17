import streamlit as st
from langchain_classic.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
import pickle
from dotenv import load_dotenv


import os
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("API_KEY")
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

llm = ChatOpenAI(
    model="deepseek/deepseek-chat-v3.1:free",   
    temperature=0.5
)


st.title("News Research Tool 📋")
st.sidebar.title("News Article URLs")
urls=[]
for i in range(3):
    url= st.sidebar.text_input(f"Enter URL {i+1}")
    urls.append(url)


process_url_clicked = st.sidebar.button("Process URLs")

if process_url_clicked:
    placeholder = st.empty()
    placeholder.text("Processing URLs...✅")
    loader=UnstructuredURLLoader(urls=urls)
    documents=loader.load()

    placeholder.text("Text Splitter Started...✅")

    text_splitter=RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs=text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-large-en-v1.5",
    model_kwargs={"device": "cpu"}  # CPU only
    )
    placeholder.text("Embedding Vector Started Building...✅")
    vectorstore=FAISS.from_documents(docs, embeddings)

    with open('vectorindex.pkl', 'wb') as f:
        
        pickle.dump(vectorstore, f)


        with open('vectorindex.pkl', 'rb') as f:
            vectorstore = pickle.load(f)
        

query=st.text_input("Enter your question :")
if st.button("Get Answer"):
    if query:
        with open('vectorindex.pkl', 'rb') as f:
            vectorstore = pickle.load(f)
        
        docs=vectorstore.similarity_search(query, k=2)

        chain=load_qa_with_sources_chain(llm=llm, chain_type="map_reduce")

        response=chain.run(input_documents=docs, question=query, return_only_outputs=True)

        st.header("Answer:")
        st.write(response)   # this is the answer

        