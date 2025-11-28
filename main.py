import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
import tempfile
import os

st.title("📘 Company Knowledge Base – AI Assistant")
st.write("Upload documents and ask questions!")

api_key = st.text_input("Enter your OpenAI API Key:", type="password")

uploaded_files = st.file_uploader("Upload PDF/TXT files", accept_multiple_files=True)

if uploaded_files and api_key:
    temp_dir = tempfile.mkdtemp()
    docs = []

    # Save uploaded files
    for file in uploaded_files:
        file_path = os.path.join(temp_dir, file.name)
        with open(file_path, "wb") as f:
            f.write(file.read())
        docs.append(file_path)

    # Load & split documents
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    texts = []

    for path in docs:
        with open(path, "r", errors="ignore") as f:
            raw_text = f.read()
        texts.extend(splitter.split_text(raw_text))

    embeddings = OpenAIEmbeddings(openai_api_key=api_key)

    vectordb = Chroma.from_texts(texts, embeddings)
    retriever = vectordb.as_retriever()

    llm = OpenAI(openai_api_key=api_key)

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    st.success("Documents processed! Ask your questions below.")

    query = st.text_input("Ask a question:")
    if query:
        result = qa_chain({"query": query})
        st.write("### Answer:")
        st.write(result["result"])
