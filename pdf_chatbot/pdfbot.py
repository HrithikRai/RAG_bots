import os
import shutil
import streamlit as st
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings, ChatCohere
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import (
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    ChatPromptTemplate,
)
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

def clean_text(text):
    return " ".join(text.split())

def setup_pdf_text_chatbot(api_key, file_paths):
    all_docs = []
    for file_path in file_paths:
        if file_path.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
        elif file_path.endswith(".txt"):
            loader = TextLoader(file_path)
        else:
            raise ValueError(f"Unsupported: {file_path}")
        docs = loader.load()
        for doc in docs:
            doc.page_content = clean_text(doc.page_content)
        all_docs.extend(docs)

    splitter = CharacterTextSplitter(separator=".", chunk_size=100, chunk_overlap=10)
    split_docs = splitter.split_documents(all_docs)

    embeddings = CohereEmbeddings(cohere_api_key=api_key, model="embed-english-v3.0")
    persist_directory = "./temp_vectorstore"
    if os.path.exists(persist_directory):
            shutil.rmtree(persist_directory)
    vectorstore = Chroma.from_documents(
        documents=split_docs, embedding=embeddings, persist_directory=persist_directory
    )

    retriever = vectorstore.as_retriever(
        search_type="mmr", search_kwargs={"k": 3, "lambda_mult": 0.3}
    )

    TEMPLATE = """
    You are a helpful chatbot that answers questions based on the content of uploaded PDF and text files.
    Answer the questions using only the following context:
    {context}
    """
    TEMPLATE_Q = "{question}"
    message_template_1 = SystemMessagePromptTemplate.from_template(template=TEMPLATE)
    message_template_2 = HumanMessagePromptTemplate.from_template(template=TEMPLATE_Q)
    chat_template = ChatPromptTemplate.from_messages([message_template_1, message_template_2])

    chat = ChatCohere(cohere_api_key=api_key)
    parser = StrOutputParser()
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | chat_template
        | chat
        | parser
    )

    return chain, persist_directory

st.title("PDF & Text File Chatbot")

api_key = st.text_input("Enter your Cohere API key", type="password")

uploaded_files = st.file_uploader(
    "Upload PDF and Text Files", type=["pdf", "txt"], accept_multiple_files=True
)

if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "persist_directory" not in st.session_state:
    st.session_state.persist_directory = None

if st.button("Set up Chatbot"):
    if not api_key or not uploaded_files:
        st.error("Please provide the API key and upload at least one file.")
    else:
        temp_dir = "./temp_uploads"
        os.makedirs(temp_dir, exist_ok=True)
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path = os.path.join(temp_dir, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.read())
            file_paths.append(file_path)
        try:
            chatbot, persist_directory = setup_pdf_text_chatbot(api_key, file_paths)
            st.session_state.chatbot = chatbot
            st.session_state.persist_directory = persist_directory
            st.success("Chatbot is ready! Start asking questions.")
        except Exception as e:
            st.error(f"Error setting up chatbot: {e}")

if st.session_state.chatbot:
    question = st.text_input("Ask your question:")
    if question:
        with st.spinner("Generating response..."):
            try:
                response = st.session_state.chatbot.invoke(question)
                st.markdown("### Response:")
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")


