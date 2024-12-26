import streamlit as st
from langchain_cohere import ChatCohere
from langchain_community.document_loaders import YoutubeLoader
from langchain_text_splitters.character import CharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import PromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="YouTube Chatbot", layout="wide")
st.title("🎥 YouTube Video Chatbot")

with st.sidebar:
    st.header("Configuration")
    cohere_api_key = st.text_input("Enter Cohere API Key", type="password")
    youtube_url = st.text_input("Enter YouTube Video URL")

def setup_chatbot(api_key, video_url):
    loader = YoutubeLoader.from_youtube_url(video_url, add_video_info=False)
    transcript = loader.load()[0].page_content.replace(u'\xa0', u'').replace(u'\uf0a7', u'')
    chat = ChatCohere(cohere_api_key=api_key)
    char_splitter = CharacterTextSplitter(separator=".", chunk_size=500, chunk_overlap=0)
    transcripts_split = char_splitter.split_text(transcript)
    embeddings = CohereEmbeddings(cohere_api_key=api_key, model="embed-english-v3.0")
    vectorstore = Chroma.from_texts(texts=transcripts_split, embedding=embeddings, persist_directory='./python-projects')
    retriever = vectorstore.as_retriever(search_type='mmr', search_kwargs={'k': 3, 'lambda_mult': 0.3})
    TEMPLATE = 'You are a helpful chatbot that answers questions on YouTube videos.\nAnswer the questions using only the following context:\n{context}'
    TEMPLATE_Q = '{question}'
    message_template_1 = SystemMessagePromptTemplate.from_template(template=TEMPLATE)
    message_template_2 = HumanMessagePromptTemplate.from_template(template=TEMPLATE_Q)
    chat_template = ChatPromptTemplate.from_messages([message_template_1, message_template_2])
    parser = StrOutputParser()
    chain = ({'context': retriever, 'question': RunnablePassthrough()} | chat_template | chat | parser)
    return chain

if cohere_api_key and youtube_url:
    chain = setup_chatbot(cohere_api_key, youtube_url)
    st.success("Chatbot setup complete! You can now ask questions below.")
    question = st.text_input("Ask a question about the video:")
    if question:
        with st.spinner("Generating response..."):
            try:
                response = chain.invoke(question)
                st.markdown("### Response:")
                st.success(response)
            except Exception as e:
                st.error(f"An error occurred: {e}")
else:
    st.warning("Please enter both the API Key and YouTube video URL in the sidebar.")
