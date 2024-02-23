import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI

import os
import sys
# Assume PyPDFLoader, Docx2txtLoader, and TextLoader are defined elsewhere or replace them with actual implementations


os.environ["OPENAI_API_KEY"]="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"

def process_pdf(file_path):
    reader = PdfReader(file_path)
    raw_text = ''
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            raw_text += text

    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=800,
        chunk_overlap=50,
        length_function=len,
    )
    texts = text_splitter.split_text(raw_text)
    embeddings = OpenAIEmbeddings()
    vectordb= FAISS.from_texts(texts, embeddings)

    return vectordb



from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


def setup_conversational_model(vectordb):
    llm = ChatOpenAI(temperature=0.7, model_name='gpt-3.5-turbo')
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectordb.as_retriever(search_kwargs={'k': 6}),
        verbose=False, memory=memory
    )
    return pdf_qa

import streamlit as st


def main():
    st.title("PDF Conversational Interface")

    # Initialize conversation and input_text in session_state if not already present
    if 'conversation' not in st.session_state:
        st.session_state.conversation = []
    if 'input_text' not in st.session_state:
        st.session_state.input_text = ""  # This will hold the text input

    uploaded_file = st.file_uploader("Choose a PDF file", type=['pdf'])
    if uploaded_file is not None and 'pdf_qa' not in st.session_state:
        vectordb = process_pdf(uploaded_file)
        st.session_state.pdf_qa = setup_conversational_model(vectordb)

    with st.form(key='question_form'):
        question = st.text_input("Ask a question about the PDF content:", value=st.session_state.input_text, key="question_input")
        submit_button = st.form_submit_button("Submit")

    if submit_button and question:
        # Process the question using the conversational model
        full_response = st.session_state.pdf_qa({"question": question})
        current_answer = full_response.get('answer', "Sorry, I couldn't find an answer.")
        st.session_state.conversation.insert(0, ("You", question))  # Insert at the beginning
        st.session_state.conversation.insert(1, ("AI", current_answer))  # Ensures it follows the question
        st.session_state.input_text = ""  # Clear the input field for the next question
    else:
        st.session_state.input_text = question  # Preserve current input across reruns

    # Display the conversation history using Streamlit's layout primitives for a chat-like interface
    for author, text in st.session_state.conversation:
        if author == "You":
            st.container().markdown(f"**You**: {text}")
        else:  # AI's response
            st.container().markdown(f"**AI**: {text}")

if __name__ == "__main__":
    main()
