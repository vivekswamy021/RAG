import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# 1️⃣ Environment Setup
groq_api_key = os.getenv("GROQ_API_KEY")

st.set_page_config(page_title="Groq RAG Chatbot", page_icon="📚", layout="wide")

if not groq_api_key:
    st.error("🚨 GROQ_API_KEY not found.")
    st.stop()

# 2️⃣ Initialize Model
llm = ChatGroq(model_name="llama-3.3-70b-versatile", groq_api_key=groq_api_key)

# 3️⃣ RAG Logic: Processing the PDF
def process_pdf(uploaded_file):
    # Save temporary file to disk for PyPDFLoader
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()
    
    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    final_documents = text_splitter.split_documents(docs)
    
    # Create Embeddings and Vector Store (Local via FAISS)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(final_documents, embeddings)
    return vectorstore

# 4️⃣ UI Layout
st.title("📚 Groq RAG: Chat with your PDFs")

with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    
    if st.button("Clear Chat"):
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
        st.rerun()

# Initialize session states
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="Use the provided context to answer the user's question.")]

if uploaded_file and "vectorstore" not in st.session_state:
    with st.spinner("Processing PDF..."):
        st.session_state.vectorstore = process_pdf(uploaded_file)
        st.success("PDF Indexed!")

# 5️⃣ Display History
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# 6️⃣ User Input & RAG Execution
user_query = st.chat_input("Ask something about the document...")

if user_query:
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").write(user_query)

    with st.chat_message("assistant"):
        if "vectorstore" in st.session_state:
            # Retrieval Step
            retriever = st.session_state.vectorstore.as_retriever()
            context_docs = retriever.invoke(user_query)
            
            # Create the prompt with context
            prompt = ChatPromptTemplate.from_template("""
            Answer the following question based only on the provided context:
            <context>
            {context}
            </context>
            Question: {input}""")
            
            document_chain = create_stuff_documents_chain(llm, prompt)
            
            # Generate Response
            response = document_chain.invoke({
                "input": user_query,
                "context": context_docs
            })
            
            st.write(response)
            st.session_state.messages.append(AIMessage(content=response))
        else:
            st.warning("Please upload a PDF first to use the RAG features.")
