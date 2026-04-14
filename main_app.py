import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

# -------------------------------
# 1️⃣ Set up Environment & Page
# -------------------------------
st.set_page_config(page_title="Groq Doc Chatbot", page_icon="🤖", layout="centered")

# Get API key from Streamlit Secrets (for deployment) or Environment Variables (for local)
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))

if not groq_api_key:
    st.error("🚨 GROQ_API_KEY not found. Please set it in your environment variables or Streamlit secrets.")
    st.stop()

# -------------------------------
# 2️⃣ Initialize Models
# -------------------------------
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=groq_api_key,
        streaming=True
    )
except Exception as e:
    st.error(f"Failed to initialize Groq model: {e}")
    st.stop()

# Cache the embeddings model so it doesn't reload on every interaction
@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# -------------------------------
# 3️⃣ Chat History Management
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]

st.title("🤖 Groq Document Chatbot")

# -------------------------------
# 4️⃣ Sidebar & File Uploading
# -------------------------------
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to chat with it", type=["pdf"])
    
    # Process the uploaded file
    if uploaded_file:
        # Check if we have already processed this exact file
        if "vector_store" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
            with st.spinner("Reading and indexing PDF..."):
                # Save uploaded file temporarily to disk
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                # Load and split the document
                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # Create the vector store
                embeddings = get_embeddings()
                vector_store = FAISS.from_documents(splits, embeddings)

                # Save to session state
                st.session_state.vector_store = vector_store
                st.session_state.uploaded_filename = uploaded_file.name
                
                # Cleanup temp file
                os.remove(tmp_file_path) 
                
            st.success("✅ PDF processed! Ask me questions about it.")

    st.divider()
    if st.button("Clear Conversation"):
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
        # Optionally clear vector store if you want a hard reset:
        # st.session_state.pop("vector_store", None)
        # st.session_state.pop("uploaded_filename", None)
        st.rerun()

# -------------------------------
# 5️⃣ Display Chat History
# -------------------------------
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)
    elif isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)

# -------------------------------
# 6️⃣ User Input & Streaming
# -------------------------------
user_query = st.chat_input("Type your message...")

if user_query:
    # Show user message
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").write(user_query)

    # Prepare messages for LLM
    messages_for_llm = st.session_state.messages.copy()

    # If a document is uploaded, inject relevant context into the System Prompt
    if "vector_store" in st.session_state:
        # Search the vector database for text matching the user query
        relevant_docs = st.session_state.vector_store.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Modify the system message dynamically for this specific turn
        rag_system_prompt = (
            "You are a helpful assistant. Use the following document context to answer the user's question. "
            "If the answer is not contained in the context, just say you don't know based on the document.\n\n"
            f"Context:\n{context}"
        )
        messages_for_llm[0] = SystemMessage(content=rag_system_prompt)

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            # Stream the response using the dynamically built messages
            for chunk in llm.stream(messages_for_llm):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            # Final update without the cursor
            response_placeholder.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
