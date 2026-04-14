import streamlit as st
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import SupabaseVectorStore
from supabase.client import Client, create_client

# -------------------------------
# 1️⃣ Set up Environment & Page
# -------------------------------
st.set_page_config(page_title="Groq + Supabase RAG", page_icon="🤖", layout="centered")

# Get API keys from Streamlit Secrets or Environment Variables
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY"))
supabase_url = st.secrets.get("SUPABASE_URL", os.getenv("SUPABASE_URL"))
supabase_key = st.secrets.get("SUPABASE_KEY", os.getenv("SUPABASE_KEY"))

if not groq_api_key or not supabase_url or not supabase_key:
    st.error("🚨 Missing API Keys. Please check your GROQ_API_KEY, SUPABASE_URL, and SUPABASE_KEY.")
    st.stop()

# -------------------------------
# 2️⃣ Initialize Models & DB Client
# -------------------------------
# -------------------------------
# 2️⃣ Initialize Models & DB Client (DEBUG MODE)
# -------------------------------
try:
    llm = ChatGroq(
        model_name="llama-3.3-70b-versatile", 
        groq_api_key=groq_api_key,
        streaming=True
    )
    # If it works, it will silently move on
except Exception as e:
    st.error(f"🚨 GROQ ERROR: {e}")
    st.stop()

try:
    supabase: Client = create_client(supabase_url, supabase_key)
    # If it works, it will silently move on
except Exception as e:
    st.error(f"🚨 SUPABASE ERROR: {e}")
    st.stop()

@st.cache_resource
def get_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create the vector store connection
vector_store = SupabaseVectorStore(
    client=supabase,
    embedding=get_embeddings(),
    table_name="documents",
    query_name="match_documents"
)

# -------------------------------
# 3️⃣ Chat History Management
# -------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]

st.title("🤖 Groq & Supabase Chatbot")
st.caption("Documents uploaded here are saved permanently to your Supabase Vector Database.")

# -------------------------------
# 4️⃣ Sidebar & File Uploading
# -------------------------------
with st.sidebar:
    st.header("Upload Document")
    uploaded_file = st.file_uploader("Upload a PDF to the database", type=["pdf"])
    
    if uploaded_file:
        # Check if we processed it in this session to avoid spamming the DB
        if st.session_state.get("uploaded_filename") != uploaded_file.name:
            with st.spinner("Uploading and indexing to Supabase..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    tmp_file_path = tmp_file.name

                loader = PyPDFLoader(tmp_file_path)
                docs = loader.load()

                text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                splits = text_splitter.split_documents(docs)

                # Push documents to Supabase (this saves them permanently)
                vector_store.add_documents(splits)

                st.session_state.uploaded_filename = uploaded_file.name
                os.remove(tmp_file_path) 
                
            st.success("✅ PDF saved to Supabase! You can now chat with it.")

    st.divider()
    if st.button("Clear Screen"):
        st.session_state.messages = [SystemMessage(content="You are a helpful assistant.")]
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
# 6️⃣ User Input & RAG Logic
# -------------------------------
# -------------------------------
# 6️⃣ User Input & RAG Logic
# -------------------------------
user_query = st.chat_input("Type your message...")

if user_query:
    st.session_state.messages.append(HumanMessage(content=user_query))
    st.chat_message("user").write(user_query)

    messages_for_llm = st.session_state.messages.copy()

    try:
        # 🚨 THE FIX: Bypass LangChain's broken search and query Supabase directly!
        embeddings = get_embeddings()
        query_vector = embeddings.embed_query(user_query)
        
        # Call the Supabase SQL function directly
        response = supabase.rpc(
            "match_documents", 
            {"query_embedding": query_vector, "match_count": 3}
        ).execute()
        
       # If the database returns matching context, inject it into the prompt
        if response.data:
            context = "\n\n".join([doc["content"] for doc in response.data])
            
           # Add this to see exactly what the database extracted!
           with st.expander("🔍 See exactly what text the database found"):
               st.write(context)

            rag_system_prompt = (
                "You are a helpful assistant. Use the following document context to answer the user's question. "
                "If the answer is not contained in the context, answer normally but clarify it isn't in the document.\n\n"
                f"Context:\n{context}"
            )
            messages_for_llm[0] = SystemMessage(content=rag_system_prompt)
            
    except Exception as e:
        st.error(f"Database search failed: {e}")

    # Generate assistant response
    with st.chat_message("assistant"):
        response_placeholder = st.empty()
        full_response = ""

        try:
            for chunk in llm.stream(messages_for_llm):
                full_response += chunk.content
                response_placeholder.markdown(full_response + "▌")
            
            response_placeholder.markdown(full_response)
            st.session_state.messages.append(AIMessage(content=full_response))
            
        except Exception as e:
            st.error(f"An error occurred: {e}")
