import streamlit as st
import os
import shutil
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# --- PAGE SETUP ---
st.set_page_config(page_title="AI Data Infrastructure", layout="wide")
st.title("Vector DB & AI Infrastructure Demo")
st.markdown("### Using Experimental Gemini 3.x Infrastructure")

# --- SIDEBAR ---
with st.sidebar:
    st.header("1. Infrastructure Config")
    google_api_key = st.text_input("Enter Google API Key", type="password")
    
    st.divider()
    if st.button("Reset Vector Store"):
        if os.path.exists("./chroma_db"):
            shutil.rmtree("./chroma_db")
            st.rerun()

# --- FILE UPLOADER ---
uploaded_file = st.file_uploader("2. Upload Research PDF", type="pdf")

if uploaded_file and google_api_key:
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    # --- PIPELINE (CACHED) ---
    @st.cache_resource
    def build_vector_db(file_path):
        loader = PyPDFLoader(file_path)
        data = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_documents(data)
        
        # LOCAL CPU EMBEDDINGS (Trend: Edge Privacy)
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        vector_db = Chroma.from_documents(chunks, embeddings, persist_directory="./chroma_db")
        return vector_db

    with st.spinner("🔧 Engineering Vector Infrastructure..."):
        db = build_vector_db("temp.pdf")
        st.success("Local Vector Database Online")

    # --- QUERYING ---
    query = st.text_input("3. Ask a question about the document:")

    if query:
        # RETRIEVAL
        docs = db.similarity_search(query, k=10)
        context = "\n\n".join([d.page_content for d in docs])

        # GENERATION (The "Gemini 3.x" Model Hunter)
        model_options = [
            "gemini-3.5-flash",
            "gemini-3.1-pro",
            "gemini-3-flash",
            "gemini-3.1-flash-lite",
            "gemini-2.0-flash-exp" # Backup
        ]
        
        response_final = None
        error_logs = []

        for model_name in model_options:
            try:
                llm = ChatGoogleGenerativeAI(model=model_name, google_api_key=google_api_key)
                prompt = f"Use this context to answer: {context}\n\nQuestion: {query}"
                
                with st.spinner(f"Requesting {model_name}..."):
                    res = llm.invoke(prompt)
                    response_final = res.content
                    model_used = model_name
                    break
            except Exception as e:
                error_logs.append(f" {model_name}: {str(e)[:150]}...")
                continue

        if response_final:
            st.subheader(f"Answer (via {model_used}):")
            st.write(response_final)
            
            with st.expander("View Vector Retrieval Logic"):
                for i, d in enumerate(docs):
                    st.info(f"**Chunk {i+1}:** {d.page_content[:500]}...")
        else:
            st.error(" All models failed to connect.")
            with st.expander("Show Technical Debugger Logs"):
                for log in error_logs:
                    st.write(log)

elif not google_api_key:
    st.warning("Please enter your API Key in the sidebar.")