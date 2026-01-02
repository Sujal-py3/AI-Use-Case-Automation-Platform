import streamlit as st
import os
import time
import shutil
import sys
from dotenv import load_dotenv

# Fix path to allow imports from root dir
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import our modules
from src.ingestion import DocumentProcessor
from src.retrieval import Retriever
from src.generation import UseCaseGenerator
from src.utils import setup_logger

# Load env vars
load_dotenv()
logger = setup_logger("app")

# Page Config
st.set_page_config(
    page_title="RAG Use Case Generator",
    page_icon="🤖",
    layout="wide"
)

# --- SESSION STATE INITIALIZATION ---
if "processor" not in st.session_state:
    st.session_state.processor = DocumentProcessor()

if "vectors_ready" not in st.session_state:
    st.session_state.vectors_ready = False

if "all_documents" not in st.session_state:
    st.session_state.all_documents = []    

if "retriever" not in st.session_state:
    st.session_state.retriever = None

if "generator" not in st.session_state:
    st.session_state.generator = UseCaseGenerator()

if "messages" not in st.session_state:
    st.session_state.messages = []

# --- SIDEBAR (Ingestion) ---
with st.sidebar:
    st.header("🔑 Configuration")
    api_key_input = st.text_input("Groq API Key", type="password", help="Enter your Groq API Key if not set in .env")
    
    # Update generator if key is provided
    if api_key_input and (st.session_state.generator.llm is None or st.session_state.generator.api_key != api_key_input):
         st.session_state.generator = UseCaseGenerator(api_key=api_key_input)
         st.success("API Key updated!")

    st.header("📂 Knowledge Base")
    
    # OPTION 1: Automatic Local Dataset Ingestion
    dataset_path = "Dataset"
    if os.path.exists(dataset_path):
        st.subheader("Local Dataset Found")
        if st.button("📥 Ingest 'Dataset' Folder"):
            with st.spinner("Scanning and Ingesting 'Dataset' folder..."):
                all_docs = []
                file_count = 0
                
                # Walk through directory
                progress_bar = st.progress(0)
                files_to_process = []
                
                for root, dirs, files in os.walk(dataset_path):
                    for file in files:
                        if file.lower().endswith(('.pdf', '.txt', '.md', '.png', '.jpg', '.jpeg', '.docx')):
                            files_to_process.append(os.path.join(root, file))
                
                if not files_to_process:
                    st.warning("No supported files found in Dataset folder.")
                else:
                    for i, file_path in enumerate(files_to_process):
                        try:
                            docs = st.session_state.processor.process_file(file_path)
                            all_docs.extend(docs)
                            file_count += 1
                        except Exception as e:
                            logger.error(f"Failed to process {file_path}: {e}")
                        
                        progress_bar.progress((i + 1) / len(files_to_process))
                    
                    if all_docs:
                        st.session_state.all_documents.extend(all_docs)
                        
                        # Re-create DB with new docs (or append? sticking to create for now)
                        st.status("Updating Vector Store...")
                        vectordb = st.session_state.processor.create_vector_db(st.session_state.all_documents)
                        st.session_state.retriever = Retriever(vectordb, st.session_state.all_documents)
                        st.session_state.vectors_ready = True
                        st.success(f"✅ Ingested {file_count} files ({len(all_docs)} chunks) from 'Dataset'!")

    st.write("--- OR ---")
    st.write("Upload Manual Files:")
    
    uploaded_files = st.file_uploader(
        "Drop files here",
        accept_multiple_files=True,
        type=["pdf", "txt", "md", "png", "jpg", "jpeg"]
    )
    
    process_btn = st.button("Process Files", type="primary")
    
    if process_btn and uploaded_files:
        with st.spinner("Ingesting and Chunking..."):
            # Create a temp dir to save uploads so our processor can read them
            temp_dir = "temp_uploads"
            if not os.path.exists(temp_dir):
                os.makedirs(temp_dir)
                
            all_docs = []
            
            progress_bar = st.progress(0)
            for i, uploaded_file in enumerate(uploaded_files):
                # Save to disk
                file_path = os.path.join(temp_dir, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Process
                docs = st.session_state.processor.process_file(file_path)
                all_docs.extend(docs)
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Clean up temp dir
            # shutil.rmtree(temp_dir) # logic comment: maybe keep for debug? na, delete clean
            try:
                shutil.rmtree(temp_dir)
            except:
                pass

            if all_docs:
                st.session_state.all_documents = all_docs
                
                # Create Vector DB
                st.status("Creating Vector Store...")
                vectordb = st.session_state.processor.create_vector_db(all_docs)
                
                # Initialize Retriever (Hybrid)
                st.status("Initializing Hybrid Search...")
                st.session_state.retriever = Retriever(vectordb, all_docs)
                
                st.session_state.vectors_ready = True
                st.success(f"✅ Indexed {len(all_docs)} chunks!")
            else:
                st.error("No valid text extracted from files.")

    # Configuration for User
    st.divider()
    st.subheader("⚙️ Settings")
    top_k = st.slider("Retrieval Count (Top K)", 1, 10, 5)

# --- MAIN CHAT AREA ---
st.title("🤖 Use Case Generator")
st.markdown("Ask me to generate use cases based on your uploaded docs.")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "json_data" in message:
            with st.expander("View JSON Output"):
                st.json(message["json_data"])

# Chat Input
if prompt := st.chat_input("Ex: 'Create use cases for Signup'"):
    # Add user message to state
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Check if we are ready
    if not st.session_state.vectors_ready:
        response = "⚠️ Please upload and process documents first."
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})
    else:
        with st.chat_message("assistant"):
            with st.spinner("🔍 Retrieving context & Generating..."):
                # 1. Retrieve
                retrieved_docs = st.session_state.retriever.query(prompt, top_k=top_k)
                
                # 2. Generate
                result = st.session_state.generator.generate(prompt, retrieved_docs)
                
                # 3. Display
                if result.get("insufficient_context"):
                    st.warning("⚠️ Insufficient Context to answer fully.")
                    if result.get("clarifications_needed"):
                        st.write("**Clarifications Needed:**")
                        for q in result["clarifications_needed"]:
                            st.write(f"- {q}")
                
                st.json(result) # Show the main JSON result
                
                # 4. Debug / Observability Section
                with st.expander("🛠️ Debug Information (Retrieval & Sources)"):
                    st.write(f"**Retrieved {len(retrieved_docs)} chunks:**")
                    for i, doc in enumerate(retrieved_docs):
                        st.markdown(f"**Chunk {i+1}** (Source: `{doc.metadata.get('source')}`, Page: `{doc.metadata.get('page_number')}`)")
                        st.text(doc.page_content[:300] + "...") # Preview
                        st.divider()
                
                # Add to history (we just store a summary string for the text part)
                st.session_state.messages.append({
                    "role": "assistant", 
                    "content": "Here are the generated use cases:",
                    "json_data": result
                })
