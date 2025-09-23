"""
🔍 AI Document Summarizer - Streamlit Application

WHAT THIS APP DOES TO GET AI SUMMARY:
===================================

1. INPUT PROCESSING:
   - Accepts 3 input types: PDF documents, Website URLs, YouTube videos
   - Extracts raw text content from each source type
   - Validates and preprocesses the input data

2. TEXT EXTRACTION & LOADING:
   - PDF: Uses PyPDFLoader to extract text from uploaded PDF files
   - Website: Uses UnstructuredURLLoader to scrape and clean web content  
   - YouTube: Uses YouTubeTranscriptApi to fetch video transcripts. For fallback, we use from langchain_community.document_loaders import YoutubeLoader
                loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)

3. DOCUMENT CHUNKING:
   - Splits large documents into smaller chunks (4000 chars with 200 overlap)
   - Ensures content fits within LLM token limits
   - Maintains context between chunks

4. AI MODEL SELECTION:
   - Provides 8+ Groq models (Llama, GPT-OSS, Qwen, etc.)
   - Different models for speed vs quality trade-offs
   - Models range from 8B to 120B parameters

5. LLM SUMMARIZATION PROCESS:
   - Uses LangChain's load_summarize_chain with "stuff" method
   - Feeds processed documents to selected Groq model
   - For large docs: Hierarchical summarization (chunk -> combine)
   - For small docs: Direct single-pass summarization

6. OUTPUT GENERATION:
   - Receives summarized text from LLM
   - Displays results with analytics (word count, reading time)
   - Provides download and copy functionality

CORE AI WORKFLOW:
Raw Input → Text Extraction → Chunking → LLM Processing → Summary Output
"""


import streamlit as st
import validators
from langchain.chains.summarize import load_summarize_chain
from langchain_community.document_loaders import UnstructuredURLLoader, PyPDFLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_groq import ChatGroq
import tempfile
import os

# Try to import YouTube functionality
try:
    from youtube_transcript_api import YouTubeTranscriptApi
    YOUTUBE_AVAILABLE = True
except ImportError:
    YOUTUBE_AVAILABLE = False

# Load .env if exists
from dotenv import load_dotenv
load_dotenv()

# ----------------------------
# Load GROQ API Key
# Priority: Streamlit secrets > Environment variable
# ----------------------------
groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))

if not groq_api_key:
    st.error("❌ GROQ_API_KEY is missing! Please set it in Streamlit secrets or your environment.")
    st.stop()

# Ensure environment variable is set for libraries that rely on it
os.environ["GROQ_API_KEY"] = groq_api_key


# Page configuration with custom theme
st.set_page_config(
    page_title="🔍 AI Document Summarizer",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Groq Model Configuration
GROQ_MODELS = {
    # Production Models (Recommended)
    "🚀 Llama 3.1 8B (Fast)": {
        "id": "llama-3.1-8b-instant", 
        "context": "131K", 
        "description": "Fast and efficient, great for quick summaries",
        "category": "production",
        "speed": "⚡ Ultra-fast",
        "cost": "💰 Low"
    },
    "🧠 Llama 3.3 70B (Smart)": {
        "id": "llama-3.3-70b-versatile", 
        "context": "131K", 
        "description": "Larger model for better quality and reasoning",
        "category": "production",
        "speed": "⚡ Fast",
        "cost": "💰 Medium"
    },
    "🔥 GPT-OSS 120B (Premium)": {
        "id": "openai/gpt-oss-120b", 
        "context": "131K", 
        "description": "OpenAI's flagship open model with reasoning",
        "category": "production",
        "speed": "⚡ Fast",
        "cost": "💰 High"
    },
    "⭐ GPT-OSS 20B (Balanced)": {
        "id": "openai/gpt-oss-20b", 
        "context": "131K", 
        "description": "Balanced performance and efficiency",
        "category": "production",
        "speed": "⚡ Very Fast",
        "cost": "💰 Medium"
    },
    
    # Preview Models (Advanced)
    "🔬 Llama 4 Maverick 17B": {
        "id": "meta-llama/llama-4-maverick-17b-128e-instruct", 
        "context": "131K", 
        "description": "Experimental Llama 4 preview model",
        "category": "preview",
        "speed": "⚡ Fast",
        "cost": "💰 Medium"
    },
    "🛡️ Llama 4 Scout 17B": {
        "id": "meta-llama/llama-4-scout-17b-16e-instruct", 
        "context": "131K", 
        "description": "Advanced reasoning and analysis",
        "category": "preview",
        "speed": "⚡ Fast",
        "cost": "💰 Medium"
    },
    "🌙 Kimi K2 Instruct": {
        "id": "moonshotai/kimi-k2-instruct-0905", 
        "context": "262K", 
        "description": "Ultra-long context window for large documents",
        "category": "preview",
        "speed": "⚡ Medium",
        "cost": "💰 High"
    },
    "🤖 Qwen3 32B": {
        "id": "qwen/qwen3-32b", 
        "context": "131K", 
        "description": "Alibaba's advanced multilingual model",
        "category": "preview",
        "speed": "⚡ Fast",
        "cost": "💰 Medium"
    }
}

# Enhanced Custom CSS for better card-based radio selection and wider sidebar
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Hide default streamlit styling */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* WIDER SIDEBAR - Key fix here */
    section[data-testid="stSidebar"] {
        width: 420px !important;
        min-width: 420px !important;
    }
    
    section[data-testid="stSidebar"] > div:first-child {
        width: 420px !important;
        min-width: 420px !important;
    }
    
    section[data-testid="stSidebar"][aria-expanded="true"] > div:first-child {
        width: 420px !important;
        margin-left: 0px !important;
    }
    
    section[data-testid="stSidebar"][aria-expanded="false"] > div:first-child {
        width: 420px !important;
        margin-left: -420px !important;
    }
    
    /* Adjust main content area to account for wider sidebar */
    .main .block-container {
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        max-width: none !important;
    }
    
    /* Main app styling */
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2.5rem 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
        box-shadow: 0 10px 40px rgba(31, 38, 135, 0.4);
        backdrop-filter: blur(10px);
    }
    
    .main-header h1 {
        font-family: 'Poppins', sans-serif;
        font-weight: 700;
        font-size: 2.8rem;
        margin: 0;
        text-shadow: 2px 2px 8px rgba(0,0,0,0.3);
        letter-spacing: -0.5px;
    }
    
    .main-header p {
        font-family: 'Inter', sans-serif;
        font-weight: 400;
        font-size: 1.3rem;
        margin: 1rem 0 0 0;
        opacity: 0.95;
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }
    
    /* Step sections with enhanced styling */
    .step-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        border-radius: 16px;
        padding: 2rem;
        margin: 2rem 0;
        border: 1px solid #e8ecf7;
        box-shadow: 0 8px 32px rgba(102, 126, 234, 0.08);
        position: relative;
        overflow: hidden;
    }
    
    .step-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #667eea, #764ba2);
    }
    
    [data-theme="dark"] .step-container {
        background: linear-gradient(145deg, #1a1a1a 0%, #2d2d2d 100%);
        border-color: #404040;
    }
    
    .step-header {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.4rem;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        color: #2c3e50;
    }
    
    [data-theme="dark"] .step-header {
        color: #ecf0f1;
    }
    
    /* Enhanced Method Selection Cards */
    .method-selection-container {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
        gap: 1.5rem;
        margin: 2rem 0;
    }
    
    .method-card {
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        border-radius: 16px;
        padding: 1.8rem;
        border: 2px solid #e8ecf7;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        cursor: pointer;
        position: relative;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
    }
    
    .method-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 12px 40px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
    }
    
    .method-card.selected {
        border-color: #667eea;
        background: linear-gradient(145deg, #f0f3ff 0%, #ffffff 100%);
        transform: translateY(-4px);
        box-shadow: 0 8px 30px rgba(102, 126, 234, 0.25);
    }
    
    [data-theme="dark"] .method-card {
        background: linear-gradient(145deg, #2d2d2d 0%, #1a1a1a 100%);
        border-color: #404040;
    }
    
    [data-theme="dark"] .method-card:hover {
        border-color: #8b9aff;
        box-shadow: 0 12px 40px rgba(139, 154, 255, 0.15);
    }
    
    .method-icon {
        font-size: 3rem;
        margin-bottom: 1rem;
        display: block;
        text-align: center;
        filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1));
    }
    
    .method-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.3rem;
        color: #2c3e50;
        margin-bottom: 0.8rem;
        text-align: center;
    }
    
    [data-theme="dark"] .method-title {
        color: #ecf0f1;
    }
    
    .method-description {
        font-family: 'Inter', sans-serif;
        font-size: 0.95rem;
        color: #64748b;
        line-height: 1.6;
        text-align: center;
        margin: 0;
    }
    
    [data-theme="dark"] .method-description {
        color: #94a3b8;
    }
    
    .method-features {
        margin-top: 1rem;
        padding-top: 1rem;
        border-top: 1px solid #e8ecf7;
    }
    
    [data-theme="dark"] .method-features {
        border-top-color: #404040;
    }
    
    .feature-item {
        font-family: 'Inter', sans-serif;
        font-size: 0.85rem;
        color: #667eea;
        margin: 0.3rem 0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        justify-content: center;
    }
    
    [data-theme="dark"] .feature-item {
        color: #8b9aff;
    }
    
    /* Hide default radio buttons completely */
    .stRadio {
        display: none !important;
    }
    
    /* Model Selection Card Styling */
    .model-card {
        background: linear-gradient(145deg, #f8faff, #ffffff);
        border: 2px solid #e8ecf7;
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        transition: all 0.3s ease;
        cursor: pointer;
    }
    
    .model-card:hover {
        border-color: #667eea;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.15);
    }
    
    .model-card.selected {
        border-color: #667eea;
        background: linear-gradient(145deg, #f0f3ff, #ffffff);
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.2);
    }
    
    [data-theme="dark"] .model-card {
        background: linear-gradient(145deg, #2d2d2d, #1a1a1a);
        border-color: #404040;
    }
    
    [data-theme="dark"] .model-card:hover {
        border-color: #8b9aff;
    }
    
    /* Enhanced Input Sections */
    .input-section {
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        border-radius: 12px;
        padding: 2rem;
        margin: 1.5rem 0;
        border: 1px solid #e8ecf7;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.06);
    }
    
    [data-theme="dark"] .input-section {
        background: linear-gradient(145deg, #2d2d2d 0%, #1a1a1a 100%);
        border-color: #404040;
    }
    
    .input-title {
        font-family: 'Poppins', sans-serif;
        font-weight: 600;
        font-size: 1.2rem;
        color: #2c3e50;
        margin-bottom: 1.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    [data-theme="dark"] .input-title {
        color: #ecf0f1;
    }
    
    /* Enhanced Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        font-size: 1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.3);
        width: 100%;
        min-height: 48px;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
        background: linear-gradient(135deg, #5a72e8 0%, #6d42a0 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* Primary button variant */
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        box-shadow: 0 6px 20px rgba(16, 185, 129, 0.3);
        font-weight: 600;
        font-size: 1.1rem;
        padding: 1rem 2rem;
    }
    
    .stButton > button[kind="primary"]:hover {
        background: linear-gradient(135deg, #059669 0%, #047857 100%);
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.4);
    }
    
    /* Status messages with better styling */
    .status-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 15px rgba(16, 185, 129, 0.2);
    }
    
    .status-error {
        background: linear-gradient(135deg, #ef4444, #dc2626);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 15px rgba(239, 68, 68, 0.2);
    }
    
    .status-info {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 15px rgba(59, 130, 246, 0.2);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        font-family: 'Inter', sans-serif;
        font-weight: 500;
        display: flex;
        align-items: center;
        gap: 0.75rem;
        box-shadow: 0 4px 15px rgba(245, 158, 11, 0.2);
    }
    
    /* File uploader enhancement */
    .stFileUploader > div {
        background: linear-gradient(145deg, #f8faff, #ffffff);
        border: 3px dashed #667eea;
        border-radius: 16px;
        padding: 3rem 2rem;
        text-align: center;
        transition: all 0.3s ease;
        position: relative;
        overflow: hidden;
    }
    
    .stFileUploader > div:hover {
        border-color: #764ba2;
        background: linear-gradient(145deg, #f0f3ff, #f8faff);
        transform: scale(1.02);
    }
    
    /* Text input enhancement */
    /* ULTIMATE INPUT TEXT COLOR FIX - Solves white-on-white issue */
    .stTextInput > div > div > input,
    .stTextInput input,
    input[type="text"],
    input[type="url"],
    input[type="email"] {
        color: #1a1a1a !important; /* Very dark text for maximum visibility */
        background-color: #ffffff !important; /* Pure white background */
        -webkit-text-fill-color: #1a1a1a !important; /* Safari/mobile fix */
        caret-color: #1a1a1a !important; /* Dark cursor */
        border: 2px solid #e8ecf7 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
    }

    /* Text areas */
    .stTextArea > div > div > textarea,
    .stTextArea textarea {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #1a1a1a !important;
        caret-color: #1a1a1a !important;
        border: 2px solid #e8ecf7 !important;
        border-radius: 12px !important;
        padding: 0.75rem 1.25rem !important;
        font-size: 1rem !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Focus states */
    .stTextInput > div > div > input:focus,
    .stTextInput input:focus,
    input[type="text"]:focus,
    input[type="url"]:focus,
    input[type="email"]:focus {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #1a1a1a !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
    }

    .stTextArea > div > div > textarea:focus,
    .stTextArea textarea:focus {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #1a1a1a !important;
        border-color: #667eea !important;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1) !important;
    }

    /* Dark mode force overrides */
    [data-theme="dark"] .stTextInput > div > div > input,
    [data-theme="dark"] .stTextInput input,
    [data-theme="dark"] input[type="text"],
    [data-theme="dark"] input[type="url"],
    [data-theme="dark"] input[type="email"],
    [data-theme="dark"] .stTextArea > div > div > textarea,
    [data-theme="dark"] .stTextArea textarea {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #1a1a1a !important;
        caret-color: #1a1a1a !important;
    }

    /* Media query for dark mode preference */
    @media (prefers-color-scheme: dark) {
        .stTextInput > div > div > input,
        .stTextInput input,
        input[type="text"],
        input[type="url"],
        input[type="email"],
        .stTextArea > div > div > textarea,
        .stTextArea textarea {
            color: #1a1a1a !important;
            background-color: #ffffff !important;
            -webkit-text-fill-color: #1a1a1a !important;
            caret-color: #1a1a1a !important;
        }
    }

    /* Placeholder styling */
    .stTextInput > div > div > input::placeholder,
    .stTextInput input::placeholder,
    input[type="text"]::placeholder,
    input[type="url"]::placeholder,
    input[type="email"]::placeholder,
    .stTextArea > div > div > textarea::placeholder,
    .stTextArea textarea::placeholder {
        color: #64748b !important;
        opacity: 0.7 !important;
        -webkit-text-fill-color: #64748b !important;
    }

    /* Extra targeting for stubborn cases */
    div[data-testid="stTextInput"] input,
    div[data-testid="stTextArea"] textarea {
        color: #1a1a1a !important;
        background-color: #ffffff !important;
        -webkit-text-fill-color: #1a1a1a !important;
    }

    /* Mobile responsive */
    @media screen and (max-width: 768px) {
        .stTextInput > div > div > input,
        .stTextInput input,
        input[type="text"],
        input[type="url"],
        input[type="email"],
        .stTextArea > div > div > textarea,
        .stTextArea textarea {
            color: #1a1a1a !important;
            background-color: #ffffff !important;
            -webkit-text-fill-color: #1a1a1a !important;
            font-size: 16px !important; /* Prevent mobile zoom */
        }
    }
    
    /*     /* Selectbox enhancement */
    .stSelectbox > div > div > div {
        border-radius: 12px;
        border: 2px solid #e8ecf7;
        background: linear-gradient(145deg, #ffffff, #f8faff);
    }
    
    .stSelectbox > div > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
    }
    
    /* Summary result styling */
    .summary-container {
        background: linear-gradient(145deg, #ffffff 0%, #f8faff 100%);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        border: 1px solid #e8ecf7;
        box-shadow: 0 12px 40px rgba(0,0,0,0.1);
        position: relative;
        overflow: hidden;
    }
    
    .summary-container::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 4px;
        background: linear-gradient(90deg, #10b981, #059669);
    }
    
    [data-theme="dark"] .summary-container {
        background: linear-gradient(145deg, #2d2d2d 0%, #1a1a1a 100%);
        border-color: #404040;
    }
    
    .summary-text {
        font-family: 'Inter', sans-serif;
        font-size: 1.1rem;
        line-height: 1.8;
        color: #2c3e50;
        background: linear-gradient(145deg, #f8faff, #ffffff);
        padding: 2rem;
        border-radius: 16px;
        border: 1px solid #e8ecf7;
        box-shadow: inset 0 2px 8px rgba(0,0,0,0.04);
    }
    
    [data-theme="dark"] .summary-text {
        color: #ecf0f1;
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border-color: #404040;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.25);
        font-family: 'Inter', sans-serif;
        transition: transform 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-4px);
    }
    
    /* Sidebar specific styling */
    .sidebar-header {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.3);
        text-align: center;
    }
    
    .sidebar-status {
        padding: 1rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        text-align: center;
        font-weight: 500;
        font-family: 'Inter', sans-serif;
    }
    
    /* Responsive design */
    @media (max-width: 768px) {
        section[data-testid="stSidebar"] {
            width: 320px !important;
            min-width: 320px !important;
        }
        
        .method-selection-container {
            grid-template-columns: 1fr;
            gap: 1rem;
        }
        
        .method-card {
            padding: 1.5rem;
        }
        
        .main-header h1 {
            font-size: 2.2rem;
        }
        
        .main-header p {
            font-size: 1.1rem;
        }
    }
    
    /* Animation keyframes */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .animated {
        animation: fadeInUp 0.6s ease-out;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'summary_result' not in st.session_state:
    st.session_state.summary_result = ""
if 'documents' not in st.session_state:
    st.session_state.documents = None
if 'content_loaded' not in st.session_state:
    st.session_state.content_loaded = False
if 'input_type' not in st.session_state:
    st.session_state.input_type = None
if 'selected_method' not in st.session_state:
    st.session_state.selected_method = "📄 PDF Document"
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = "🚀 Llama 3.1 8B (Fast)"

def get_llm():
    """Initialize the LLM with selected Groq model"""
    model_config = GROQ_MODELS[st.session_state.selected_model]
    return ChatGroq(
        temperature=0,
        model=model_config["id"],
        max_tokens=None,
        timeout=None,
        max_retries=2
    )

def extract_video_id(youtube_url):
    """Extract video ID from YouTube URL"""
    if "youtu.be/" in youtube_url:
        return youtube_url.split("youtu.be/")[1].split("?")[0]
    elif "watch?v=" in youtube_url:
        return youtube_url.split("watch?v=")[1].split("&")[0]
    return None

def load_pdf_documents(uploaded_file):
    """Load and process PDF documents"""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        os.unlink(tmp_file_path)
        return documents
    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

def load_url_documents(url):
    """Load and process URL documents"""
    try:
        loader = UnstructuredURLLoader(urls=[url])
        documents = loader.load()
        return documents
    except Exception as e:
        st.error(f"Error loading URL: {str(e)}")
        return None

def load_youtube_documents(youtube_url):
    """Load and process YouTube documents using the correct API"""
    if not YOUTUBE_AVAILABLE:
        st.error("YouTube transcript functionality not available. Please install: pip install youtube-transcript-api")
        return None
        
    try:
        video_id = extract_video_id(youtube_url)
        if not video_id:
            st.error("Invalid YouTube URL format")
            return None
        
        # Create an instance of YouTubeTranscriptApi
        ytt_api = YouTubeTranscriptApi()
        
        try:
            # Use the new fetch() method instead of get_transcript()
            fetched_transcript = ytt_api.fetch(video_id)
            
            # Extract text from the transcript
            transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
            
            documents = [Document(
                page_content=transcript_text,
                metadata={
                    "source": youtube_url,
                    "video_id": video_id,
                    "language": fetched_transcript.language,
                    "language_code": fetched_transcript.language_code,
                    "is_generated": fetched_transcript.is_generated
                }
            )]
            return documents
            
        except Exception as e:
            # Try the fallback method using LangChain's YoutubeLoader
            try:
                from langchain_community.document_loaders import YoutubeLoader
                loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
                documents = loader.load()
                return documents
                
            except Exception as fallback_error:
                st.markdown(f"""
                <div class="status-error">
                    <span style="font-size: 1.2rem;">❌</span>
                    <div>
                        <strong>Could not fetch YouTube transcript</strong><br>
                        This could be due to:<br>
                        • Video has no subtitles/captions<br>
                        • Subtitles are disabled<br>
                        • Video is private or restricted<br>
                        • Geographic restrictions<br><br>
                        <small>Primary: {str(e)[:100]}... | Fallback: {str(fallback_error)[:100]}...</small>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                return None
                
    except Exception as e:
        st.error(f"Error loading YouTube video: {str(e)}")
        return None

def summarize_documents(documents, llm):
    """Summarize documents using stuff chain"""
    try:
        chain = load_summarize_chain(
            llm=llm, 
            chain_type="stuff",
            verbose=False
        )
        
        text_splitter = CharacterTextSplitter(
            chunk_size=4000,
            chunk_overlap=200,
            separator="\n"
        )
        
        # Handle large documents
        if len(documents) == 1 and len(documents[0].page_content) > 4000:
            split_docs = text_splitter.split_documents(documents)
            if len(split_docs) > 1:
                summaries = []
                for doc in split_docs:
                    summary = chain.invoke({"input_documents": [doc]})
                    summaries.append(Document(page_content=summary["output_text"]))
                final_summary = chain.invoke({"input_documents": summaries})
                return final_summary["output_text"]
        
        summary = chain.invoke({"input_documents": documents})
        return summary["output_text"]
        
    except Exception as e:
        st.error(f"Error during summarization: {str(e)}")
        return None

# Reset content loaded when method changes
def reset_content_state():
    st.session_state.content_loaded = False
    st.session_state.documents = None
    st.session_state.summary_result = ""

# Enhanced Header
st.markdown("""
<div class="main-header animated">
    <h1>🔍 AI Document Summarizer</h1>
    <p>Transform your documents, articles, and videos into concise, intelligent summaries powered by advanced AI technology</p>
</div>
""", unsafe_allow_html=True)

# Step 1: Enhanced card-based method selection with working functionality
st.markdown("""
<div class="step-container animated">
    <div class="step-header">🎯 Step 1: Choose Your Input Method</div>
</div>
""", unsafe_allow_html=True)

# Create three columns for method cards
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📄 PDF Document", key="pdf_card", use_container_width=True):
        st.session_state.selected_method = "📄 PDF Document"
        reset_content_state()
    
    # Card styling
    selected_class = "selected" if st.session_state.selected_method == "📄 PDF Document" else ""
    st.markdown(f"""
    <div class="method-card {selected_class}">
        <div class="method-icon">📄</div>
        <div class="method-title">PDF Document</div>
        <div class="method-description">Upload and analyze PDF files with intelligent text extraction and processing</div>
        <div class="method-features">
            <div class="feature-item">✓ Multi-page support</div>
            <div class="feature-item">✓ Text & image extraction</div>
            <div class="feature-item">✓ Structured content analysis</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    if st.button("🌐 Website Article", key="website_card", use_container_width=True):
        st.session_state.selected_method = "🌐 Website Article"
        reset_content_state()
    
    selected_class = "selected" if st.session_state.selected_method == "🌐 Website Article" else ""
    st.markdown(f"""
    <div class="method-card {selected_class}">
        <div class="method-icon">🌐</div>
        <div class="method-title">Website Article</div>
        <div class="method-description">Extract and summarize content from any web article, blog post, or online resource</div>
        <div class="method-features">
            <div class="feature-item">✓ Real-time web scraping</div>
            <div class="feature-item">✓ Clean content extraction</div>
            <div class="feature-item">✓ Multiple formats supported</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    if st.button("📺 YouTube Video", key="youtube_card", use_container_width=True):
        st.session_state.selected_method = "📺 YouTube Video"
        reset_content_state()
    
    selected_class = "selected" if st.session_state.selected_method == "📺 YouTube Video" else ""
    st.markdown(f"""
    <div class="method-card {selected_class}">
        <div class="method-icon">📺</div>
        <div class="method-title">YouTube Video</div>
        <div class="method-description">Generate intelligent summaries from video transcripts and automated captions</div>
        <div class="method-features">
            <div class="feature-item">✓ Automatic transcript fetch</div>
            <div class="feature-item">✓ Multiple language support</div>
            <div class="feature-item">✓ Video metadata included</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Step 2: Enhanced input section based on selection
st.markdown("""
<div class="step-container animated">
    <div class="step-header">📝 Step 2: Provide Your Input</div>
</div>
""", unsafe_allow_html=True)

if st.session_state.selected_method == "📄 PDF Document":
    st.markdown("""
    <div class="input-section">
        <div class="input-title">📤 Upload PDF File</div>
    </div>
    """, unsafe_allow_html=True)
    
    uploaded_pdf = st.file_uploader(
        "Choose a PDF file to summarize",
        type="pdf",
        help="Upload a PDF document (max 200MB) - supports multi-page documents with text and images"
    )
    
    if uploaded_pdf:
        st.markdown(f'''
        <div class="status-success">
            <span style="font-size: 1.2rem;">✅</span>
            <div>
                <strong>PDF uploaded successfully!</strong><br>
                <small>File: {uploaded_pdf.name} • Size: {uploaded_pdf.size/1024:.1f} KB</small>
            </div>
        </div>
        ''', unsafe_allow_html=True)
        
        # Load PDF button
        if st.button("📖 Load & Process PDF", type="secondary", key="load_pdf"):
            with st.spinner("📖 Processing PDF document..."):
                documents = load_pdf_documents(uploaded_pdf)
                if documents:
                    st.session_state.documents = documents
                    st.session_state.content_loaded = True
                    st.session_state.input_type = "PDF"
                    st.markdown(f'''
                    <div class="status-success">
                        <span style="font-size: 1.2rem;">✅</span>
                        <div>
                            <strong>PDF content processed successfully!</strong><br>
                            <small>{len(documents)} pages extracted • Ready for AI analysis</small>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                else:
                    st.markdown('<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>Failed to process PDF</strong><br><small>Please check the file format and try again</small></div></div>', unsafe_allow_html=True)

elif st.session_state.selected_method == "🌐 Website Article":
    st.markdown("""
    <div class="input-section">
        <div class="input-title">🔗 Enter Website URL</div>
    </div>
    """, unsafe_allow_html=True)
    
    website_url = st.text_input(
        "Website URL",
        placeholder="https://example.com/article",
        help="Enter a valid website URL to summarize its content - works with articles, blogs, and news sites"
    )
    
    if website_url:
        if validators.url(website_url):
            st.markdown(f'''
            <div class="status-success">
                <span style="font-size: 1.2rem;">✅</span>
                <div>
                    <strong>Valid URL detected!</strong><br>
                    <small>{website_url[:50]}{'...' if len(website_url) > 50 else ''}</small>
                </div>
            </div>
            ''', unsafe_allow_html=True)
            
            # Load URL button
            if st.button("🌐 Load & Extract Content", type="secondary", key="load_url"):
                with st.spinner("🌐 Fetching and extracting website content..."):
                    documents = load_url_documents(website_url)
                    if documents:
                        st.session_state.documents = documents
                        st.session_state.content_loaded = True
                        st.session_state.input_type = "Website"
                        content_length = len(documents[0].page_content) if documents else 0
                        st.markdown(f'''
                        <div class="status-success">
                            <span style="font-size: 1.2rem;">✅</span>
                            <div>
                                <strong>Website content extracted successfully!</strong><br>
                                <small>{content_length:,} characters extracted • Ready for AI analysis</small>
                            </div>
                        </div>
                        ''', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>Failed to extract website content</strong><br><small>Please verify the URL is accessible and try again</small></div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>Invalid URL format</strong><br><small>Please enter a complete URL starting with http:// or https://</small></div></div>', unsafe_allow_html=True)

elif st.session_state.selected_method == "📺 YouTube Video":
    st.markdown("""
    <div class="input-section">
        <div class="input-title">🎥 Enter YouTube URL</div>
    </div>
    """, unsafe_allow_html=True)
    
    if not YOUTUBE_AVAILABLE:
        st.markdown('<div class="status-warning"><span style="font-size: 1.2rem;">⚠️</span><div><strong>YouTube functionality requires additional package</strong><br><small>Run: <code>pip install youtube-transcript-api</code></small></div></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="status-info"><span style="font-size: 1.2rem;">🔧</span><div><strong>YouTube Transcript API Ready</strong><br><small>Using v1.0+ API with LangChain fallback support</small></div></div>', unsafe_allow_html=True)
        
        youtube_url = st.text_input(
            "YouTube URL",
            placeholder="https://youtube.com/watch?v=... or https://youtu.be/...",
            help="Enter a YouTube video URL to summarize its transcript - works best with videos that have captions"
        )
        
        if youtube_url:
            video_id = extract_video_id(youtube_url)
            if video_id:
                st.markdown(f'''
                <div class="status-success">
                    <span style="font-size: 1.2rem;">✅</span>
                    <div>
                        <strong>Valid YouTube URL detected!</strong><br>
                        <small>Video ID: {video_id}</small>
                    </div>
                </div>
                ''', unsafe_allow_html=True)
                
                # Load YouTube button
                if st.button("📺 Load & Extract Transcript", type="secondary", key="load_youtube"):
                    with st.spinner("📺 Fetching YouTube transcript..."):
                        documents = load_youtube_documents(youtube_url)
                        if documents:
                            st.session_state.documents = documents
                            st.session_state.content_loaded = True
                            st.session_state.input_type = "YouTube"
                            transcript_length = len(documents[0].page_content) if documents else 0
                            language = documents[0].metadata.get('language', 'Unknown') if documents else 'Unknown'
                            st.markdown(f'''
                            <div class="status-success">
                                <span style="font-size: 1.2rem;">✅</span>
                                <div>
                                    <strong>YouTube transcript extracted successfully!</strong><br>
                                    <small>{transcript_length:,} characters • Language: {language} • Ready for AI analysis</small>
                                </div>
                            </div>
                            ''', unsafe_allow_html=True)
                        else:
                            st.markdown('<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>Failed to extract YouTube transcript</strong><br><small>Check error details above for more information</small></div></div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>Invalid YouTube URL format</strong><br><small>Please use: youtube.com/watch?v=... or youtu.be/...</small></div></div>', unsafe_allow_html=True)

# Step 3: Enhanced summary generation
st.markdown("""
<div class="step-container animated">
    <div class="step-header">🚀 Step 3: Generate AI Summary</div>
</div>
""", unsafe_allow_html=True)

# Show content loaded status
if st.session_state.content_loaded and st.session_state.documents:
    content_preview = st.session_state.documents[0].page_content[:200] + "..." if st.session_state.documents else ""
    current_model = GROQ_MODELS[st.session_state.selected_model]
    st.markdown(f'''
    <div class="status-info">
        <span style="font-size: 1.2rem;">✅</span>
        <div>
            <strong>{st.session_state.input_type} content loaded and ready!</strong><br>
            <small>Will use: {st.session_state.selected_model} • Preview: {content_preview}</small>
        </div>
    </div>
    ''', unsafe_allow_html=True)
    
    # Generate Summary Button - only show when content is loaded
    if st.button("✨ Generate AI Summary", type="primary"):
        with st.spinner(f"🤖 {current_model['description']} is analyzing your content..."):
            try:
                llm = get_llm()
                summary = summarize_documents(st.session_state.documents, llm)
                
                if summary:
                    st.session_state.summary_result = summary
                    st.markdown('<div class="status-success"><span style="font-size: 1.2rem;">✅</span><div><strong>AI summary generated successfully!</strong><br><small>Summary is ready for review and download</small></div></div>', unsafe_allow_html=True)
                    st.experimental_rerun()  # Refresh to show results
                else:
                    st.markdown('<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>Failed to generate summary</strong><br><small>Please try again or contact support</small></div></div>', unsafe_allow_html=True)
            except Exception as e:
                st.markdown(f'<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>AI processing error</strong><br><small>{str(e)[:100]}...</small></div></div>', unsafe_allow_html=True)

elif st.session_state.content_loaded and not st.session_state.documents:
    st.markdown('<div class="status-error"><span style="font-size: 1.2rem;">❌</span><div><strong>Content loading issue detected</strong><br><small>Please try reloading your content using the Load button above</small></div></div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="status-info"><span style="font-size: 1.2rem;">👆</span><div><strong>Ready to process your content</strong><br><small>Please load your content first using the Load button above</small></div></div>', unsafe_allow_html=True)

# Enhanced Results Display
if st.session_state.summary_result:
    st.markdown("""
    <div class="summary-container animated">
        <h2 style="color: #2c3e50; margin-bottom: 1.5rem; font-family: 'Poppins', sans-serif; font-weight: 600;">📝 AI-Generated Summary</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Create enhanced tabs
    tab1, tab2, tab3 = st.tabs(["📖 Summary", "📊 Analytics", "🔧 Actions"])
    
    with tab1:
        st.markdown(f"""
        <div class="summary-text">
            {st.session_state.summary_result}
        </div>
        """, unsafe_allow_html=True)
    
    with tab2:
        col1, col2, col3 = st.columns(3)
        
        word_count = len(st.session_state.summary_result.split())
        char_count = len(st.session_state.summary_result)
        reading_time = max(1, word_count // 200)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">Word Count</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{word_count:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">Characters</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{char_count:,}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3 style="margin: 0; font-size: 1rem; opacity: 0.9;">Read Time</h3>
                <p style="margin: 0.5rem 0 0 0; font-size: 2rem; font-weight: 700;">{reading_time} min</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Additional analytics
        sentences = st.session_state.summary_result.count('.') + st.session_state.summary_result.count('!') + st.session_state.summary_result.count('?')
        paragraphs = len([p for p in st.session_state.summary_result.split('\n') if p.strip()])
        
        st.markdown("### 📈 Content Analysis")
        col4, col5, col6 = st.columns(3)
        with col4:
            st.metric("Sentences", f"{sentences}")
        with col5:
            st.metric("Paragraphs", f"{paragraphs}")
        with col6:
            current_model = GROQ_MODELS[st.session_state.selected_model]
            st.metric("Model Used", current_model['id'].split('/')[-1])
    
    with tab3:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy to Clipboard", key="copy_btn"):
                st.code(st.session_state.summary_result, language=None)
                st.success("📋 Summary ready to copy - select the text above!")
        
        with col2:
            # Enhanced download with timestamp and model info
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_name = GROQ_MODELS[st.session_state.selected_model]['id'].replace('/', '_')
            filename = f"{st.session_state.input_type.lower()}_{model_name}_{timestamp}.txt"
            
            st.download_button(
                label="💾 Download Summary",
                data=st.session_state.summary_result,
                file_name=filename,
                mime="text/plain",
                help="Download the summary as a text file with model info"
            )
        
        with col3:
            if st.button("🔄 Create New Summary", key="new_summary_btn"):
                # Reset all states
                for key in ['summary_result', 'content_loaded', 'documents', 'input_type']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.experimental_rerun()

# Enhanced Sidebar (Now Much Wider with Model Selection!)
with st.sidebar:
    st.markdown("""
    <div class="sidebar-header">
        <h2 style="margin: 0; font-family: 'Poppins', sans-serif; font-weight: 600;">🤖 AI Model Selection</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # MODEL SELECTION - The main new feature!
    st.markdown("### 🧠 Choose Your AI Model")
    
    # Group models by category
    production_models = [k for k, v in GROQ_MODELS.items() if v['category'] == 'production']
    preview_models = [k for k, v in GROQ_MODELS.items() if v['category'] == 'preview']
    
    st.markdown("#### 🚀 **Production Models** (Recommended)")
    for model_name in production_models:
        model_info = GROQ_MODELS[model_name]
        if st.button(
            f"{model_name}",
            key=f"model_{model_name}",
            help=f"{model_info['description']} • Context: {model_info['context']} • {model_info['speed']} • {model_info['cost']}",
            use_container_width=True
        ):
            st.session_state.selected_model = model_name
            st.experimental_rerun()
        
        # Show selection indicator
        if st.session_state.selected_model == model_name:
            st.markdown(f'''
            <div style="background: linear-gradient(135deg, #10b981, #059669); color: white; padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.85rem;">
                ✅ <strong>Selected:</strong> {model_info['description']}<br>
                <small>{model_info['speed']} • {model_info['cost']} • Context: {model_info['context']}</small>
            </div>
            ''', unsafe_allow_html=True)
    
    st.markdown("#### 🔬 **Preview Models** (Experimental)")
    with st.expander("Show Preview Models", expanded=False):
        for model_name in preview_models:
            model_info = GROQ_MODELS[model_name]
            if st.button(
                f"{model_name}",
                key=f"model_preview_{model_name}",
                help=f"{model_info['description']} • Context: {model_info['context']} • {model_info['speed']} • {model_info['cost']}",
                use_container_width=True
            ):
                st.session_state.selected_model = model_name
                st.experimental_rerun()
            
            # Show selection indicator for preview models
            if st.session_state.selected_model == model_name:
                st.markdown(f'''
                <div style="background: linear-gradient(135deg, #f59e0b, #d97706); color: white; padding: 0.5rem; border-radius: 8px; margin: 0.5rem 0; font-size: 0.85rem;">
                    ⚠️ <strong>Preview Selected:</strong> {model_info['description']}<br>
                    <small>{model_info['speed']} • {model_info['cost']} • Context: {model_info['context']}</small>
                </div>
                ''', unsafe_allow_html=True)
    
    # Current Status
    st.markdown("""
    <div class="sidebar-header" style="margin-top: 1.5rem;">
        <h2 style="margin: 0; font-family: 'Poppins', sans-serif; font-weight: 600;">📊 Current Status</h2>
    </div>
    """, unsafe_allow_html=True)
    
    # Show current status with enhanced styling
    if st.session_state.content_loaded:
        st.markdown(f"""
        <div class="sidebar-status" style="background: linear-gradient(135deg, #10b981, #059669); color: white;">
            ✅ {st.session_state.input_type} Content Loaded
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="sidebar-status" style="background: linear-gradient(135deg, #6b7280, #4b5563); color: white;">
            ⏳ No Content Loaded Yet
        </div>
        """, unsafe_allow_html=True)
    
    # Show selected method
    st.markdown(f"""
    <div class="sidebar-status" style="background: linear-gradient(135deg, #667eea, #764ba2); color: white;">
        🎯 Input: {st.session_state.selected_method.replace("📄 ", "").replace("🌐 ", "").replace("📺 ", "")}
    </div>
    """, unsafe_allow_html=True)
    
    # Show YouTube availability
    youtube_status = "✅ Available" if YOUTUBE_AVAILABLE else "❌ Not Available"
    youtube_color = "#10b981" if YOUTUBE_AVAILABLE else "#ef4444"
    
    st.markdown(f"""
    ### 🌟 **Features Overview**
    
    **📄 PDF Processing**
    - Multi-page document support
    - Advanced text and image extraction
    - Structured content analysis
    - File format validation
    
    **🌐 Web Content Extraction**
    - Real-time web scraping
    - Clean content extraction
    - Multiple site compatibility
    - URL validation & safety checks
    
    **📺 Video Transcript Analysis**
    - Status: <span style="color: {youtube_color};">{youtube_status}</span>
    - Automatic transcript fetching
    - Multi-language support
    - Video metadata inclusion
    - Fallback methods for reliability
    
    ### 🤖 **Current AI Configuration**
    
    **Selected Model:** `{GROQ_MODELS[st.session_state.selected_model]['id']}`  
    **Description:** {GROQ_MODELS[st.session_state.selected_model]['description']}  
    **Context Window:** {GROQ_MODELS[st.session_state.selected_model]['context']} tokens  
    **Performance:** {GROQ_MODELS[st.session_state.selected_model]['speed']}  
    **Cost Level:** {GROQ_MODELS[st.session_state.selected_model]['cost']}
    
    **Processing Features:**
    - Temperature: 0 (Deterministic output)
    - Intelligent text chunking for large docs
    - Hierarchical summarization
    - Error recovery & fallbacks
    - Content validation
    
    ### 📋 **Usage Guide**
    
    **Step-by-Step Process:**
    1. **Select AI Model** - Choose from 8+ Groq models above
    2. **Choose Input Method** - Pick content type from cards
    3. **Upload/Enter Content** - Provide your source material
    4. **Load & Process** - System extracts content
    5. **Generate Summary** - AI creates intelligent summary
    6. **Review & Download** - Access results with analytics
    
    **Model Recommendations:**
    - **Fast Processing:** Llama 3.1 8B
    - **High Quality:** Llama 3.3 70B or GPT-OSS 120B
    - **Large Documents:** Kimi K2 (262K context)
    - **Experimental:** Llama 4 models
    """, unsafe_allow_html=True)
    
    # Expandable sections with enhanced styling
    with st.expander("🔧 **Installation Guide**", expanded=False):
        st.code("""
# Core Requirements (Essential)
pip install streamlit langchain langchain-community
pip install langchain-groq validators pypdf
pip install unstructured python-dotenv

# YouTube Support (Recommended)
pip install youtube-transcript-api

# Optional Enhancements  
pip install beautifulsoup4 requests

# Environment Setup
# Create .env file with:
# GROQ_API_KEY=your_groq_api_key_here
        """, language="bash")
    
    with st.expander("💡 **Model Comparison & Tips**", expanded=False):
        st.markdown(f"""
        **🚀 Production Models:**
        
        **Llama 3.1 8B** - Ultra-fast, perfect for quick summaries
        • Best for: Speed, low cost, simple documents
        • Context: 131K tokens
        
        **Llama 3.3 70B** - Best balance of quality and speed  
        • Best for: Complex documents, detailed summaries
        • Context: 131K tokens
        
        **GPT-OSS 120B** - Premium quality with reasoning
        • Best for: Academic papers, complex analysis
        • Context: 131K tokens
        
        **GPT-OSS 20B** - Balanced performance
        • Best for: General use, good quality/speed ratio
        • Context: 131K tokens
        
        **🔬 Preview Models:**
        
        **Llama 4 Maverick/Scout** - Next-gen experimental
        • Best for: Testing latest capabilities
        • Context: 131K tokens
        
        **Kimi K2** - Ultra-long context
        • Best for: Very large documents (books, reports)
        • Context: 262K tokens (longest available!)
        
        **Qwen3 32B** - Multilingual specialist
        • Best for: Non-English content, translations
        • Context: 131K tokens
        
        **📊 Performance Tips:**
        - Faster models: 8B → 20B → 32B → 70B → 120B
        - Better quality: 8B → 20B → 32B → 70B → 120B  
        - Longest context: Kimi K2 (262K tokens)
        - Most cost-effective: Llama 3.1 8B
        - Best overall: Llama 3.3 70B or GPT-OSS 120B
        """)
    
    with st.expander("🎨 **Theme & Customization**", expanded=False):
        st.markdown("""
        **Theme Support:**
        - ✅ Light mode optimized with professional colors
        - ✅ Dark mode compatible with adaptive styling
        - ✅ Auto-adaptation based on system preferences
        - ✅ Enhanced sidebar width (420px)
        
        **Model Selection Features:**
        - 8+ AI models to choose from
        - Real-time model switching
        - Performance indicators
        - Cost and speed information
        - Production vs Preview categorization
        
        **Custom Design Features:**
        - Beautiful gradient backgrounds
        - Smooth hover animations
        - Interactive model selection buttons
        - Professional typography (Poppins & Inter)
        - Color-coded status messages
        - Responsive mobile design
        """)

# Enhanced Footer with model information
st.markdown(f"""
<div style="background: linear-gradient(135deg, #2c3e50, #3498db); color: white; text-align: center; padding: 2rem; border-radius: 16px; margin-top: 3rem; box-shadow: 0 -8px 32px rgba(0,0,0,0.15);">
    <div style="font-family: 'Poppins', sans-serif; font-size: 1.2rem; font-weight: 600; margin-bottom: 0.5rem;">
        Built with ❤️ using <strong>Streamlit & LangChain</strong>
    </div>
    <div style="font-family: 'Inter', sans-serif; opacity: 0.9; margin-bottom: 1rem;">
        Powered by <strong>Groq's Lightning-Fast LLM Inference</strong>
    </div>
    <div style="font-family: 'Inter', sans-serif; opacity: 0.8; font-size: 0.9rem;">
        Current Model: <strong>{GROQ_MODELS[st.session_state.selected_model]['id']}</strong><br>
        {GROQ_MODELS[st.session_state.selected_model]['description']}
    </div>
    <div style="font-family: 'Inter', sans-serif; opacity: 0.7; font-size: 0.9rem; margin-top: 0.5rem;">
        © 2025 AI Document Summarizer • Professional AI-Powered Content Analysis<br>
        <small>Version 3.0 • Multi-Model Support • 8 AI Models • 420px Sidebar</small>
    </div>
</div>
""", unsafe_allow_html=True)

