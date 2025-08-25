import streamlit as st
import datetime
import os
import asyncio
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

st.set_page_config(
    page_title="AI Chore Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global Dark Theme */
    .stApp {
        background-color: #0a0a0a;
        color: #e0e0e0;
    }

    .main {
        font-family: 'Inter', sans-serif;
        background-color: #0a0a0a;
    }

    /* Sidebar Dark Theme */
    .css-1d391kg, .css-1y4p8pa {
        background-color: #1a1a1a !important;
    }

    .sidebar .sidebar-content {
        background-color: #1a1a1a;
    }

    /* Header Styling - Purple Gradient */
    .main-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%, #9333ea 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.3);
    }

    .main-header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.5);
    }

    .main-header p {
        font-size: 1.1rem;
        opacity: 0.9;
        font-weight: 400;
    }

    /* Date Display - Purple Theme */
    .date-display {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
        padding: 1rem;
        border-radius: 15px;
        text-align: center;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(147, 51, 234, 0.2);
        border: 1px solid #4c1d95;
    }

    .date-display h3 {
        margin: 0;
        color: #e0e7ff;
        font-weight: 600;
        font-size: 1.2rem;
    }

    .date-display p {
        margin: 0.5rem 0 0 0;
        color: #c4b5fd;
        font-size: 0.9rem;
    }

    /* Chat Messages Dark Theme */
    .stChatMessage {
        background-color: #1a1a1a !important;
        border: 1px solid #2a2a2a;
        margin-bottom: 1rem;
    }

    /* Chat Input Dark Theme */
    .stChatInputContainer {
        background: #1a1a1a !important;
        border-radius: 25px;
        box-shadow: 0 2px 15px rgba(147, 51, 234, 0.1);
        border: 1px solid #4c1d95;
    }

    /* Button Styling - Purple Theme */
    .stButton > button {
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.2rem;
        border-radius: 20px;
        font-weight: 500;
        transition: all 0.3s ease;
        width: 100%;
        box-shadow: 0 2px 10px rgba(147, 51, 234, 0.3);
        font-family: 'Inter', sans-serif;
    }

    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.4);
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #a855f7 100%);
    }

    /* Clear Chat Button - Red Purple Theme */
    .clear-chat-btn {
        background: linear-gradient(135deg, #dc2626 0%, #9333ea 100%);
        color: white;
        border: none;
        padding: 0.7rem 1.5rem;
        border-radius: 20px;
        font-weight: 600;
        width: 100%;
        transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3);
    }

    /* Sidebar Text Color */
    .css-1d391kg .css-1y4p8pa h3 {
        color: #e0e7ff !important;
    }

    /* Hide Streamlit Elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dark scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        background-color: #1a1a1a;
    }

    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #6366f1, #8b5cf6);
        border-radius: 4px;
    }

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏠 AI Chore Assistant</h1>
    <p>Your smart household chore schedule companion</p>
</div>
""", unsafe_allow_html=True)

# --- 3. API Key and LLM Initialization ---
try:
    if "GOOGLE_API_KEY" not in os.environ:
        os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0.2
    )
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
except Exception as e:
    st.error(f"Error initializing AI models: {e}")
    st.stop()

CYCLE_START_DATE = datetime.date(2025, 6, 30)


def validate_and_get_week_info(target_date=None):
    if target_date is None:
        target_date = datetime.date.today()

    delta_days = (target_date - CYCLE_START_DATE).days

    if delta_days < 0:
        return target_date, 1, target_date.strftime("%A"), "⚠️ Invalid Date"

    week_number = (delta_days // 7) % 4 + 1
    day_name = target_date.strftime("%A")
    status = "🔴 Today" if target_date == datetime.date.today() else "📅 Selected Date"

    return target_date, week_number, day_name, status

@st.cache_resource
def get_vectorstore():
    try:
        if not os.path.exists("chore_schedule.pdf"):
            st.error("chore_schedule.pdf not found!")
            return None

        loader = PyPDFLoader("chore_schedule.pdf")
        docs = loader.load()

        if not docs:
            st.error("PDF loaded but no content found!")
            return None

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=150,
            separators=["\n\n", "\n", "Week", "Chore", "•", " ", ""]
        )
        splits = text_splitter.split_documents(docs)

        for i, split in enumerate(splits):
            split.metadata["chunk_id"] = i
            split.metadata["char_count"] = len(split.page_content)

        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore

    except Exception as e:
        st.error(f"Error loading PDF: {str(e)}")
        return None

vectorstore = get_vectorstore()
if not vectorstore:
    st.stop()

contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "reformulate the question to be standalone and clear. "
    "Do NOT answer the question, just reformulate it if needed."
)

qa_system_prompt = """
You are a helpful AI assistant for a household chore schedule system.

HOUSEHOLD MEMBERS (in rotation order):
1. Tanmay
2. Soham  
3. Pranjul
4. Ishika

RESPONSE FORMATTING RULES:

For "list duties" or "current week" questions, use this TABULAR FORMAT:

| **Person** | **Main Chore** | **Kitchen Trash Day** |
|------------|----------------|----------------------|
| **Tanmay** | Bathroom Floor & Bathtub Cleaning | Thursday |
| **Soham** | Kitchen Foil & Platform Cleaning | Monday |
| **Pranjul** | Kitchen Floor Cleaning | Sunday |
| **Ishika** | Toilet & Sink Cleaning + Black Trash (Tue) | N/A |

For single person questions:
"Tanmay is responsible for bathroom floor and bathtub cleaning this week."

For daily questions:
"Today, Soham takes out the kitchen trash."

RESPONSE STYLE:
- Use markdown tables for multiple assignments
- Bold person names for clarity  
- Keep individual responses conversational
- Never mention "Week 1", "Week 2", etc. - just say "this week"
- Answer open-ended questions naturally and helpfully
- Be conversational and engaging while staying informative

Use the context below to answer accurately:

{context}
"""

contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", qa_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}")
])

history_aware_retriever = create_history_aware_retriever(
    llm, vectorstore.as_retriever(search_kwargs={"k": 6}), contextualize_q_prompt )
