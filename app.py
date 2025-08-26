import streamlit as st
import datetime
import os
import pickle
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.embeddings import HuggingFaceEmbeddings

# --- 1. App Configuration ---
st.set_page_config(
    page_title="AI Chore Assistant",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 2. Custom Styling ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    .stApp { background-color: #0a0a0a; color: #e0e0e0; }
    .main { font-family: 'Inter', sans-serif; background-color: #0a0a0a; }
    .sidebar .sidebar-content { background-color: #1a1a1a; }
    .main-header { text-align: center; padding: 2rem 0 1rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%, #9333ea 100%);
        border-radius: 15px; margin-bottom: 2rem; color: white;
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.3); }
    .main-header h1 { font-size: 2.5rem; font-weight: 700; margin-bottom: 0.5rem; }
    .main-header p { font-size: 1.1rem; opacity: 0.9; font-weight: 400; }
    .date-display { background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%, #4c1d95 100%);
        padding: 1rem; border-radius: 15px; text-align: center; margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(147, 51, 234, 0.2); border: 1px solid #4c1d95; }
    .date-display h3 { margin: 0; color: #e0e7ff; font-weight: 600; font-size: 1.2rem; }
    .date-display p { margin: 0.5rem 0 0 0; color: #c4b5fd; font-size: 0.9rem; }
    .stChatMessage { background-color: #1a1a1a !important; border: 1px solid #2a2a2a; margin-bottom: 1rem; }
    .stChatInputContainer { background: #1a1a1a !important; border-radius: 25px;
        box-shadow: 0 2px 15px rgba(147, 51, 234, 0.1); border: 1px solid #4c1d95; }
    .stButton > button { background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 50%, #a855f7 100%);
        color: white; border: none; padding: 0.7rem 1.2rem; border-radius: 20px;
        font-weight: 500; transition: all 0.3s ease; width: 100%;
        box-shadow: 0 2px 10px rgba(147, 51, 234, 0.3); font-family: 'Inter', sans-serif; }
    .stButton > button:hover { transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.4);
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 50%, #a855f7 100%); }
    .clear-chat-btn { background: linear-gradient(135deg, #dc2626 0%, #9333ea 100%);
        color: white; border: none; padding: 0.7rem 1.5rem; border-radius: 20px;
        font-weight: 600; width: 100%; transition: all 0.3s ease;
        box-shadow: 0 2px 8px rgba(220, 38, 38, 0.3); }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
    .error-container { background-color: #2d1b1b; border: 1px solid #dc2626; padding: 1rem; border-radius: 10px; margin: 1rem 0; }
</style>
""", unsafe_allow_html=True)

# --- 3. Header ---
st.markdown("""
<div class="main-header">
    <h1>🏠 AI Chore Assistant</h1>
    <p>Your smart household chore schedule companion</p>
</div>
""", unsafe_allow_html=True)


# --- 4. Load Precomputed FAISS Index ---
@st.cache_resource
def get_vectorstore():
    """Load vectorstore with fallback options"""
    try:
        # Try loading pickle file first
        if os.path.exists("faiss_index.pkl"):
            with open("faiss_index.pkl", "rb") as f:
                vectorstore = pickle.load(f)
            return vectorstore
    except Exception as e:
        pass

    try:
        # Try loading from save_local directory
        if os.path.exists("faiss_db"):
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            vectorstore = FAISS.load_local("faiss_db", embeddings, allow_dangerous_deserialization=True)
            return vectorstore
    except Exception as e:
        pass

    # If both methods fail, show error
    st.error("❌ Could not load vectorstore. Please run build_index.py first.")
    return None


@st.cache_resource
def get_llm():
    """Initialize LLM with better error handling"""
    try:
        # Try to get API key from multiple sources
        api_key = None

        # Check environment variable first
        if "GOOGLE_API_KEY" in os.environ:
            api_key = os.environ["GOOGLE_API_KEY"]

        # Check Streamlit secrets
        elif hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = api_key

        if not api_key:
            raise ValueError("Google API key not found")

        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=api_key
        )

        return llm

    except Exception as e:
        st.error(f"Error initializing AI model: {e}")
        st.markdown("""
        <div class="error-container">
        <h4>🔑 API Key Setup Required</h4>
        <p>To use this app, you need a Google API key:</p>
        <ol>
        <li>Get a free API key from <a href="https://makersuite.google.com/app/apikey" target="_blank">Google AI Studio</a></li>
        <li>Add it to your Streamlit secrets or environment variables as <code>GOOGLE_API_KEY</code></li>
        </ol>
        </div>
        """, unsafe_allow_html=True)
        return None


# Load resources
vectorstore = get_vectorstore()
llm = get_llm()

if not vectorstore or not llm:
    st.stop()

# --- 5. Date Logic ---
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


# --- 6. RAG Setup ---
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question, "
    "reformulate the question to be standalone and clear. "
    "Do NOT answer the question, just reformulate it if needed."
)

qa_system_prompt = """You are a helpful AI assistant for a household chore schedule system.

HOUSEHOLD MEMBERS (rotation order):
1. Tanmay
2. Soham  
3. Pranjul
4. Ishika

CHORE TYPES:
Weekly Chores (rotate weekly):
- Toilet & Sink Cleaning
- Bathroom Floor & Bathtub Cleaning  
- Kitchen Foil & Platform Cleaning
- Kitchen Floor Cleaning
- Black Trash Can Disposal (Tuesday night)

Kitchen Trash (3x per week): Monday, Thursday, Sunday

RESPONSE FORMATTING:
For weekly overview questions, use this format:

**🏠 This Week's Chore Assignments:**

| **Person** | **Weekly Chore** | **Black Trash (Tue)** |
|------------|------------------|----------------------|
| Tanmay     | [Chore Name]     | ✓ / -               |
| Soham      | [Chore Name]     | ✓ / -               |
| Pranjul    | [Chore Name]     | ✓ / -               |
| Ishika     | [Chore Name]     | ✓ / -               |

**🗑️ Kitchen Trash Schedule:**

| **Day** | **Person** |
|---------|------------|
| Monday  | [Name]     |
| Thursday| [Name]     |
| Sunday  | [Name]     |

For specific person questions: "[Name] is responsible for [specific chore] this week."

For daily questions: "Today is [Day], so [Name] takes out the [trash type]."

STYLE RULES:
- Be conversational and friendly
- Never mention "Week 1/2/3/4" - just say "this week"
- Use emojis to make responses visually appealing
- Keep answers concise but complete
- Always base answers on the provided context"""

try:
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt + "\n\nContext:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

except Exception as e:
    st.error(f"Error setting up RAG chain: {e}")
    st.stop()

# --- 7. Date Context ---
current_date, current_week, current_day, status = validate_and_get_week_info()


def create_detailed_context(date, week_num, day):
    return f"""
CURRENT DATE AND WEEK CONTEXT:
- Today's date: {date.strftime('%A, %B %d, %Y')}
- Current week in 4-week rotation: Week {week_num}
- Day of the week: {day}

IMPORTANT: Use the exact week number (Week {week_num}) to look up assignments from the provided chore schedule tables.
"""


# --- 8. Sidebar ---
with st.sidebar:
    st.markdown(f"""
    <div class="date-display">
        <h3>📅 {current_date.strftime('%A')}</h3>
        <p>{current_date.strftime('%B %d, %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 💡 Quick Questions")
    sample_questions = [
        "List the duties for the current week",
        "Who takes out the kitchen trash today?",
        "What does Tanmay need to do this week?"
    ]

    for i, q in enumerate(sample_questions):
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            # Add the question to chat history
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []

            # Add user message
            st.session_state.chat_history.append({"role": "user", "content": q})

            # Generate response immediately
            time_context = create_detailed_context(current_date, current_week, current_day)
            contextual_prompt = f"{time_context}\n\nUSER QUESTION: {q}"

            try:
                response = ""
                for chunk in rag_chain.stream({
                    "input": contextual_prompt,
                    "chat_history": st.session_state.chat_history[:-1]
                }):
                    if "answer" in chunk:
                        response += chunk["answer"]

                # Add assistant response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})

            except Exception as e:
                error_response = f"Sorry, I encountered an error: {str(e)}. Please try again."
                st.session_state.chat_history.append({"role": "assistant", "content": error_response})

            st.rerun()

    if st.button("🗑️ Clear Chat History", key="clear_chat", use_container_width=True):
        st.session_state.chat_history = []
        st.rerun()

# --- 9. Main Chat ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Handle new messages
if prompt := st.chat_input("Ask about chores..."):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    time_context = create_detailed_context(current_date, current_week, current_day)
    contextual_prompt = f"{time_context}\n\nUSER QUESTION: {prompt}"

    with st.chat_message("assistant"):
        try:
            def stream_generator():
                accumulated_response = ""
                for chunk in rag_chain.stream({
                    "input": contextual_prompt,
                    "chat_history": st.session_state.chat_history[:-1]
                }):
                    if "answer" in chunk:
                        content = chunk["answer"]
                        accumulated_response += content
                        yield content
                return accumulated_response


            full_response = st.write_stream(stream_generator())

        except Exception as e:
            full_response = f"Sorry, I encountered an error: {str(e)}. Please try rephrasing your question."
            st.error(full_response)

    if 'full_response' in locals():
        st.session_state.chat_history.append({"role": "assistant", "content": full_response})