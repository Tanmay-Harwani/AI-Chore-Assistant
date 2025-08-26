import streamlit as st
import datetime
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# PAGE CONFIG
st.set_page_config(
    page_title="AI Chore Assistant",
    page_icon="🏠",
    layout="wide"
)

# STYLING
st.markdown("""
<style>
    .main { background-color: #0a0a0a; color: #e0e0e0; font-family: 'Inter', sans-serif; }
    .main-header { 
        text-align: center; padding: 2rem; margin-bottom: 2rem;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px; color: white;
        box-shadow: 0 4px 20px rgba(147, 51, 234, 0.3);
    }
    .date-info {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 50%);
        padding: 1rem; border-radius: 10px; margin-bottom: 1rem;
        text-align: center; color: #e0e7ff;
    }
    .stChatMessage { background-color: #1a1a1a !important; border: 1px solid #2a2a2a; }
    .stButton > button { 
        background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
        color: white; border: none; border-radius: 20px;
        font-weight: 500; width: 100%;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #7c3aed 0%, #9333ea 100%);
        transform: translateY(-2px);
    }
</style>
""", unsafe_allow_html=True)

# HEADER
st.markdown("""
<div class="main-header">
    <h1>🏠 AI Chore Assistant</h1>
    <p>Your smart household chore schedule companion</p>
</div>
""", unsafe_allow_html=True)


@st.cache_resource
def get_llm():
    """Initialize Gemini LLM"""
    try:
        api_key = None

        if "GOOGLE_API_KEY" in os.environ:
            api_key = os.environ["GOOGLE_API_KEY"]
        elif hasattr(st, 'secrets') and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]

        if not api_key:
            st.error("🔑 **API Key Missing!** Add `GOOGLE_API_KEY` to Streamlit secrets or environment variables.")
            st.info("For local testing: `export GOOGLE_API_KEY='your-key'`")
            st.info("For Streamlit Cloud: Add secret in app settings")
            st.stop()

        return ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=api_key
        )

    except Exception as e:
        st.error(f"❌ LLM Error: {e}")
        st.stop()


@st.cache_resource
def build_simple_search():
    """Build simple search from PDF using scikit-learn"""
    try:
        with st.spinner("🔍 Building search index..."):
            # Load PDF
            if not os.path.exists("chore_schedule.pdf"):
                st.error("📄 **chore_schedule.pdf not found!** Make sure it's in your repo.")
                st.stop()

            loader = PyPDFLoader("chore_schedule.pdf")
            docs = loader.load()

            if not docs:
                st.error("❌ PDF loaded but no content found!")
                st.stop()

            # Split documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=600,
                chunk_overlap=100,
                separators=["\nWeek", "\n\n", "\n•", "\n", " "]
            )
            splits = text_splitter.split_documents(docs)

            # Create embeddings
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

            # Get embeddings for all chunks
            texts = [doc.page_content for doc in splits]
            vectors = embeddings.embed_documents(texts)

            return {
                'texts': texts,
                'vectors': np.array(vectors),
                'documents': splits,
                'embeddings': embeddings
            }

    except Exception as e:
        st.error(f"❌ Search setup error: {e}")
        st.stop()


def simple_search(search_data, query, k=4):
    """Simple similarity search using cosine similarity"""
    try:
        query_vector = search_data['embeddings'].embed_query(query)
        similarities = cosine_similarity([query_vector], search_data['vectors'])[0]
        top_indices = np.argsort(similarities)[-k:][::-1]

        results = []
        for idx in top_indices:
            results.append({
                'content': search_data['texts'][idx],
                'document': search_data['documents'][idx],
                'similarity': similarities[idx]
            })
        return results
    except Exception as e:
        st.error(f"Search error: {e}")
        return []


def get_week_info():
    """Calculate current week in rotation"""
    CYCLE_START_DATE = datetime.date(2025, 6, 30)
    today = datetime.date.today()
    delta_days = (today - CYCLE_START_DATE).days
    week_number = (delta_days // 7) % 4 + 1 if delta_days >= 0 else 1
    return today, week_number, today.strftime("%A")


def get_trash_person_today():
    """Calculate who takes out kitchen trash today"""
    # Kitchen trash days: Monday, Thursday, Sunday
    trash_days = {
        'Monday': 0, 'Thursday': 1, 'Sunday': 2
    }

    if current_day not in trash_days:
        return None, None

    # Get day index
    day_index = trash_days[current_day]

    # Calculate rotation: each week has 3 trash days, rotate through 4 people
    total_days_passed = (current_week - 1) * 3 + day_index
    person_index = total_days_passed % 4

    members = ["Tanmay", "Soham", "Pranjul", "Ishika"]
    return members[person_index], current_day


def get_rag_response(question, chat_history):
    """Simple RAG without complex chains"""
    try:
        # Search for relevant content
        results = simple_search(search_data, question, k=3)
        context = "\n\n".join([r['content'] for r in results])

        # Create prompt
        system_msg = f"""You are a helpful AI assistant for household chore scheduling.

HOUSEHOLD MEMBERS (rotation order): Tanmay, Soham, Pranjul, Ishika

CURRENT CONTEXT:
- Today: {current_date.strftime('%A, %B %d, %Y')} 
- Current week: Week {current_week} in 4-week rotation
- Day: {current_day}

CHORE TYPES:
- Weekly rotating chores (Toilet & Sink, Bathroom Floor & Bathtub, Kitchen Foil & Platform, Kitchen Floor)
- Kitchen trash (Monday, Thursday, Sunday - rotates among all members)
- Black trash disposal (Tuesday nights - rotates weekly)

RESPONSE STYLE:
- Be friendly and conversational
- Use emojis and clear formatting
- Give specific, actionable answers
- Never mention "Week 1/2/3/4" - just say "this week"

Based on this context from the chore schedule:

{context}

Question: {question}

Please provide a helpful response based on the schedule and current date."""

        response = llm.invoke(system_msg)
        return response.content

    except Exception as e:
        return f"Sorry, I encountered an error: {str(e)}"


# INITIALIZE
llm = get_llm()
search_data = build_simple_search()
current_date, current_week, current_day = get_week_info()

# SIDEBAR
with st.sidebar:
    st.markdown(f"""
    <div class="date-info">
        <h3>📅 {current_day}</h3>
        <p>{current_date.strftime('%B %d, %Y')}</p>
        <small>Week {current_week} of rotation</small>
    </div>
    """, unsafe_allow_html=True)

    # Show today's trash info
    trash_person, trash_day = get_trash_person_today()
    if trash_person:
        st.markdown(f"### 🗑️ Today's Kitchen Trash")
        st.success(f"**{trash_person}** takes it out!")
    else:
        st.markdown("### 🗑️ Kitchen Trash")
        st.info("No pickup today\n(Mon/Thu/Sun only)")

    st.markdown("### 💡 Quick Questions")

    questions = [
        "What are this week's chore assignments?",
        "Who takes out kitchen trash today?",
        "What does Tanmay need to do this week?"
    ]

    for i, q in enumerate(questions):
        if st.button(q, key=f"quick_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# CHAT INTERFACE
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Process last message if it's from user and needs response
if (st.session_state.messages and
        st.session_state.messages[-1]["role"] == "user" and
        (len(st.session_state.messages) == 1 or st.session_state.messages[-2]["role"] != "assistant")):

    user_question = st.session_state.messages[-1]["content"].lower()

    with st.chat_message("assistant"):
        try:
            # Special handling for kitchen trash questions
            if "kitchen trash" in user_question and "today" in user_question:
                trash_person, trash_day = get_trash_person_today()
                if trash_person:
                    response = f"🗑️ **{trash_person}** takes out the kitchen trash today ({current_day})!"
                    st.markdown(response)
                else:
                    next_days = []
                    days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    current_day_index = days_of_week.index(current_day)

                    # Find next kitchen trash days
                    trash_days = ['Monday', 'Thursday', 'Sunday']
                    for day in trash_days:
                        day_index = days_of_week.index(day)
                        if day_index > current_day_index:
                            next_days.append(day)
                            break

                    if not next_days:  # If no days left this week, next Monday
                        next_days.append('Monday')

                    response = f"📅 No kitchen trash pickup today ({current_day}).\n\n🗑️ Kitchen trash goes out on **Monday**, **Thursday**, and **Sunday**.\n\nNext pickup: **{next_days[0]}**"
                    st.markdown(response)

                st.session_state.messages.append({"role": "assistant", "content": response})

            else:
                # Use simple RAG for other questions
                with st.spinner("Thinking..."):
                    response = get_rag_response(st.session_state.messages[-1]["content"],
                                                st.session_state.messages[:-1])

                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})

        except Exception as e:
            error_msg = f"Sorry, I encountered an error: {str(e)}"
            st.error(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})

# Handle new input
if prompt := st.chat_input("Ask about chores, schedules, or responsibilities..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.rerun()