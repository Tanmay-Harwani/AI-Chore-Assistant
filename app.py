import streamlit as st
import datetime
import os
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# --- 1. App Configuration and Title ---
st.set_page_config(page_title="AI Chore Assistant", page_icon="🏠")
st.title("🏠 AI Chore Assistant")
st.caption("Your household chore schedule, powered by Gemini!")

# --- 2. API Key and LLM Initialization ---
# This MUST be placed before any other function calls that use the LLM.
# It reads the secret key from Streamlit's secrets manager.
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]

# Initialize the Gemini LLM and Embeddings model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# --- 3. Define the Schedule's Start Date (Crucial for context) ---
# Make sure this is the correct start date for Week 1 of your schedule.
CYCLE_START_DATE = datetime.date(2025, 6, 30)

# --- 4. Function to Create the Vector Store (Knowledge Base) ---
@st.cache_resource
def get_vectorstore():
    try:
        loader = PyPDFLoader("chore_schedule.pdf")
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        # Use the Gemini embeddings model we initialized earlier
        vectorstore = FAISS.from_documents(documents=splits, embedding=embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error loading or creating vector store: {e}")
        return None

# --- 5. Initialize Vector Store and RAG Chain ---
vectorstore = get_vectorstore()
if not vectorstore:
    st.warning("Could not load the chore schedule. Please ensure chore_schedule.pdf is available.")
    st.stop()

# The RAG chain logic remains the same, it's modular!
contextualize_q_system_prompt = (
    "Given a chat history and the latest user question "
    "which might reference context in the chat history, "
    "formulate a standalone question which can be understood "
    "without the chat history. Do NOT answer the question, "
    "just reformulate it if needed and otherwise return it as is."
)
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [("system", contextualize_q_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
history_aware_retriever = create_history_aware_retriever(
    llm, vectorstore.as_retriever(), contextualize_q_prompt
)
qa_system_prompt = (
    "You are an assistant for answering questions about a household chore schedule. "
    "Use the following pieces of retrieved context to answer the question. "
    "Be concise and directly answer the user's question based on the schedule. "
    "If you don't know the answer, just say that you don't know. "
    "\n\n"
    "Context:\n{context}"
)
qa_prompt = ChatPromptTemplate.from_messages(
    [("system", qa_system_prompt), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)
question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- 6. Handle Chat History and Time Context ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

st.sidebar.header("📅 Time Context")
selected_date = st.sidebar.date_input(
    "Select a date to ask about:",
    value=datetime.date.today(),
    min_value=CYCLE_START_DATE,
)
delta_days = (selected_date - CYCLE_START_DATE).days
current_week_number = (delta_days // 7) % 4 + 1
current_day_name = selected_date.strftime("%A")
st.sidebar.success(f"Context: **{current_day_name}** in **Week {current_week_number}**")
time_context = (
    f"For context, the user is asking about {selected_date.strftime('%A, %B %d, %Y')}, "
    f"which is in Week {current_week_number} of the schedule. "
)

# --- 7. Display Chat History and Handle User Input with Streaming ---
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Who cleans the kitchen floor today?"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    contextual_prompt = time_context + prompt

    with st.chat_message("assistant"):
        # --- THIS IS THE CORRECTED LOGIC ---
        def stream_generator():
            # This variable is now local to the generator
            accumulated_response = ""
            for chunk in rag_chain.stream(
                {"input": contextual_prompt, "chat_history": st.session_state.chat_history}
            ):
                if "answer" in chunk:
                    content = chunk["answer"]
                    accumulated_response += content
                    yield content  # Yield content for real-time display
            
            # After the loop, return the complete message
            return accumulated_response

        # Use st.write_stream and assign its return value to full_response
        full_response = st.write_stream(stream_generator())
        # --- END OF CORRECTED LOGIC ---

    # Add the complete response to history after the stream is finished
    st.session_state.chat_history.append({"role": "assistant", "content": full_response})
