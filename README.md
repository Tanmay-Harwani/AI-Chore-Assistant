# AI Chore Assistant


A smart household chore management system that uses AI to answer questions about chore schedules and track rotating responsibilities.

Features

- AI-powered chat - Ask questions about chores, schedules, and assignments
- Automatic rotation tracking - Calculates current week and person assignments
- PDF knowledge base - Loads chore schedule from PDF file
- Real-time updates - Shows current date, week
- Quick actions - Pre-built questions for common queries

Setup

1. Install Dependencies


	pip install -r requirements.txt


2. Add Your Google API Key


Option A: Environment Variable


	export GOOGLE_API_KEY="your-gemini-api-key-here"

Option B: Streamlit Secrets (for deployment)
Create .streamlit/secrets.toml:


	GOOGLE_API_KEY = "your-gemini-api-key-here"

3. Add Your Chore Schedule


Place your chore schedule PDF file as chore_schedule.pdf in the project root.

4. Run the App

	streamlit run app.py

# Usage

1. Chat Interface - Type questions about chores and schedules
2. Quick Questions - Use sidebar buttons for common queries
3. Current Info - Check sidebar for today's assignments
4. Clear Chat - Reset conversation anytime

Example Questions

- "What are this week's chore assignments?"
- "Who takes out kitchen trash today?"
- "What does Tanmay need to do this week?"

File Structure

	├── app.py                 # Main application
	├── requirements.txt       # Python dependencies
	├── chore_schedule.pdf     # Your chore schedule (add this)
	└── README.md             # This file

Requirements

- Python 3.8+
- Google Gemini API key
- PDF file with chore schedule

Technologies Used

- Streamlit - Web interface
- Google Gemini - AI responses
- LangChain - AI framework
- HuggingFace - Text embeddings
- NumPy - Vector calculations

Configuration


The app uses a 4-week rotation cycle. Kitchen trash runs on Monday, Thursday, and Sunday with rotation among 4 household members: Tanmay, Soham, Pranjul, and Ishika.

To modify these settings, update the relevant functions in app.py.
