import streamlit as st
import random
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification

# Set the page configuration for a wider layout
st.set_page_config(page_title="FinAssist AI", layout="wide")

# --- Helper Functions for Data Simulation ---

def get_mock_transactions():
    """Generates a list of mock financial transactions for  demonstration."""
    transactions = [
        {'date': '2023-10-01', 'amount': 50.00, 'type': 'expense', 'category': 'Groceries'},
        {'date': '2023-10-02', 'amount': 15.75, 'type': 'expense', 'category': 'Dining Out'},
        {'date': '2023-10-03', 'amount': 1500.00, 'type': 'income', 'category': 'Salary'},
        {'date': '2023-10-05', 'amount': 75.20, 'type': 'expense', 'category': 'Shopping'},
        {'date': '2023-10-06', 'amount': 25.00, 'type': 'expense', 'category': 'Dining Out'},
        {'date': '2023-10-07', 'amount': 45.00, 'type': 'expense', 'category': 'Groceries'},
        {'date': '2023-10-08', 'amount': 100.00, 'type': 'expense', 'category': 'Entertainment'},
        {'date': '2023-10-10', 'amount': 30.00, 'type': 'expense', 'category': 'Dining Out'},
    ]
    return transactions

def generate_budget_summary(transactions, user_profile):
    """Analyzes transactions and generates a data-driven summary."""
    total_income = sum(t['amount'] for t in transactions if t['type'] == 'income')
    total_expenses = sum(t['amount'] for t in transactions if t['type'] == 'expense')
    net_flow = total_income - total_expenses

    # Identify the top spending category
    spending_categories = {}
    for t in transactions:
        if t['type'] == 'expense':
            spending_categories[t['category']] = spending_categories.get(t['category'], 0) + t['amount']

    largest_expense_category = max(spending_categories, key=spending_categories.get) if spending_categories else "N/A"

    # Simulate a demographic-aware response
    if user_profile['demographic'] == 'Student':
        response_intro = "Hey there! Let's take a look at your budget. "
    elif user_profile['demographic'] == 'Professional':
        response_intro = "Hello! Here's a quick summary of your financial status. "
    else:
        response_intro = "Hi! I've put together a summary of your recent transactions. "

    response = (
        f"{response_intro}This month, you've had a total income of ${total_income:.2f} "
        f"and total expenses of ${total_expenses:.2f}. "
        f"This leaves you with a net flow of ${net_flow:.2f}. "
        f"Your largest spending category was {largest_expense_category}."
    )
    return response

def get_spending_insights(transactions, user_profile):
    """Analyzes transactions to provide actionable insights."""
    groceries_spend = sum(t['amount'] for t in transactions if t['category'] == 'Groceries')
    dining_out_spend = sum(t['amount'] for t in transactions if t['category'] == 'Dining Out')

    # Simulate a demographic-aware response
    if user_profile['demographic'] == 'Student':
        response = (
            f"Looking at your spending, you've spent ${groceries_spend:.2f} on groceries "
            f"and ${dining_out_spend:.2f} on dining out. A great way to save money is to try cooking more at home!"
        )
    elif user_profile['demographic'] == 'Professional':
        response = (
            f"Your spending shows ${groceries_spend:.2f} on groceries and ${dining_out_spend:.2f} on dining out. "
            f"Consider meal-prepping to optimize your expenses and free up some cash for savings or investments."
        )
    else:
        response = (
            f"Based on your transactions, you spent ${groceries_spend:.2f} on groceries and ${dining_out_spend:.2f} on dining out. "
            f"These categories are a good place to look for opportunities to save."
        )
    return response

# --- Model Loading with Caching ---

@st.cache_resource
def load_sentiment_model():
    """
    Loads the specified sentiment analysis model.
    Note: This model is for sentiment analysis, not text generation.
    It is loaded here to show the requested model is present.
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        model = AutoModelForSequenceClassification.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment-latest")
        sentiment_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
        st.info("Sentiment analysis model loaded successfully, but it's not a generative model. The app will use simulated responses.")
        return sentiment_pipe
    except Exception as e:
        st.error(f"Failed to load the model. Please check your internet connection or try again later. Error: {e}")
        return None

# --- Session State Initialization ---

if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'user_profile' not in st.session_state:
    st.session_state.user_profile = {'demographic': 'Student', 'age': '20'}
if 'mock_transactions' not in st.session_state:
    st.session_state.mock_transactions = get_mock_transactions()

# --- Streamlit UI and Logic ---

st.title("FinAssist AI: Your Personal Financial Guide")
st.markdown("Ask me anything about your finances, or use the sidebar for quick actions.")

# Sidebar for User Profile and Quick Actions
with st.sidebar:
    st.header("Your Profile")
    demographic = st.radio("Select your demographic:", ['Student', 'Professional', 'Retiree'])
    age = st.text_input("Enter your age:", value='20')

    st.session_state.user_profile['demographic'] = demographic
    st.session_state.user_profile['age'] = age

    st.header("Quick Actions")
    if st.button("Generate Budget Summary", help="Get an AI-generated summary of your spending."):
        with st.spinner("Generating summary..."):
            response = generate_budget_summary(st.session_state.mock_transactions, st.session_state.user_profile)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

    if st.button("Get Spending Insights", help="Receive insights and suggestions on your spending habits."):
        with st.spinner("Analyzing spending..."):
            response = get_spending_insights(st.session_state.mock_transactions, st.session_state.user_profile)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.rerun()

# Load the sentiment model (for demonstration)
sentiment_pipe = load_sentiment_model()

# Display chat messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input text box
if prompt := st.chat_input("Ask a financial question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            # --- Simulated Conversational Logic (for the hackathon) ---
            # A real generative model would be used here.
            # This is a placeholder for your future IBM model integration.

            # Simple keyword-based response for the hackathon prototype
            if "investment" in prompt.lower():
                response = "Investing is a great way to grow your wealth over time! I can provide general advice, but for specific strategies, it's best to consult a professional."
            elif "save" in prompt.lower() or "savings" in prompt.lower():
                response = "Saving money is a key part of financial health. A good strategy is to set a specific savings goal each month and treat it like a bill."
            elif "tax" in prompt.lower() or "taxes" in prompt.lower():
                response = "Taxes can be complex. While I can offer basic information, please consult a tax professional for detailed advice on your specific situation."
            else:
                response = f"I am a simulated assistant. I can help you with budget summaries and spending insights. To truly answer this, you would need to integrate a generative model that can handle queries like: '{prompt}'"

            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
