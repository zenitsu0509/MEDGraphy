import os
import streamlit as st
import groq
from dotenv import load_dotenv
import json

# Load environment variables for local development
load_dotenv()

@st.cache_resource
def get_groq_client():
    """Initializes and returns the Groq client."""
    # Prioritize Streamlit secrets, fall back to .env for local dev
    if hasattr(st, 'secrets') and "GROQ_API_KEY" in st.secrets:
        groq_api_key = st.secrets["GROQ_API_KEY"]
        st.sidebar.info("Using Groq API key from Streamlit secrets.")
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
        st.sidebar.info("Using Groq API key from local .env file.")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
    return groq.Groq(api_key=groq_api_key)

def get_rag_response(user_query: str, context: dict, groq_client) -> str:
    """Generates a response from the LLM based on the user's query and retrieved context."""

    context_str = json.dumps(context, indent=2)

    system_prompt = """
    You are a friendly and knowledgeable AI Drug Assistant. Your job is to help answer people's questions using 
    only the information provided in the context below. Don’t make things up or use outside knowledge. 
    Be clear, helpful, and explain your answers using the data you’ve been given. 
    If the context doesn’t have the answer, let the person know.
    """

    human_prompt = f"""Here's what you know:
    {context_str}

    Someone asked:
    "{user_query}"

    Using only the context above, give them a clear and helpful answer.
    """

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},  # use "user" here as it's expected by the API
            ],
            model="llama3-8b-8192",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the response: {e}"
