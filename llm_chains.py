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
    """Generates a response from the LLM based on the user query and retrieved context."""
    
    context_str = json.dumps(context, indent=2)

    system_prompt = """
    You are an expert AI Drug Assistant. Your task is to answer the user's question based *only* on the
    structured context provided below. Do not use any external knowledge.
    Present the information clearly and explain your reasoning based on the provided data.
    If the context does not contain the answer, say that you cannot answer based on the information available.
    """
    
    human_prompt = f"""CONTEXT:
    {context_str}

    USER'S QUESTION:
    {user_query}

    Based on the context, please answer the user's question."""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            model="llama3-8b-8192",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the response: {e}"