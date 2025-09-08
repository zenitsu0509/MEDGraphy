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
        print("Using Groq API key from Streamlit secrets.")
    else:
        groq_api_key = os.getenv("GROQ_API_KEY")
        print("Using Groq API key from local .env file.")

    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found. Please set it in your .env file or Streamlit secrets.")
    return groq.Groq(api_key=groq_api_key)

def get_rag_response(user_query: str, context: dict, groq_client) -> str:
    """Generates a response from the LLM based on the user's query and retrieved context."""

    context_str = json.dumps(context, indent=2)

    system_prompt = """
    You are a friendly and knowledgeable AI Drug Assistant.
    Use ONLY the provided structured context (which may include composition, raw use text, parsed uses (conditions), side_effects, ingredients, manufacturer, review percentages, and image_url).
    Guidelines:
    - If asked to explain: summarize composition (plain terms), primary uses (conditions), and major side effects.
    - If asked to compare or recommend and only one medicine is provided, say more data may be needed.
    - If review percentages exist, include a short sentiment summary.
    - If image_url present, mention that an image is available (do NOT fabricate description of the image content beyond name).
    - Do NOT invent medical advice beyond the context – if missing, state that.
    """

    human_prompt = f"""Structured Context:\n{context_str}\n\nUser Question: {user_query}\n\nReturn a concise, bullet-style answer when listing items."""

    try:
        chat_completion = groq_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": human_prompt},
            ],
            model="llama-3.1-8b-instant",
            temperature=0.3,
        )
        return chat_completion.choices[0].message.content
    except Exception as e:
        return f"An error occurred while generating the response: {e}"
