import os
import streamlit as st
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.graph_qa.cypher import GraphCypherQAChain

from graph_db import graph # Import the LangChain graph object

# Determine Groq API key based on the environment
if 'GROQ_API_KEY' in st.secrets:
    groq_api_key = st.secrets["GROQ_API_KEY"]
    print("Using Groq API key from Streamlit secrets.")
else:
    groq_api_key = os.getenv("GROQ_API_KEY")
    print("Using Groq API key from local .env file.")

# Initialize the LLM with Groq
llm = ChatGroq(temperature=0, model_name="llama3-70b-8192", groq_api_key=groq_api_key)

# --- 1. AI Drug Assistant (Explanation) ---
assistant_template = """
You are an AI Drug Assistant. Your goal is to provide a clear and concise explanation of a medicine
based on the structured data provided. Do not mention that the data is from a graph. Just present the information.

Context from database:
{context}

Question:
{question}

Answer in a user-friendly format:
"""
assistant_prompt = PromptTemplate(template=assistant_template, input_variables=["context", "question"])
assistant_chain = LLMChain(llm=llm, prompt=assistant_prompt)


# --- 2. Prescription Justifier ---
justifier_template = """
You are a pharmacology expert. Your task is to justify a given prescription based on a patient's diagnosis
and information from a drug database.

Diagnosis: {diagnosis}
Prescription: {prescription}

Here is the context retrieved from the drug graph database. Each entry shows a medicine from the prescription and any
direct links it has to either the diagnosis or a side effect of another prescribed drug.
- 'treats_diagnosis': Shows if the medicine is a known treatment for the patient's diagnosis.
- 'treats_side_effect_of': Shows if the medicine is used to treat a side effect caused by another drug in the prescription.

Context:
{context}

Based on this context, provide a step-by-step justification for each medicine in the prescription.
If a medicine does not have a clear link, state that its purpose is not clear from the available data.

Example Output:
- **Avastin**: Justified. It is a primary treatment for Lung cancer.
- **Paracetamol**: Justified. It is used to manage Fever, which is a known side effect of Avastin.

Justification:
"""
justifier_prompt = PromptTemplate(
    template=justifier_template, 
    input_variables=["diagnosis", "prescription", "context"]
)
justifier_chain = LLMChain(llm=llm, prompt=justifier_prompt)


# --- 5. Zero-Shot Question Answering over Graph ---
# This chain can translate a natural language question into a Cypher query
# We explicitly provide the schema to help the LLM generate correct queries.
cypher_qa_chain = GraphCypherQAChain.from_llm(
    llm,
    graph=graph,
    verbose=True, # Set to False in production for cleaner output
    # WARNING: This allows the LLM to execute arbitrary queries on your database.
    # Ensure your database connection has limited permissions to prevent data corruption or loss.
    # Only use this if you understand and accept the risks.
    # See https://python.langchain.com/docs/security for more information.
    allow_dangerous_requests=True,
    cypher_prompt=PromptTemplate(
        input_variables={'question', 'schema'},
        template='''You are a Neo4j expert. Given an input question, create a Cypher query to answer the question.
        Do not use any undeclared variables.

        Schema:
        {schema}

        Question: {question}
        Cypher Query:
        '''
    )
)