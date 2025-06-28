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


# --- 5. Enhanced Graph RAG Question Answering ---

# Chain 1: Enhanced Cypher Query Generation
cypher_prompt_template = """You are a Neo4j expert who writes Cypher queries for a medical knowledge graph.
The graph contains Medicine, Disease, Composition, Manufacturer, and SideEffect nodes with relationships:
- Medicine-[:USED_FOR]->Disease
- Medicine-[:HAS_COMPOSITION]->Composition  
- Medicine-[:MANUFACTURED_BY]->Manufacturer
- Medicine-[:HAS_SIDE_EFFECT]->SideEffect

IMPORTANT: 
- Use toLower() for case-insensitive matching
- Use CONTAINS for partial matching of disease names
- For lung cancer queries, search for diseases containing 'lung cancer' or 'lung'
- Always return relevant node properties

Schema:
{schema}

Question: {question}

Output ONLY the Cypher query:"""

cypher_prompt = PromptTemplate(
    template=cypher_prompt_template,
    input_variables=["schema", "question"]
)

cypher_generation_chain = LLMChain(llm=llm, prompt=cypher_prompt)

# Chain 2: Enhanced Answer Generation
qa_template = """You are a medical AI assistant that provides comprehensive answers about medicines and diseases.
Use the provided graph database results to give detailed, helpful responses.

Question: {question}

Graph Database Results: {context}

Provide a detailed answer that includes:
1. Direct answer to the question
2. Medicine names and their relevant details
3. Any additional relevant information from the data

If no relevant data is found, suggest alternative search terms or indicate that the information might not be in the database.

Answer:"""

qa_prompt = PromptTemplate(
    template=qa_template, 
    input_variables=["question", "context"]
)

qa_chain = LLMChain(llm=llm, prompt=qa_prompt)

# Chain 3: Graph RAG Chain for Complex Queries
graph_rag_template = """You are an expert medical information assistant with access to a comprehensive medicine knowledge graph.

Question: {question}

Available Data from Knowledge Graph:
{graph_results}

Search Strategies Used: {strategies_used}

Diseases Extracted from Query: {extracted_diseases}

Based on the comprehensive search results above, provide a detailed and informative answer. 

If medicines were found:
- List the medicines with their key details
- Explain what conditions they treat
- Include composition and manufacturer information when available
- Mention any relevant side effects if asked

If no medicines were found:
- Explain that no matches were found in the current database
- Suggest alternative search terms
- Recommend consulting with healthcare professionals

Answer:"""

graph_rag_prompt = PromptTemplate(
    template=graph_rag_template,
    input_variables=["question", "graph_results", "strategies_used", "extracted_diseases"]
)

graph_rag_chain = LLMChain(llm=llm, prompt=graph_rag_prompt)