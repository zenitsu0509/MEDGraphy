import streamlit as st
import faiss
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from neo4j import GraphDatabase
from dotenv import load_dotenv 
import os

load_dotenv()

st.set_page_config(page_title="Medicine GraphRAG AI", layout="wide")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USERNAME")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

FAISS_INDEX_PATH = "db/medicine_embeddings.index"
METADATA_PATH = "db/metadata.json"

EMBED_MODEL = "BAAI/bge-large-en-v1.5"
LLM_MODEL = "openai/gpt-oss-120b"       


# ---------------------------------------------------------
#           LOAD MODELS & DATABASES (CACHED)
# ---------------------------------------------------------

@st.cache_resource
def load_faiss():
    return faiss.read_index(FAISS_INDEX_PATH)

@st.cache_resource
def load_metadata():
    with open(METADATA_PATH, "r") as f:
        return json.load(f)

@st.cache_resource
def load_embedder():
    return SentenceTransformer(EMBED_MODEL)

@st.cache_resource
def load_llm():
    return Groq(api_key=GROQ_API_KEY)

@st.cache_resource
def load_neo4j():
    driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        max_connection_lifetime=3600,
        max_connection_pool_size=50,
        connection_acquisition_timeout=120
    )
    # Test the connection
    driver.verify_connectivity()
    return driver


faiss_index = load_faiss()
metadata = load_metadata()
embedder = load_embedder()
groq_client = load_llm()

# Load Neo4j with error handling
try:
    neo4j_driver = load_neo4j()
    st.sidebar.success("✅ Connected to Neo4j")
except Exception as e:
    st.sidebar.error(f"❌ Neo4j Connection Failed: {str(e)}")
    st.error(f"Database connection error. Please check your Neo4j credentials and connection: {str(e)}")
    neo4j_driver = None


# ---------------------------------------------------------
#       GRAPH EXPANSION — FETCH RELATED NODES
# ---------------------------------------------------------

def get_graph_info(drug_name):
    if neo4j_driver is None:
        return {}
    
    query = """
    MATCH (d:Drug {name: $name})-[r]->(n)
    RETURN type(r) AS relation, n.name AS value
    LIMIT 200
    """
    try:
        with neo4j_driver.session(database=NEO4J_DATABASE) as session:
            result = session.run(query, name=drug_name).data()
    except Exception as e:
        st.warning(f"Could not fetch graph data for {drug_name}: {str(e)}")
        return {}

    graph_dict = {}
    for row in result:
        relation = row["relation"]
        value = row["value"]
        graph_dict.setdefault(relation, []).append(value)

    return graph_dict


# ---------------------------------------------------------
#            SEMANTIC SEARCH (FAISS)
# ---------------------------------------------------------

def semantic_search(query, top_k=5):
    query_emb = embedder.encode(query).astype("float32")

    distances, indices = faiss_index.search(
        np.array([query_emb]), top_k
    )

    results = []
    for idx in indices[0]:
        results.append(metadata[idx])
    return results


# ---------------------------------------------------------
#            LLM ANSWER USING GROQ
# ---------------------------------------------------------

def answer_with_groq(query, retrieved, graph_info):
    system_prompt = """
    You are a medical question answering assistant.
    You must:
    - Use the retrieved medicine information.
    - Use graph relations (substitutes, side effects, uses, classes).
    - Never hallucinate facts.
    - Respond using ONLY provided context.
    """

    # Build context from FAISS metadata
    text_block = ""
    for item in retrieved:
        text_block += f"""
        Medicine: {item['name']}
        Uses: {item['uses']}
        Side Effects: {item['side_effects']}
        Manufacturer: {item['manufacturer']}
        """

    # Add graph info
    graph_text = ""
    for medicine, relations in graph_info.items():
        graph_text += f"\nGraph Data for {medicine}:\n"
        for rel, vals in relations.items():
            graph_text += f"{rel}: {', '.join(vals)}\n"

    full_prompt = f"""
    {system_prompt}

    User Query:
    {query}

    Retrieved Medicine Data:
    {text_block}

    Graph Knowledge:
    {graph_text}

    Final Answer:
    """

    response = groq_client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": full_prompt}],
        temperature=0.2,
    )

    return response.choices[0].message.content


# ---------------------------------------------------------
#                     STREAMLIT UI
# ---------------------------------------------------------

st.title("💊 Medicine GraphRAG AI")
st.write("Semantic Search + Graph DB + LLM reasoning using Groq GPT-OSS-20B")

query = st.text_input("Enter your medical query:", placeholder="e.g., best medicine for acidity")

if st.button("Search"):
    if not query.strip():
        st.warning("Please enter a query.")
    else:
        st.info("🔍 Searching medicines via FAISS semantic search...")
        results = semantic_search(query)

        st.write("### 🔬 Top Relevant Medicines")
        for r in results:
            st.write(f"**{r['name']}** — {r['uses']}")

        st.info("🧠 Expanding Knowledge Graph for all retrieved medicines...")

        graph_dict = {}
        for r in results:
            graph_dict[r["name"]] = get_graph_info(r["name"])

        st.write("### 🧬 Graph Relations Found")
        st.json(graph_dict)

        st.success("🤖 Generating LLM Answer...")
        answer = answer_with_groq(query, results, graph_dict)

        st.write("### 🩺 Final Answer")
        st.success(answer)
