import streamlit as st
import os
from dotenv import load_dotenv
from graph_rag_query import GraphQueryEngine
from llm_chains import get_rag_response, get_groq_client
from streamlit_agraph import agraph, Node, Edge, Config

# --- CONFIGURATION ---
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MEDGraph", layout="wide")
st.title("⚕️ MEDGraph: Neo4j-Powered Drug Information App")

# --- HELPER FUNCTIONS ---
def display_results(result):
    if not result:
        st.warning("No results found.")
        return
    if isinstance(result, list):
        for record in result:
            st.json(record)
    else:
        st.json(result)

# --- INITIALIZE ENGINES ---
@st.cache_resource
def init_query_engine():
    return GraphQueryEngine()

@st.cache_resource
def init_groq_client():
    return get_groq_client()

engine = init_query_engine()
groq_client = init_groq_client()

# --- MAIN APP ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "❓ Full RAG Query", 
    "💊 Direct Lookup", 
    "🩺 Condition Lookup", 
    "⚠️ Interaction Check", 
    "🔎 Vector Search",
    "📊 Graph Visualization"
])

with tab1:
    st.header("❓ Full Graph RAG Query")
    st.write("The full RAG pipeline: Retrieve -> Augment -> Generate.")
    user_query = st.text_area("Enter your question:", "What are the side effects of a medicine for headaches?", height=100, key="rag_query")
    if st.button("Run RAG Query", type="primary"):
        with st.spinner("Running Full RAG pipeline..."):
            rag_context = engine.retrieve_context_for_rag(user_query)
            if not rag_context or not rag_context.get("context"):
                st.error("Could not find a relevant medicine or context for your query.")
            else:
                st.success(f"Found most relevant medicine: **{rag_context['medicine_found']}**")
                response = get_rag_response(user_query, rag_context['context'], groq_client)
                st.subheader("LLM Response:")
                st.markdown(response)
                with st.expander("View Raw Context from Graph"):
                    st.json(rag_context['context'])

with tab2:
    st.header("💊 Direct Medicine Lookup")
    st.write("Finds uses and side effects for a specific medicine.")
    med_name_direct = st.text_input("Enter Medicine Name:", "Paracetamol", key="direct")
    if st.button("Find Details"):
        with st.spinner("Looking up medicine..."):
            result = engine.direct_lookup(med_name_direct)
            display_results(result)

with tab3:
    st.header("🩺 Reverse Lookup by Condition")
    st.write("Finds medicines that treat a specific condition.")
    condition_name = st.text_input("Enter Condition Name:", "Hypertension", key="reverse")
    if st.button("Find Medicines"):
        with st.spinner("Finding medicines for condition..."):
            result = engine.reverse_lookup(condition_name)
            st.write(result)

with tab4:
    st.header("⚠️ Potential Interaction Check")
    st.write("Finds other medicines that share the same active ingredient.")
    med_name_interact = st.text_input("Enter Medicine Name:", "Aspirin", key="interaction")
    if st.button("Check for Interactions"):
        with st.spinner("Checking for potential interactions..."):
            result = engine.check_interactions(med_name_interact)
            display_results(result)

with tab5:
    st.header("🔎 Vector Similarity Search")
    st.write("Uses embeddings to find semantically similar medicines.")
    query_text = st.text_input("Enter a search query:", "medicine for joint pain", key="vector")
    if st.button("Search by Similarity"):
        with st.spinner("Performing vector search..."):
            result = engine.vector_similarity_search(query_text)
            display_results(result)

with tab6:
    st.header("📊 Graph Visualization")
    st.write("Generates an interactive graph diagram for a medicine.")
    vis_medicine = st.text_input("Enter a medicine name to visualize:", "Aspirin", key="vis_med")

    if st.button("Generate Visualization", type="primary"):
        if vis_medicine:
            with st.spinner("Generating graph from Neo4j data..."):
                results = engine.get_graph_for_visualization(vis_medicine)
                
                if results:
                    nodes = []
                    edges = []
                    node_ids = set()

                    for record in results:
                        source_node, rel, target_node = record['m'], record['r'], record['n']
                        
                        source_id = source_node.element_id
                        source_label = list(source_node.labels)[0]
                        source_name = source_node._properties.get('name', 'N/A')
                        
                        target_id = target_node.element_id
                        target_label = list(target_node.labels)[0]
                        target_name = target_node._properties.get('name', 'N/A')

                        if source_id not in node_ids:
                            nodes.append(Node(id=source_id, label=source_name, shape="dot", size=25, font={"size": 20}, color="#FF9900", title=source_label))
                            node_ids.add(source_id)

                        if target_id not in node_ids:
                            color_map = {"Condition": "#FFC0CB", "SideEffect": "#ADD8E6", "ActiveIngredient": "#90EE90"}
                            nodes.append(Node(id=target_id, label=target_name, shape="box", color=color_map.get(target_label, "#E0E0E0"), title=target_label))
                            node_ids.add(target_id)
                        
                        rel_type = rel.type.replace("_", " ").title()
                        edges.append(Edge(source=source_id, target=target_id, label=rel_type))
                    
                    config = Config(width=1100, height=700, directed=True, physics=True, hierarchical=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                    agraph(nodes=nodes, edges=edges, config=config)
                else:
                    st.warning(f"Could not find information to visualize for '{vis_medicine}'.")
