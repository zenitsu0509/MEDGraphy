import streamlit as st
from dotenv import load_dotenv
import llm_chains
from graph_db import Neo4jConnection
from streamlit_agraph import agraph, Node, Edge, Config
from graph_rag_query import GraphRAGQueryEngine
import pandas as pd

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="GraphRAG Medical Assistant", layout="wide")
st.title("⚕️ Enhanced GraphRAG-Powered AI Medical Assistant")
st.write("Advanced medical knowledge graph with semantic search capabilities.")

# --- Initialize Connections (cached for performance) ---
@st.cache_resource
def get_neo4j_connection():
    return Neo4jConnection()

@st.cache_resource
def get_graph_rag_engine():
    return GraphRAGQueryEngine()

db = get_neo4j_connection()
rag_engine = get_graph_rag_engine()

# --- UI Tabs for each feature ---
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "1. Enhanced RAG Search",
    "2. AI Drug Assistant", 
    "3. Prescription Justifier",
    "4. Symptom to Drug",
    "5. Visual Explainer",
    "6. Data Management"
])

# =====================================================================================
# TAB 1: Enhanced Graph RAG Search
# =====================================================================================
with tab1:
    st.header("🔍 Enhanced Graph RAG Search")
    st.write("Ask natural language questions about medicines and diseases with advanced semantic search.")
    
    # Sample questions
    with st.expander("💡 Sample Questions"):
        st.write("""
        - Which medicines treat lung cancer?
        - What are the side effects of Advacan?
        - Medicines for breast cancer made by Roche
        - Show me all cancer treatments
        - What does Avastin treat?
        - Medicines that cause headache
        """)
    
    question = st.text_area(
        "Your Question:", 
        value="Which medicines treat lung cancer?",
        height=100,
        help="Ask any question about medicines, diseases, manufacturers, or side effects"
    )

    if st.button("🚀 Search with Enhanced RAG", type="primary"):
        if question:
            with st.spinner("Searching knowledge graph with advanced semantic matching..."):
                try:
                    # Use the enhanced RAG engine
                    rag_results = rag_engine.answer_natural_language_query(question)
                    
                    if rag_results['medicines_found']:
                        st.success(f"✅ Found {rag_results['total_medicines']} medicines!")
                        
                        # Display results in a nice format
                        for i, medicine in enumerate(rag_results['medicines_found'], 1):
                            with st.container():
                                col1, col2 = st.columns([3, 1])
                                with col1:
                                    st.markdown(f"**{i}. {medicine['medicine']}**")
                                    st.write(f"🎯 Treats: {medicine['disease_treated']}")
                                    if medicine.get('composition'):
                                        st.write(f"💊 Composition: {medicine['composition']}")
                                    if medicine.get('manufacturer'):
                                        st.write(f"🏭 Manufacturer: {medicine['manufacturer']}")
                                
                                with col2:
                                    if st.button(f"Details", key=f"detail_{i}"):
                                        detailed_info = rag_engine.get_comprehensive_medicine_info(medicine['medicine'])
                                        st.json(detailed_info)
                                
                                st.divider()
                        
                        # Show search details
                        with st.expander("🔍 Search Details"):
                            st.write(f"**Original Query:** {rag_results['original_query']}")
                            st.write(f"**Normalized Query:** {rag_results['normalized_query']}")
                            st.write(f"**Extracted Diseases:** {rag_results['extracted_diseases']}")
                            st.write(f"**Search Strategies:** {rag_results['search_strategies_used']}")
                    
                    else:
                        st.warning("⚠️ No medicines found for your query.")
                        st.info("💡 Try different terms like 'cancer', 'pain', 'infection', or specific medicine names.")
                        
                        # Show what was searched
                        with st.expander("🔍 Search Analysis"):
                            st.json(rag_results)
                
                except Exception as e:
                    st.error(f"❌ Search failed: {str(e)}")
                    st.info("Try rephrasing your question or check if the graph database is properly loaded.")

# =====================================================================================
# TAB 2: AI Drug Assistant (Explain, Compare, Recommend)
# =====================================================================================
with tab2:
    st.header("AI Drug Assistant")
    st.write("Explains medicine details or recommends drugs for a condition.")
    
    assist_mode = st.radio("Select mode:", ("Explain a Medicine", "Recommend a Medicine"), horizontal=True, label_visibility="collapsed")

    if assist_mode == "Explain a Medicine":
        medicine_name = st.text_input("Enter a medicine name (e.g., Avastin, Paracetamol):", key="explain_med")
        if st.button("Explain Medicine", type="primary"):
            if medicine_name:
                with st.spinner("Fetching data and generating explanation..."):
                    query = """
                    MATCH (m:Medicine {name: $med_name})
                    OPTIONAL MATCH (m)-[:HAS_COMPOSITION]->(c:Composition)
                    OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
                    OPTIONAL MATCH (m)-[:USED_FOR]->(d:Disease)
                    OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
                    RETURN m.name as medicine, c.name as composition, man.name as manufacturer,
                           collect(DISTINCT d.name) as uses,
                           collect(DISTINCT s.name) as side_effects
                    """
                    result = db.query(query, parameters={"med_name": medicine_name})
                    if result and result[0]['medicine']:
                        context = result[0]
                        question = f"Explain what {medicine_name} is, what it's used for, and its common side effects."
                        response = llm_chains.assistant_chain.invoke({"context": dict(context), "question": question})
                        st.success("Explanation:")
                        st.markdown(response['text'])
                        with st.expander("View Raw Data from Graph"):
                            st.json(dict(context))
                    else:
                        st.error(f"Medicine '{medicine_name}' not found in the database.")

    elif assist_mode == "Recommend a Medicine":
        disease_name = st.text_input("Enter a disease or condition (e.g., Lung cancer, Pain):", key="rec_disease")
        if st.button("Find Recommendations", type="primary"):
            if disease_name:
                with st.spinner("Searching for recommendations..."):
                    query = """
                    MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
                    WHERE toLower(d.name) CONTAINS toLower($disease_name)
                    RETURN m.name as medicine ORDER BY medicine
                    LIMIT 15
                    """
                    result = db.query(query, parameters={"disease_name": disease_name})
                    if result:
                        st.success(f"Found {len(result)} medicines for '{disease_name}':")
                        for row in result:
                            st.markdown(f"- **{row['medicine']}**")
                    else:
                        st.warning(f"No medicines found for '{disease_name}'.")

# =====================================================================================
# TAB 3: Prescription Justifier
# =====================================================================================
with tab3:
    st.header("Prescription Justifier")
    st.write("Analyzes a prescription against a diagnosis to explain why each drug is included.")
    
    diagnosis = st.text_input("Enter Diagnosis:", value="Lung cancer")
    prescription = st.text_area("Enter Prescription (one drug per line):", value="Avastin\nParacetamol", height=100)

    if st.button("Justify Prescription", type="primary"):
        if diagnosis and prescription:
            with st.spinner("Analyzing prescription with the knowledge graph..."):
                meds_list = [med.strip() for med in prescription.split('\n') if med.strip()]
                
                query = """
                UNWIND $meds as med_name
                MATCH (m:Medicine {name: med_name})
                // Path 1: Direct treatment for diagnosis
                OPTIONAL MATCH (m)-[:USED_FOR]->(d:Disease)
                WHERE toLower(d.name) = toLower($diag)
                // Path 2: Treats side effect of another prescribed drug
                OPTIONAL MATCH (m)-[:USED_FOR]->(disease_treated)
                MATCH (other_med:Medicine)-[:HAS_SIDE_EFFECT]->(se:SideEffect)
                WHERE other_med.name IN $meds AND toLower(disease_treated.name) = toLower(se.name) AND m <> other_med
                RETURN m.name as medicine, 
                       d.name as treats_diagnosis,
                       COLLECT(DISTINCT {treated_side_effect: se.name, caused_by: other_med.name}) as treats_side_effect_of
                """
                
                result = db.query(query, parameters={"meds": meds_list, "diag": diagnosis})
                
                if result:
                    context = [dict(row) for row in result]
                    response = llm_chains.justifier_chain.invoke({
                        "diagnosis": diagnosis,
                        "prescription": ", ".join(meds_list),
                        "context": context
                    })
                    st.success("Justification Analysis:")
                    st.markdown(response['text'])
                    with st.expander("View Justification Data from Graph"):
                        st.json(context)
                else:
                    st.error("Could not retrieve information for the given prescription.")

# =====================================================================================
# TAB 4: Symptom-to-Drug Recommender
# =====================================================================================
with tab4:
    st.header("Symptom-to-Drug Finder")
    st.write("Finds which medicines might be causing a set of symptoms (side effects).")
    symptoms = st.text_input("Enter symptoms/side effects (comma-separated):", value="Nosebleeds, Fatigue")

    if st.button("Find Potential Drugs", type="primary"):
        if symptoms:
            with st.spinner("Searching for drugs based on side effects..."):
                symptoms_list = [s.strip().lower() for s in symptoms.split(',')]
                
                query = """
                MATCH (m:Medicine)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
                WHERE toLower(s.name) IN $symptoms
                WITH m, count(s) as matched_symptoms
                // Ensure the medicine has all the specified side effects
                WHERE matched_symptoms = size($symptoms)
                RETURN m.name as medicine
                """
                result = db.query(query, parameters={"symptoms": symptoms_list})
                if result:
                    st.success("Medicines that might cause these symptoms:")
                    for row in result:
                        st.markdown(f"- **{row['medicine']}**")
                else:
                    st.warning("No single medicine found that causes all the specified symptoms.")

# =====================================================================================
# TAB 5: RAG-Based Visual Explainer
# =====================================================================================
with tab5:
    st.header("Medicine Visual Explainer")
    st.write("Generates an interactive graph diagram for a medicine.")
    vis_medicine = st.text_input("Enter a medicine name to visualize:", value="Avastin")

    if st.button("Generate Visualization", type="primary"):
        if vis_medicine:
            with st.spinner("Generating graph from Neo4j data..."):
                query = """
                MATCH path = (m:Medicine {name: $med_name})-[r]-(n)
                RETURN m, r, n
                """
                results = db.query(query, parameters={"med_name": vis_medicine})
                
                if results:
                    nodes = []
                    edges = []
                    node_ids = set()

                    for record in results:
                        source_node, rel, target_node = record['m'], record['r'], record['n']
                        
                        source_id, source_label = source_node.element_id, list(source_node.labels)[0]
                        source_name = source_node._properties.get('name', 'N/A')
                        
                        target_id, target_label = target_node.element_id, list(target_node.labels)[0]
                        target_name = target_node._properties.get('name', 'N/A')

                        if source_id not in node_ids:
                            nodes.append(Node(id=source_id, label=source_name, shape="dot", size=25, font={"size": 20}, color="#FF9900", title=source_label))
                            node_ids.add(source_id)

                        if target_id not in node_ids:
                            color_map = {"Disease": "#FFC0CB", "SideEffect": "#ADD8E6", "Composition": "#90EE90", "Manufacturer": "#D3D3D3"}
                            nodes.append(Node(id=target_id, label=target_name, shape="box", color=color_map.get(target_label, "#E0E0E0"), title=target_label))
                            node_ids.add(target_id)
                        
                        rel_type = rel.type.replace("_", " ").title()
                        edges.append(Edge(source=source_id, target=target_id, label=rel_type))
                    
                    config = Config(width=1100, height=700, directed=True, physics=True, hierarchical=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                    agraph(nodes=nodes, edges=edges, config=config)

# =====================================================================================
# TAB 6: Data Management
# =====================================================================================
with tab6:
    st.header("📊 Data Management & Diagnostics")
    st.write("Manage and diagnose the knowledge graph data.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Reload Graph Data")
        st.write("Reload the CSV data into the Neo4j graph with enhanced parsing.")
        
        if st.button("🔄 Reload Data from CSV", type="secondary"):
            with st.spinner("Reloading graph data..."):
                try:
                    from graph_rag_loader import GraphRAGLoader
                    loader = GraphRAGLoader()
                    loader.load_csv_to_graph("Medicine_Details.csv")
                    st.success("✅ Graph data reloaded successfully!")
                    st.info("🔄 Please refresh the page to see updated results.")
                except Exception as e:
                    st.error(f"❌ Failed to reload data: {str(e)}")
    
    with col2:
        st.subheader("🔍 Debug Lung Cancer Data")
        st.write("Check if lung cancer data is properly loaded in the graph.")
        
        if st.button("🔍 Debug Lung Cancer", type="secondary"):
            with st.spinner("Debugging lung cancer data..."):
                try:
                    # Debug lung cancer data
                    debug_queries = [
                        "MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'lung' RETURN d.name ORDER BY d.name",
                        "MATCH (m:Medicine)-[:USED_FOR]->(d:Disease) WHERE toLower(d.name) CONTAINS 'lung' RETURN m.name, d.name"
                    ]
                    
                    for i, query in enumerate(debug_queries, 1):
                        st.write(f"**Query {i}:**")
                        st.code(query, language='cypher')
                        results = db.query(query)
                        if results:
                            st.write(f"Found {len(results)} results:")
                            for result in results[:10]:
                                st.write(f"  - {result}")
                        else:
                            st.write("No results found")
                        st.divider()
                
                except Exception as e:
                    st.error(f"❌ Debug failed: {str(e)}")
    
    st.subheader("📈 Graph Statistics")
    if st.button("📊 Get Graph Stats", type="secondary"):
        with st.spinner("Fetching graph statistics..."):
            try:
                stats_queries = {
                    "Total Medicines": "MATCH (m:Medicine) RETURN count(m) as count",
                    "Total Diseases": "MATCH (d:Disease) RETURN count(d) as count", 
                    "Total Manufacturers": "MATCH (man:Manufacturer) RETURN count(man) as count",
                    "Total Side Effects": "MATCH (s:SideEffect) RETURN count(s) as count",
                    "Lung Cancer Medicines": "MATCH (m:Medicine)-[:USED_FOR]->(d:Disease) WHERE toLower(d.name) CONTAINS 'lung' RETURN count(DISTINCT m) as count"
                }
                
                stats_results = {}
                for stat_name, query in stats_queries.items():
                    result = db.query(query)
                    stats_results[stat_name] = result[0]['count'] if result else 0
                
                # Display stats
                stat_cols = st.columns(len(stats_results))
                for i, (stat_name, count) in enumerate(stats_results.items()):
                    with stat_cols[i]:
                        st.metric(stat_name, count)
                        
            except Exception as e:
                st.error(f"❌ Failed to fetch stats: {str(e)}")

                else:
                    st.error(f"Could not find information to visualize for '{vis_medicine}'.")