import streamlit as st
from dotenv import load_dotenv
import llm_chains
from graph_db import Neo4jConnection
from streamlit_agraph import agraph, Node, Edge, Config

# Load environment variables
load_dotenv()

# --- Page Configuration ---
st.set_page_config(page_title="GraphRAG Medical Assistant", layout="wide")
st.title("⚕️ GraphRAG-Powered AI Medical Assistant")
st.write("Interacting with a Neo4j medicine knowledge graph.")

# --- Initialize Connections (cached for performance) ---
@st.cache_resource
def get_neo4j_connection():
    return Neo4jConnection()

db = get_neo4j_connection()

# --- UI Tabs for each feature ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "1. AI Drug Assistant",
    "2. Prescription Justifier",
    "3. Symptom to Drug",
    "4. Visual Explainer",
    "5. General QA"
])

# =====================================================================================
# TAB 1: AI Drug Assistant (Explain, Compare, Recommend)
# =====================================================================================
with tab1:
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
# TAB 2: Prescription Justifier
# =====================================================================================
with tab2:
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
# TAB 3: Symptom-to-Drug Recommender
# =====================================================================================
with tab3:
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
# TAB 4: RAG-Based Visual Explainer
# =====================================================================================
with tab4:
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

                else:
                    st.error(f"Could not find information to visualize for '{vis_medicine}'.")

# =====================================================================================
# TAB 5: Zero-Shot Question Answering
# =====================================================================================
with tab5:
    st.header("Zero-Shot Question Answering over Graph")
    st.write("Ask any natural language question about the medical data.")
    question = st.text_area("Your Question:", value="Which medicines treat Lung cancer and are made by Genentech?")

    if st.button("Get Answer", type="primary"):
        if question:
            with st.spinner("Thinking..."):
                try:
                    result = llm_chains.cypher_qa_chain.invoke(question)
                    st.success("Answer:")
                    st.markdown(result['result'])
                    
                    # Check if intermediate steps are available before trying to display them
                    if 'intermediate_steps' in result and result['intermediate_steps']:
                        with st.expander("Show Details"):
                            st.write("**Generated Cypher Query:**")
                            st.code(result['intermediate_steps'][0]['query'], language='cypher')
                            st.write("**GraphDB Result:**")
                            st.json(result['intermediate_steps'][1]['context'])

                except Exception as e:
                    st.error(f"An error occurred. The LLM might have generated an invalid query. Please try rephrasing. Error: {e}")