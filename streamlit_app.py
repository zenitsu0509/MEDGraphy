import streamlit as st
import os
from urllib.parse import urlparse
from dotenv import load_dotenv
from graph_rag_query import GraphQueryEngine
from llm_chains import get_rag_response, get_groq_client
from streamlit_agraph import agraph, Node, Edge, Config

# --- CONFIGURATION ---
load_dotenv()

# --- PAGE CONFIGURATION ---
st.set_page_config(page_title="MEDGraph", layout="wide", page_icon="⚕️")

# Hide the sidebar completely
st.markdown("""
<style>
    .css-1d391kg {display: none}
    .st-emotion-cache-16idsys {display: none}
    section[data-testid="stSidebar"] {display: none !important}
    .css-1rs6os {display: none}
    .css-17eq0hr {display: none}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div style='display:flex;align-items:center;gap:0.75rem;'>
    <h1 style='margin:0'>⚕️ MEDGraphy</h1>
    <span style='font-size:0.9rem;opacity:0.75'>Graph RAG Drug Intelligence</span>
</div>
<hr style='margin-top:0.25rem;margin-bottom:1rem;'>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data(show_spinner=False)
def _cache_med_card(name: str):
    """Return a JSON-serialisable dict for a medicine card.
    Converts Neo4j Record objects into plain dicts so Streamlit caching & hashing work.
    """
    record = engine.get_medicine_with_image(name)
    if record is None:
        return None
    # neo4j.Record provides .data() in recent drivers
    try:
        if hasattr(record, "data"):
            record = record.data()
        else:
            # Fallback: attempt to build dict from items()
            record = {k: record[k] for k in record.keys()} if hasattr(record, "keys") else dict(record)
    except Exception:
        # As last resort wrap in dict with name only
        return {"name": name}
    return record

def sanitize_image_url(image_url: str | None):
    if not image_url or not isinstance(image_url, str):
        return None
    url = image_url.strip().strip('"')
    if not url:
        return None
    # Heuristic: if it's a relative path or missing scheme, skip to avoid broken request
    parsed = urlparse(url)
    if not parsed.scheme:
        # sometimes stored without scheme (e.g., //domain.com/img.jpg)
        if url.startswith('//'):
            return f"https:{url}"
        # ignore local file paths for web
        return None
    return url

def display_medicine_image(image_url, medicine_name, height=160):
    """Display medicine image with robust validation and graceful fallback."""
    safe_url = sanitize_image_url(image_url)
    img_container = st.container()
    with img_container:
        if safe_url:
            try:
                st.image(safe_url, caption=medicine_name, use_container_width=True)
            except Exception as e:
                st.markdown(f"<div style='border:1px solid #ccc;padding:0.75rem;border-radius:8px;text-align:center;'>🖼️ <b>{medicine_name}</b><br><span style='font-size:0.8rem;color:#b00;'>Image failed: {e}</span></div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='border:1px dashed #ccc;padding:0.9rem;border-radius:8px;text-align:center;background:#fafafa;'>🧪<br><b>{medicine_name}</b><br><span style='font-size:0.75rem;opacity:0.7'>No image available</span></div>", unsafe_allow_html=True)

def render_review_bar(med_card):
    if med_card is None:
        return
    ex = med_card.get('excellent_review_pct') or 0
    avg = med_card.get('average_review_pct') or 0
    poor = med_card.get('poor_review_pct') or 0
    total = ex + avg + poor
    if total <= 0:
        return st.markdown("<span style='font-size:0.75rem;opacity:0.65'>No review data</span>", unsafe_allow_html=True)
    # Horizontal segmented bar
    st.markdown(
        f"""
        <div style='display:flex;height:14px;width:100%;border-radius:7px;overflow:hidden;border:1px solid #ddd;margin:4px 0 6px 0;'>
          <div style='flex:{ex};background:#2e8540' title='Excellent {ex}%'></div>
          <div style='flex:{avg};background:#ffcc33' title='Average {avg}%'></div>
          <div style='flex:{poor};background:#d9534f' title='Poor {poor}%'></div>
        </div>
        <div style='font-size:0.7rem;display:flex;justify-content:space-between;opacity:0.75'>
          <span>👍 {ex}%</span><span>😐 {avg}%</span><span>👎 {poor}%</span>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_medicine_card(med_card, show_sections: list[str] | None = None, expandable: bool = False, subtitle: str | None = None):
    if not med_card:
        st.warning("Medicine not found.")
        return
    show_sections = show_sections or ["composition", "manufacturer", "conditions", "side_effects", "ingredients"]
    card_id = med_card.get('name','medicine')
    content_block = st.expander(f"💊 {med_card['name']}{' — ' + subtitle if subtitle else ''}") if expandable else st.container()
    with content_block:
        cols = st.columns([1,2])
        with cols[0]:
            display_medicine_image(med_card.get('image_url'), med_card.get('name'))
        with cols[1]:
            st.markdown(f"### {med_card.get('name')}")
            if 'composition' in show_sections and med_card.get('composition'):
                st.markdown(f"**Composition:** {med_card['composition']}")
            if 'manufacturer' in show_sections and med_card.get('manufacturer'):
                st.markdown(f"**Manufacturer:** {med_card['manufacturer']}")
            if 'conditions' in show_sections and med_card.get('conditions'):
                filtered = [c for c in med_card.get('conditions', []) if c]
                if filtered:
                    st.markdown(f"**Treats:** {', '.join(filtered[:25])}")
            if 'side_effects' in show_sections and med_card.get('side_effects'):
                ses = [s for s in med_card.get('side_effects', []) if s]
                if ses:
                    st.markdown(f"**Side Effects (sample):** {', '.join(ses[:30])}")
            if 'ingredients' in show_sections and med_card.get('ingredients'):
                ing = [i for i in med_card.get('ingredients', []) if i]
                if ing:
                    st.markdown(f"**Ingredients:** {', '.join(ing[:15])}")
            render_review_bar(med_card)
            if med_card.get('uses_text'):
                with st.expander("Raw Uses Text"):
                    st.write(med_card['uses_text'])
            if med_card.get('side_effects_text'):
                with st.expander("Raw Side Effects Text"):
                    st.write(med_card['side_effects_text'])

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

# Subtle helper tooltip line
st.caption("Powered by Neo4j Graph + Vector RAG + Groq LLM")

# --- MAIN APP ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "❓ Full RAG Query", 
    "💊 Direct Lookup", 
    "🩺 Condition Lookup", 
    "⚠️ Interaction Check", 
    "🔎 Vector Search",
    "📊 Graph Visualization",
    "🔧 Debug Images"
])

with tab1:
    st.header("❓ General FAQ")
    st.write("End-to-end: Retrieve ➜ Augment ➜ Generate.")
    user_query = st.text_area("Ask a question about medicines, uses, or side effects:", "What are the side effects of a medicine for headaches?", height=110, key="rag_query")
    col_q1, col_q2 = st.columns([1,1])
    with col_q1:
        run_rag = st.button("🔍 Run Full RAG", type="primary")
    with col_q2:
        quick_clear = st.button("🧹 Clear Context")
    if quick_clear:
        st.experimental_rerun()
    if run_rag:
        with st.spinner("Retrieving most relevant medicine & building context..."):
            rag_context = engine.retrieve_context_for_rag(user_query)
            if not rag_context or not rag_context.get("context"):
                st.error("No relevant medicine found for that query.")
            else:
                med_name = rag_context['medicine_found']
                med_card = _cache_med_card(med_name)
                st.success(f"Top relevant medicine: {med_name}")
                render_medicine_card(med_card, expandable=False, subtitle="RAG Anchor")
                response = get_rag_response(user_query, rag_context['context'], groq_client)
                st.subheader("LLM Response")
                st.markdown(response)
                with st.expander("🔬 Raw Graph Context JSON"):
                    st.json(rag_context['context'])

with tab2:
    st.header("💊 Direct Medicine Lookup")
    st.write("Fetch full structured info for a single medicine.")
    lookup_cols = st.columns([3,1])
    with lookup_cols[0]:
        med_name_direct = st.text_input("Medicine Name", "Kelvin 500mg Tablet", key="direct")
    with lookup_cols[1]:
        go_lookup = st.button("Lookup")
    if go_lookup:
        with st.spinner("Querying graph..."):
            med_card = _cache_med_card(med_name_direct)
            render_medicine_card(med_card, expandable=False)

with tab3:
    st.header("🩺 Reverse Lookup by Condition")
    st.write("Find medicines that treat a given condition.")
    cond_cols = st.columns([3,1])
    with cond_cols[0]:
        condition_name = st.text_input("Condition", "Hypoglycemia", key="reverse")
    with cond_cols[1]:
        go_reverse = st.button("Find")
    if go_reverse:
        with st.spinner("Searching graph..."):
            result = engine.reverse_lookup(condition_name)
            if result:
                st.subheader(f"Medicines for {condition_name}")
                for med_name in result:
                    med_card = _cache_med_card(med_name)
                    render_medicine_card(med_card, expandable=True)
            else:
                st.info("No matches.")

with tab4:
    st.header("⚠️ Potential Interaction Check")
    st.write("Find medicines sharing active ingredients (potential interaction list).")
    inter_cols = st.columns([3,1])
    with inter_cols[0]:
        med_name_interact = st.text_input("Medicine", "Kidnymax Tablet", key="interaction")
    with inter_cols[1]:
        go_inter = st.button("Check")
    if go_inter:
        with st.spinner("Resolving ingredient overlaps..."):
            result = engine.check_interactions(med_name_interact)
            if result:
                for interaction in result:
                    other_med = interaction.get('other_medicine')
                    shared = interaction.get('shared_ingredient')
                    med_card = _cache_med_card(other_med) if other_med else None
                    render_medicine_card(med_card, expandable=True, subtitle=f"Shared: {shared}")
            else:
                st.info("None found.")

with tab5:
    st.header("🔎 Vector Similarity Search")
    st.write("Semantic similarity search over embedded medicines.")
    vect_cols = st.columns([3,1])
    with vect_cols[0]:
        query_text = st.text_input("Search Phrase", "medicine for joint pain", key="vector")
    with vect_cols[1]:
        go_vec = st.button("Search")
    if go_vec:
        with st.spinner("Running vector index query..."):
            result = engine.vector_similarity_search(query_text)
            if result:
                for med_result in result:
                    med_name = med_result.get('medicine.name')
                    med_card = _cache_med_card(med_name) if med_name else None
                    render_medicine_card(med_card, expandable=True, subtitle=f"Score: {med_result.get('score', 0):.3f}")
            else:
                st.info("No similar medicines found.")

with tab6:
    st.header("📊 Graph Visualization")
    st.write("Interactive subgraph around a medicine node.")
    vis_cols = st.columns([3,1])
    with vis_cols[0]:
        vis_medicine = st.text_input("Medicine to Visualize", "Kronostar 300 Tablet CR", key="vis_med")
    with vis_cols[1]:
        go_vis = st.button("Render")

    if go_vis and vis_medicine:
        with st.spinner("Building graph model..."):
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
                        nodes.append(Node(id=source_id, label=source_name, shape="dot", size=28, font={"size": 22}, color="#FF9900", title=source_label))
                        node_ids.add(source_id)
                    if target_id not in node_ids:
                        color_map = {"Condition": "#ffb3c6", "SideEffect": "#b3d9ff", "ActiveIngredient": "#baf5ba", "Manufacturer": "#ffe1a8"}
                        nodes.append(Node(id=target_id, label=target_name, shape="box", color=color_map.get(target_label, "#E0E0E0"), title=target_label))
                        node_ids.add(target_id)
                    rel_type = rel.type.replace("_", " ").title()
                    edges.append(Edge(source=source_id, target=target_id, label=rel_type))
                config = Config(width=1100, height=700, directed=True, physics=True, hierarchical=False, nodeHighlightBehavior=True, highlightColor="#F7A7A6")
                agraph(nodes=nodes, edges=edges, config=config)
            else:
                st.warning("No graph data available for that medicine.")

with tab7:
    st.header("🔧 Image Debug Tool")
    st.write("Investigate how image URLs are being resolved & rendered.")
    debug_med = st.text_input("Medicine", "Avastin 400mg Injection", key="debug_med")
    if st.button("Inspect"):
        with st.spinner("Pulling record..."):
            med_card = _cache_med_card(debug_med)
            if med_card:
                st.subheader("Record JSON")
                st.json(med_card)
                url = med_card.get('image_url')
                st.markdown(f"**Raw URL:** `{url}`")
                safe = sanitize_image_url(url)
                st.markdown(f"**Sanitized URL:** `{safe}`")
                display_medicine_image(url, debug_med)
            else:
                st.error("Not found.")
    st.subheader("Quick Sample Tests")
    sample_meds = ["Avastin 400mg Injection", "Augmentin 625 Duo Tablet", "Azithral 500 Tablet"]
    cols_dbg = st.columns(len(sample_meds))
    for i, med in enumerate(sample_meds):
        with cols_dbg[i]:
            med_card = _cache_med_card(med)
            display_medicine_image(med_card.get('image_url') if med_card else None, med)
            if med_card:
                st.caption("OK" if sanitize_image_url(med_card.get('image_url')) else "No URL")
