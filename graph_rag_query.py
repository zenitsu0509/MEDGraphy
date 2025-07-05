import os
from graph_db import Neo4jConnection
from sentence_transformers import SentenceTransformer
import streamlit as st

# --- CONFIGURATION ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", 'all-MiniLM-L6-v2')
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", 'medicine_embeddings')

@st.cache_resource
def get_embedding_model():
    """Initializes and returns the SentenceTransformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)

class GraphQueryEngine:
    """Handles queries to the Neo4j database."""

    def __init__(self):
        self.db = Neo4jConnection()
        self.model = get_embedding_model()
        st.sidebar.success("Embedding model loaded")

    def get_embedding(self, text: str) -> list[float]:
        """Generates an embedding for a given text."""
        return self.model.encode(text).tolist()

    def direct_lookup(self, medicine_name: str) -> list[dict]:
        """Finds uses and side effects for a specific medicine."""
        query = """
            MATCH (m:Medicine {name: $med_name})
            OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
            RETURN m.name AS medicine, 
                   collect(DISTINCT c.name) AS uses, 
                   collect(DISTINCT s.name) AS side_effects
        """
        result = self.db.query(query, parameters={"med_name": medicine_name}, db="neo4j")
        return result

    def reverse_lookup(self, condition: str) -> list[str]:
        """Finds medicines that treat a specific condition."""
        query = """
            MATCH (c:Condition {name: $cond_name})<-[:TREATS]-(m:Medicine)
            RETURN m.name AS medicine
            LIMIT 10
        """
        result = self.db.query(query, parameters={"cond_name": condition}, db="neo4j")
        return [record['medicine'] for record in result] if result else []

    def check_interactions(self, medicine_name: str) -> list[dict]:
        """Finds other medicines that share the same active ingredient."""
        query = """
            MATCH (m1:Medicine {name: $med_name})-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
            MATCH (m2:Medicine)-[:CONTAINS_INGREDIENT]->(i)
            WHERE m1 <> m2
            RETURN m2.name AS other_medicine, i.name AS shared_ingredient
            LIMIT 10
        """
        result = self.db.query(query, parameters={"med_name": medicine_name}, db="neo4j")
        return result

    def vector_similarity_search(self, query: str) -> list[dict]:
        """Uses embeddings to find semantically similar medicines."""
        query_embedding = self.get_embedding(query)
        cypher_query = f"""
            CALL db.index.vector.queryNodes($index_name, 5, $embedding)
            YIELD node AS medicine, score
            RETURN medicine.name, score
        """
        result = self.db.query(
            cypher_query, 
            parameters={'index_name': VECTOR_INDEX_NAME, 'embedding': query_embedding},
            db="neo4j"
        )
        return result

    def retrieve_context_for_rag(self, user_query: str) -> dict | None:
        """Retrieves the context from the graph for the RAG pipeline."""
        # 1. RETRIEVE: Find the most relevant medicine using vector search
        query_embedding = self.get_embedding(user_query)
        retrieval_query = f"""
            CALL db.index.vector.queryNodes($index_name, 1, $embedding)
            YIELD node AS medicine
            RETURN medicine.name AS med_name
        """
        retrieval_result = self.db.query(
            retrieval_query, 
            parameters={'index_name': VECTOR_INDEX_NAME, 'embedding': query_embedding},
            db="neo4j"
        )

        if not retrieval_result:
            return None

        top_medicine_name = retrieval_result[0]['med_name']

        # 2. AUGMENT: Fetch its full context from the graph
        context_query = """
            MATCH (m:Medicine {name: $med_name})
            OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
            OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
            RETURN m.name AS medicine,
                   collect(DISTINCT c.name) AS uses,
                   collect(DISTINCT s.name) AS side_effects,
                   collect(DISTINCT i.name) AS ingredients
        """
        context_result = self.db.query(context_query, parameters={"med_name": top_medicine_name}, db="neo4j")
        
        return {
            "medicine_found": top_medicine_name,
            "context": context_result[0] if context_result else {}
        }

    def get_graph_for_visualization(self, medicine_name: str) -> list:
        """Fetches a subgraph for a given medicine for visualization."""
        query = """
            MATCH path = (m:Medicine {name: $med_name})-[r]-(n)
            RETURN m, r, n
        """
        result = self.db.query(query, parameters={"med_name": medicine_name}, db="neo4j")
        return result if result else []
