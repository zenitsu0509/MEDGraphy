import os
from graph_db import Neo4jConnection
from sentence_transformers import SentenceTransformer
import streamlit as st

# --- CONFIGURATION ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_INDEX_NAME = os.getenv("VECTOR_INDEX_NAME", "medicine_embeddings")


@st.cache_resource
def get_embedding_model():
    """Initializes and returns the SentenceTransformer model."""
    return SentenceTransformer(EMBEDDING_MODEL)


class GraphQueryEngine:
    """Handles queries to the Neo4j database."""

    def __init__(self):
        self.db = Neo4jConnection()
        self.model = get_embedding_model()
        print("Embedding model loaded")

    # ----------------------------
    # Basic helpers
    # ----------------------------
    def get_embedding(self, text: str) -> list[float]:
        """Generates an embedding for a given text."""
        return self.model.encode(text).tolist()

    # ----------------------------
    # Core queries
    # ----------------------------
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
        """Finds medicines that treat a specific condition (case-insensitive).
        Falls back to searching uses_text if Condition nodes are missing.
        """
        query = """
            CALL {
              WITH $cond_name AS q
              MATCH (m:Medicine)-[:TREATS]->(c:Condition)
              WHERE toLower(c.name) = toLower(q) OR toLower(c.name) CONTAINS toLower(q)
              RETURN DISTINCT m.name AS medicine
              UNION
              WITH $cond_name AS q
              MATCH (m:Medicine)
              WHERE m.uses_text IS NOT NULL AND toLower(m.uses_text) CONTAINS toLower(q)
              RETURN DISTINCT m.name AS medicine
            }
            RETURN medicine
            LIMIT 25
        """
        result = self.db.query(query, parameters={"cond_name": condition}, db="neo4j")
        return [record["medicine"] for record in result] if result else []

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
        cypher_query = """
            CALL db.index.vector.queryNodes($index_name, 5, $embedding)
            YIELD node AS medicine, score
            RETURN medicine.name, score
        """
        result = self.db.query(
            cypher_query,
            parameters={"index_name": VECTOR_INDEX_NAME, "embedding": query_embedding},
            db="neo4j",
        )
        return result

    
    def _best_medicine_for_condition(self, condition: str) -> str | None:
        """Pick a representative medicine for a condition using simple heuristics.
        Prefers higher excellent review %, then average; falls back to any match.
        """
        if not condition:
            return None
        cypher = """
        CALL {
          WITH $cond AS q
          MATCH (m:Medicine)-[:TREATS]->(c:Condition)
          WHERE toLower(c.name) = toLower(q) OR toLower(c.name) CONTAINS toLower(q)
          RETURN m
          UNION
          WITH $cond AS q
          MATCH (m:Medicine)
          WHERE m.uses_text IS NOT NULL AND toLower(m.uses_text) CONTAINS toLower(q)
          RETURN m
        }
        RETURN m.name AS name,
               coalesce(m.excellent_review_pct,0) AS excellent,
               coalesce(m.average_review_pct,0) AS average
        ORDER BY excellent DESC, average DESC
        LIMIT 1
        """
        res = self.db.query(cypher, {"cond": condition}, db="neo4j")
        if res and len(res) > 0:
            try:
                return res[0]["name"]
            except Exception:
                rec = res[0]
                if hasattr(rec, "data"):
                    return rec.data().get("name")
                if hasattr(rec, "keys"):
                    return rec["name"]
        return None

    def _extract_condition_from_query(self, user_query: str) -> str | None:
        """Very light heuristic to extract a condition phrase like 'fever' from queries.
        Examples: 'medicine for fever', 'drug for high fever', 'fever medicine'.
        """
        if not user_query:
            return None
        q = user_query.strip().lower()
        # Rule 1: look for ' for <cond>' pattern
        if " for " in q:
            tail = q.split(" for ", 1)[1]
            for sep in ["?", ".", ",", ";", ":"]:
                tail = tail.split(sep)[0]
            cond = tail.strip()
            for lead in ["a ", "an ", "the "]:
                if cond.startswith(lead):
                    cond = cond[len(lead):]
            cond = cond.strip()
            if cond:
                return cond
        # Rule 2: '<cond> medicine' pattern
        if q.endswith(" medicine"):
            cond = q.rsplit(" ", 1)[0].strip()
            return cond or None
        # Rule 3: common single-word conditions
        common = {"fever", "cold", "cough", "pain", "migraine", "diarrhea", "diarrhoea"}
        tokens = [t for t in q.replace("?", " ").split() if t.isalpha()]
        for t in tokens:
            if t in common:
                return t
        return None

    def retrieve_context_for_rag(self, user_query: str) -> dict | None:
        """Retrieves the context from the graph for the RAG pipeline.
        Strategy: Try condition-first (e.g., 'medicine for fever') -> choose a best medicine.
        Fall back to vector retrieval if condition-based selection fails.
        """
        top_medicine_name: str | None = None

        # A) Condition-first heuristic
        cond = self._extract_condition_from_query(user_query)
        if cond:
            best = self._best_medicine_for_condition(cond)
            if best:
                top_medicine_name = best

        # B) Vector fallback if no clear condition-based anchor
        if not top_medicine_name:
            query_embedding = self.get_embedding(user_query)
            retrieval_query = """
                CALL db.index.vector.queryNodes($index_name, 1, $embedding)
                YIELD node AS medicine
                RETURN medicine.name AS med_name
            """
            retrieval_result = self.db.query(
                retrieval_query,
                parameters={"index_name": VECTOR_INDEX_NAME, "embedding": query_embedding},
                db="neo4j",
            )
            if not retrieval_result:
                return None
            top_medicine_name = retrieval_result[0]["med_name"]

        # 2. AUGMENT: Fetch its full context from the graph
        context_query = """
            MATCH (m:Medicine {name: $med_name})
            OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
            OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
            OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
            OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
            RETURN m.name AS medicine,
                   m.composition AS composition,
                   m.uses_text AS uses_text,
                   m.side_effects_text AS side_effects_text,
                   m.image_url AS image_url,
                   m.excellent_review_pct AS excellent_review_pct,
                   m.average_review_pct AS average_review_pct,
                   m.poor_review_pct AS poor_review_pct,
                   mf.name AS manufacturer,
                   collect(DISTINCT c.name) AS uses,
                   collect(DISTINCT s.name) AS side_effects,
                   collect(DISTINCT i.name) AS ingredients
        """
        context_result = self.db.query(
            context_query, parameters={"med_name": top_medicine_name}, db="neo4j"
        )

        return {
            "medicine_found": top_medicine_name,
            "context": context_result[0] if context_result else {},
        }

    # ----------------------------
    # Utilities for UI
    # ----------------------------
    def get_graph_for_visualization(self, medicine_name: str) -> list:
        """Fetches a subgraph for a given medicine for visualization."""
        query = """
            MATCH path = (m:Medicine {name: $med_name})-[r]-(n)
            RETURN m, r, n
        """
        result = self.db.query(query, parameters={"med_name": medicine_name}, db="neo4j")
        return result if result else []

    def get_medicine_with_image(self, name: str):
        query = """
        MATCH (m:Medicine {name: $name})
        OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(mf:Manufacturer)
        RETURN m.name as name, m.image_url AS image_url, m.composition AS composition,
               m.uses_text AS uses_text, m.side_effects_text AS side_effects_text,
               collect(DISTINCT c.name) AS conditions,
               collect(DISTINCT s.name) AS side_effects,
               collect(DISTINCT i.name) AS ingredients,
               mf.name AS manufacturer,
               m.excellent_review_pct AS excellent_review_pct,
               m.average_review_pct AS average_review_pct,
               m.poor_review_pct AS poor_review_pct
        """
        res = self.db.query(query, {"name": name}, db="neo4j")
        return res[0] if res else None

    def symptom_to_medicines(self, symptoms: list[str], limit: int = 10):
        # Leverage side effect nodes to map symptoms -> medicines
        query = """
        UNWIND $symptoms AS sym
        MATCH (s:SideEffect)
        WHERE toLower(s.name) CONTAINS toLower(sym)
        MATCH (m:Medicine)-[:HAS_SIDE_EFFECT]->(s)
        RETURN s.name AS matched_symptom, collect(DISTINCT m.name) AS medicines
        LIMIT $limit
        """
        return self.db.query(query, {"symptoms": symptoms, "limit": limit}, db="neo4j")

    def justify_prescription(self, medicines: list[str]):
        # Return structured data for each medicine for LLM justification step
        query = """
        UNWIND $medicines AS med
        MATCH (m:Medicine {name: med})
        OPTIONAL MATCH (m)-[:TREATS]->(c:Condition)
        OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        OPTIONAL MATCH (m)-[:CONTAINS_INGREDIENT]->(i:ActiveIngredient)
        RETURN m.name AS medicine,
               m.composition AS composition,
               collect(DISTINCT c.name) AS conditions,
               collect(DISTINCT i.name) AS ingredients,
               collect(DISTINCT s.name) AS side_effects
        """
        return self.db.query(query, {"medicines": medicines}, db="neo4j")

    def interaction_conflicts(self, medicine: str):
        # Interacts via previously created INTERACTS_WITH rel
        query = """
        MATCH (m:Medicine {name: $medicine})-[:INTERACTS_WITH]-(o:Medicine)
        RETURN o.name AS interacting_medicine
        LIMIT 25
        """
        return self.db.query(query, {"medicine": medicine}, db="neo4j")
