import os
import streamlit as st
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

class Neo4jConnection:
    """
    A class to manage the connection to a Neo4j database.
    It uses the credentials sourced from Streamlit secrets or a local .env file.
    """
    def __init__(self):
        # Prioritize Streamlit secrets, fall back to .env for local dev
        if hasattr(st, 'secrets') and "NEO4J_URI" in st.secrets:
            uri = st.secrets["NEO4J_URI"]
            user = st.secrets["NEO4J_USER"]
            password = st.secrets["NEO4J_PASSWORD"]
            print("Connecting to Neo4j using Streamlit secrets.")
        else:
            uri = os.getenv("NEO4J_URI")
            user = os.getenv("NEO4J_USER")
            password = os.getenv("NEO4J_PASSWORD")
            print("Connecting to Neo4j using local .env file.")

        self._driver = GraphDatabase.driver(uri, auth=(user, password))
        try:
            # Verify connection
            self._driver.verify_connectivity()
            print("Connected to Neo4j")
        except Exception as e:
            print(f"Neo4j connection failed: {e}")

    def close(self):
        if self._driver is not None:
            self._driver.close()

    def query(self, query, parameters=None, db=None):
        """Runs a Cypher query and returns the results."""
        assert self._driver is not None, "Driver not initialized!"
        session = None
        response = None
        try:
            session = self._driver.session(database=db) if db is not None else self._driver.session()
            response = list(session.run(query, parameters))
        except Exception as e:
            print("Query failed:", e)
        finally:
            if session is not None:
                session.close()
        return response

# Instantiate a global graph object for LangChain
# This makes it easy for LangChain agents/chains to access graph schema and data
graph = Neo4jGraph(
    url=st.secrets["NEO4J_URI"], 
    username=st.secrets["NEO4J_USER"], 
    password=st.secrets["NEO4J_PASSWORD"]
)

# Refresh schema information for the LangChain graph object
# This helps the LLM generate more accurate Cypher queries
try:
    graph.refresh_schema()
except Exception as e:
    print(f"Warning: Could not refresh graph schema. The LLM might generate less accurate queries. Error: {e}")