import os
import streamlit as st
from neo4j import GraphDatabase
from langchain_community.graphs import Neo4jGraph
from dotenv import load_dotenv

# Load environment variables from .env file for local development
load_dotenv()

# Determine credentials based on the environment (local vs. Streamlit Cloud)
# In Streamlit Cloud, credentials should be set in the secrets management.
# For local development, they are loaded from the .env file.
if 'NEO4J_URI' in st.secrets:
    uri = st.secrets["NEO4J_URI"]
    user = st.secrets["NEO4J_USERNAME"]
    password = st.secrets["NEO4J_PASSWORD"]
    print("Connecting to Neo4j using Streamlit secrets.")
else:
    uri = os.getenv("NEO4J_URI")
    user = os.getenv("NEO4J_USERNAME")
    password = os.getenv("NEO4J_PASSWORD")
    print("Connecting to Neo4j using local .env file.")


class Neo4jConnection:
    """
    A class to manage the connection to a Neo4j database.
    It uses the credentials sourced from Streamlit secrets or a local .env file.
    """
    def __init__(self):
        self.uri = uri
        self.user = user
        self.password = password
        self.driver = None
        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            # Verify connection
            self.driver.verify_connectivity()
            print("Successfully connected to Neo4j.")
        except Exception as e:
            print(f"Failed to create Neo4j driver: {e}")

    def close(self):
        if self.driver is not None:
            self.driver.close()

    def query(self, query, parameters=None):
        """Runs a Cypher query and returns the results."""
        if self.driver is None:
            print("Driver not initialized.")
            return []
        with self.driver.session() as session:
            result = session.run(query, parameters)
            return [record for record in result]

# Instantiate a global graph object for LangChain
# This makes it easy for LangChain agents/chains to access graph schema and data
graph = Neo4jGraph(
    url=uri, 
    username=user, 
    password=password
)

# Refresh schema information for the LangChain graph object
# This helps the LLM generate more accurate Cypher queries
try:
    graph.refresh_schema()
except Exception as e:
    print(f"Warning: Could not refresh graph schema. The LLM might generate less accurate queries. Error: {e}")