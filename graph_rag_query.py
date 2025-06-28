from typing import List, Dict, Any, Optional
from graph_db import Neo4jConnection
import re

class GraphRAGQueryEngine:
    """Enhanced Graph RAG query engine with semantic search capabilities"""
    
    def __init__(self):
        self.db = Neo4jConnection()
        self.disease_synonyms = {
            'lung cancer': ['lung cancer', 'pulmonary cancer', 'lung carcinoma', 'bronchogenic carcinoma', 'non-small cell lung cancer', 'small cell lung cancer'],
            'breast cancer': ['breast cancer', 'mammary cancer', 'breast carcinoma'],
            'kidney cancer': ['kidney cancer', 'renal cancer', 'renal cell carcinoma'],
            'pancreatic cancer': ['pancreatic cancer', 'pancreas cancer'],
            'brain tumor': ['brain tumor', 'brain cancer', 'cerebral tumor'],
            'ovarian cancer': ['ovarian cancer', 'ovary cancer'],
            'cervical cancer': ['cervical cancer', 'cervix cancer']
        }
    
    def normalize_query(self, query: str) -> str:
        """Normalize user query to match graph data better"""
        query = query.lower().strip()
        
        # Handle common variations
        replacements = {
            'lung cancers': 'lung cancer',
            'breast cancers': 'breast cancer',
            'what medicines treat': '',
            'which medicines treat': '',
            'medicines for': '',
            'drugs for': '',
            'treatment for': '',
            'medicine that treats': '',
        }
        
        for old, new in replacements.items():
            query = query.replace(old, new)
        
        return query.strip()
    
    def extract_disease_from_query(self, query: str) -> List[str]:
        """Extract disease names from natural language query"""
        normalized_query = self.normalize_query(query)
        found_diseases = []
        
        # Check for exact matches with known diseases
        for main_disease, synonyms in self.disease_synonyms.items():
            for synonym in synonyms:
                if synonym in normalized_query:
                    found_diseases.append(main_disease)
                    break
        
        # If no exact matches, try to extract from the query
        if not found_diseases:
            # Look for medical terms that might be diseases
            medical_terms = re.findall(r'\b(\w+(?:\s+\w+){0,2}(?:\s+cancer|\s+tumor|\s+disease|\s+syndrome|\s+infection))\b', normalized_query, re.IGNORECASE)
            found_diseases.extend([term.strip() for term in medical_terms])
        
        return list(set(found_diseases))
    
    def get_comprehensive_medicine_info(self, medicine_name: str) -> Dict[str, Any]:
        """Get comprehensive information about a medicine"""
        query = """
        MATCH (m:Medicine {name: $medicine_name})
        OPTIONAL MATCH (m)-[:HAS_COMPOSITION]->(c:Composition)
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
        OPTIONAL MATCH (m)-[:USED_FOR]->(d:Disease)
        OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        RETURN m.name as medicine,
               c.name as composition,
               man.name as manufacturer,
               collect(DISTINCT d.name) as diseases_treated,
               collect(DISTINCT s.name) as side_effects
        """
        
        result = self.db.query(query, {'medicine_name': medicine_name})
        return dict(result[0]) if result else {}
    
    def semantic_disease_search(self, disease_query: str) -> List[Dict[str, Any]]:
        """Perform semantic search for diseases with multiple matching strategies"""
        # Strategy 1: Exact match
        exact_query = """
        MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
        WHERE toLower(d.name) = toLower($disease_query)
        OPTIONAL MATCH (m)-[:HAS_COMPOSITION]->(c:Composition)
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
        RETURN DISTINCT m.name as medicine,
               d.name as disease_treated,
               c.name as composition,
               man.name as manufacturer
        ORDER BY m.name
        """
        
        exact_results = self.db.query(exact_query, {'disease_query': disease_query})
        
        if exact_results:
            return [dict(r) for r in exact_results]
        
        # Strategy 2: Contains match
        contains_query = """
        MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($disease_query)
        OPTIONAL MATCH (m)-[:HAS_COMPOSITION]->(c:Composition)
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
        RETURN DISTINCT m.name as medicine,
               d.name as disease_treated,
               c.name as composition,
               man.name as manufacturer
        ORDER BY length(d.name) ASC, m.name
        """
        
        contains_results = self.db.query(contains_query, {'disease_query': disease_query})
        
        if contains_results:
            return [dict(r) for r in contains_results]
        
        # Strategy 3: Word-based fuzzy search
        words = disease_query.lower().split()
        if len(words) > 1:
            word_query = """
            MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
            WHERE ANY(word IN $words WHERE toLower(d.name) CONTAINS word)
            OPTIONAL MATCH (m)-[:HAS_COMPOSITION]->(c:Composition)
            OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
            WITH m, d, c, man,
                 size([word IN $words WHERE toLower(d.name) CONTAINS word]) as word_matches
            WHERE word_matches >= 1
            RETURN DISTINCT m.name as medicine,
                   d.name as disease_treated,
                   c.name as composition,
                   man.name as manufacturer,
                   word_matches
            ORDER BY word_matches DESC, length(d.name) ASC, m.name
            """
            
            word_results = self.db.query(word_query, {'words': words})
            return [dict(r) for r in word_results]
        
        return []
    
    def answer_natural_language_query(self, query: str) -> Dict[str, Any]:
        """Answer natural language queries about medicines and diseases"""
        normalized_query = self.normalize_query(query)
        extracted_diseases = self.extract_disease_from_query(query)
        
        results = {
            'original_query': query,
            'normalized_query': normalized_query,
            'extracted_diseases': extracted_diseases,
            'medicines_found': [],
            'total_medicines': 0,
            'search_strategies_used': []
        }
        
        # Search for each extracted disease
        all_medicines = []
        for disease in extracted_diseases:
            search_results = self.semantic_disease_search(disease)
            all_medicines.extend(search_results)
            if search_results:
                results['search_strategies_used'].append(f"Found medicines for '{disease}'")
        
        # If no diseases were extracted, try searching with the entire query
        if not extracted_diseases:
            search_results = self.semantic_disease_search(normalized_query)
            all_medicines.extend(search_results)
            if search_results:
                results['search_strategies_used'].append(f"Direct search with query")
        
        # Remove duplicates and format results
        unique_medicines = {}
        for medicine in all_medicines:
            med_name = medicine['medicine']
            if med_name not in unique_medicines:
                unique_medicines[med_name] = medicine
        
        results['medicines_found'] = list(unique_medicines.values())
        results['total_medicines'] = len(unique_medicines)
        
        return results
    
    def generate_cypher_query(self, natural_query: str) -> str:
        """Generate appropriate Cypher query based on natural language input"""
        normalized_query = self.normalize_query(natural_query)
        
        # Pattern matching for different query types
        if any(phrase in normalized_query for phrase in ['which medicines', 'what medicines', 'medicines for', 'drugs for']):
            # This is a medicine recommendation query
            diseases = self.extract_disease_from_query(natural_query)
            if diseases:
                disease = diseases[0]  # Use the first extracted disease
                return f"""
                MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
                WHERE toLower(d.name) CONTAINS toLower('{disease}')
                OPTIONAL MATCH (m)-[:HAS_COMPOSITION]->(c:Composition)
                OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
                RETURN m.name as medicine, d.name as disease, c.name as composition, man.name as manufacturer
                ORDER BY m.name
                """
        
        elif any(phrase in normalized_query for phrase in ['side effects', 'adverse effects']):
            # This is a side effects query
            return """
            MATCH (m:Medicine)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
            WHERE toLower(m.name) CONTAINS toLower($medicine_name)
            RETURN m.name as medicine, collect(s.name) as side_effects
            """
        
        elif any(phrase in normalized_query for phrase in ['made by', 'manufactured by', 'manufacturer']):
            # This is a manufacturer query
            return """
            MATCH (m:Medicine)-[:MANUFACTURED_BY]->(man:Manufacturer)
            WHERE toLower(man.name) CONTAINS toLower($manufacturer_name)
            RETURN m.name as medicine, man.name as manufacturer
            ORDER BY m.name
            """
        
        else:
            # Generic search query
            return """
            MATCH (m:Medicine)-[r]-(n)
            WHERE toLower(m.name) CONTAINS toLower($search_term)
               OR toLower(n.name) CONTAINS toLower($search_term)
            RETURN m.name as medicine, type(r) as relationship, n.name as related_entity, labels(n) as entity_type
            LIMIT 20
            """
    
    def debug_lung_cancer_search(self):
        """Debug function to check lung cancer data in the graph"""
        queries = [
            # Check all diseases containing 'lung'
            "MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'lung' RETURN d.name ORDER BY d.name",
            
            # Check all diseases containing 'cancer'
            "MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'cancer' RETURN d.name ORDER BY d.name LIMIT 20",
            
            # Check medicines for lung cancer
            "MATCH (m:Medicine)-[:USED_FOR]->(d:Disease) WHERE toLower(d.name) CONTAINS 'lung' RETURN m.name, d.name",
            
            # Check specific medicine Advacan
            "MATCH (m:Medicine {name: 'Advacan 0.25mg Tablet'})-[:USED_FOR]->(d:Disease) RETURN m.name, d.name",
            
            # Check all relationships for Advacan
            "MATCH (m:Medicine)-[r]-(n) WHERE m.name CONTAINS 'Advacan' RETURN m.name, type(r), n.name, labels(n)"
        ]
        
        for i, query in enumerate(queries, 1):
            print(f"\n=== Debug Query {i} ===")
            print(f"Query: {query}")
            results = self.db.query(query)
            if results:
                for result in results[:10]:  # Show first 10 results
                    print(f"  {result}")
            else:
                print("  No results found")
    
    def test_lung_cancer_queries(self):
        """Test various lung cancer queries"""
        test_queries = [
            "Which medicines treat lung cancer?",
            "What drugs are used for lung cancer?",
            "Medicines for lung cancer",
            "Treatment for lung cancer",
            "lung cancer medicines",
            "Advacan lung cancer"
        ]
        
        print("=== Testing Lung Cancer Queries ===")
        for query in test_queries:
            print(f"\nQuery: '{query}'")
            results = self.answer_natural_language_query(query)
            print(f"Found {results['total_medicines']} medicines")
            if results['medicines_found']:
                for med in results['medicines_found'][:3]:  # Show first 3
                    print(f"  - {med['medicine']} (treats: {med['disease_treated']})")
            else:
                print("  No medicines found")

if __name__ == "__main__":
    # Test the query engine
    engine = GraphRAGQueryEngine()
    
    # Debug lung cancer data
    engine.debug_lung_cancer_search()
    
    # Test lung cancer queries
    engine.test_lung_cancer_queries()
