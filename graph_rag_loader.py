import pandas as pd
import re
from graph_db import Neo4jConnection
from typing import List, Dict, Any

class GraphRAGLoader:
    """Enhanced Graph RAG loader with better parsing and semantic search capabilities"""
    
    def __init__(self):
        self.db = Neo4jConnection()
    
    def parse_uses_field(self, uses_text: str) -> List[str]:
        """Parse the uses field to extract individual diseases/conditions"""
        if not uses_text or pd.isna(uses_text):
            return []
        
        # Split by 'Treatment of' and 'Prevention of' patterns
        # This will handle cases like "Treatment of Breast cancerTreatment of Lung cancer"
        uses_list = []
        
        # First, try to split by known patterns
        patterns = [
            r'Treatment of ([^T]+?)(?=Treatment of|Prevention of|$)',
            r'Prevention of ([^T]+?)(?=Treatment of|Prevention of|$)'
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, uses_text, re.IGNORECASE)
            for match in matches:
                # Clean up the match
                clean_match = match.strip()
                if clean_match:
                    uses_list.append(clean_match)
        
        # If no matches found, try simple splitting
        if not uses_list:
            # Split by common separators
            separators = [' Treatment of ', ' Prevention of ', ',', ';', ' and ', ' or ']
            parts = [uses_text]
            
            for sep in separators:
                new_parts = []
                for part in parts:
                    new_parts.extend(part.split(sep))
                parts = new_parts
            
            uses_list = [part.strip() for part in parts if part.strip()]
        
        # Clean up each use
        cleaned_uses = []
        for use in uses_list:
            # Remove leading/trailing whitespace and common prefixes
            use = use.strip()
            use = re.sub(r'^(Treatment of |Prevention of )', '', use, flags=re.IGNORECASE)
            if use and len(use) > 2:  # Avoid very short strings
                cleaned_uses.append(use)
        
        return cleaned_uses
    
    def parse_side_effects_field(self, effects_text: str) -> List[str]:
        """Parse side effects field"""
        if not effects_text or pd.isna(effects_text):
            return []
        
        # Split by common separators
        effects = re.split(r'[,;]|\s+(?=[A-Z])', effects_text)
        cleaned_effects = []
        
        for effect in effects:
            effect = effect.strip()
            if effect and len(effect) > 2:
                cleaned_effects.append(effect)
        
        return cleaned_effects
    
    def create_enhanced_graph_constraints(self):
        """Create constraints and indexes for better performance"""
        constraints_queries = [
            "CREATE CONSTRAINT medicine_name IF NOT EXISTS FOR (m:Medicine) REQUIRE m.name IS UNIQUE",
            "CREATE CONSTRAINT disease_name IF NOT EXISTS FOR (d:Disease) REQUIRE d.name IS UNIQUE",
            "CREATE CONSTRAINT composition_name IF NOT EXISTS FOR (c:Composition) REQUIRE c.name IS UNIQUE",
            "CREATE CONSTRAINT manufacturer_name IF NOT EXISTS FOR (man:Manufacturer) REQUIRE man.name IS UNIQUE",
            "CREATE CONSTRAINT side_effect_name IF NOT EXISTS FOR (s:SideEffect) REQUIRE s.name IS UNIQUE"
        ]
        
        indexes_queries = [
            "CREATE INDEX disease_name_index IF NOT EXISTS FOR (d:Disease) ON (d.name)",
            "CREATE INDEX medicine_name_index IF NOT EXISTS FOR (m:Medicine) ON (m.name)",
            "CREATE FULLTEXT INDEX disease_fulltext IF NOT EXISTS FOR (d:Disease) ON EACH [d.name, d.normalized_name]",
            "CREATE FULLTEXT INDEX medicine_fulltext IF NOT EXISTS FOR (m:Medicine) ON EACH [m.name, m.normalized_name]"
        ]
        
        for query in constraints_queries + indexes_queries:
            try:
                self.db.query(query)
                print(f"✓ Executed: {query}")
            except Exception as e:
                print(f"✗ Failed to execute {query}: {e}")
    
    def load_csv_to_graph(self, csv_path: str):
        """Load CSV data into Neo4j graph with enhanced parsing"""
        print("Loading CSV data into Neo4j graph...")
        
        # Create constraints and indexes first
        self.create_enhanced_graph_constraints()
        
        # Clear existing data
        print("Clearing existing data...")
        self.db.query("MATCH (n) DETACH DELETE n")
        
        # Read CSV
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} medicines in CSV")
        
        successful_loads = 0
        failed_loads = 0
        
        for index, row in df.iterrows():
            try:
                medicine = str(row['Medicine Name']).strip()
                composition = str(row['Composition']).strip() if pd.notna(row['Composition']) else "Unknown"
                manufacturer = str(row['Manufacturer']).strip() if pd.notna(row['Manufacturer']) else "Unknown"
                
                # Enhanced parsing of uses and side effects
                uses_list = self.parse_uses_field(str(row['Uses']) if pd.notna(row['Uses']) else "")
                effects_list = self.parse_side_effects_field(str(row['Side_effects']) if pd.notna(row['Side_effects']) else "")
                
                # Create medicine node with additional properties
                medicine_query = """
                MERGE (m:Medicine {name: $medicine})
                SET m.normalized_name = toLower($medicine),
                    m.composition = $composition,
                    m.manufacturer = $manufacturer
                """
                
                self.db.query(medicine_query, {
                    'medicine': medicine,
                    'composition': composition,
                    'manufacturer': manufacturer
                })
                
                # Create composition and manufacturer relationships
                comp_query = """
                MERGE (c:Composition {name: $composition})
                MERGE (man:Manufacturer {name: $manufacturer})
                MERGE (m:Medicine {name: $medicine})
                MERGE (m)-[:HAS_COMPOSITION]->(c)
                MERGE (m)-[:MANUFACTURED_BY]->(man)
                """
                
                self.db.query(comp_query, {
                    'medicine': medicine,
                    'composition': composition,
                    'manufacturer': manufacturer
                })
                
                # Create disease relationships with enhanced matching
                for disease in uses_list:
                    disease_query = """
                    MERGE (d:Disease {name: $disease})
                    SET d.normalized_name = toLower($disease)
                    WITH d
                    MERGE (m:Medicine {name: $medicine})
                    MERGE (m)-[:USED_FOR]->(d)
                    """
                    
                    self.db.query(disease_query, {
                        'medicine': medicine,
                        'disease': disease
                    })
                
                # Create side effect relationships
                for effect in effects_list:
                    effect_query = """
                    MERGE (s:SideEffect {name: $effect})
                    SET s.normalized_name = toLower($effect)
                    WITH s
                    MERGE (m:Medicine {name: $medicine})
                    MERGE (m)-[:HAS_SIDE_EFFECT]->(s)
                    """
                    
                    self.db.query(effect_query, {
                        'medicine': medicine,
                        'effect': effect
                    })
                
                successful_loads += 1
                
                if successful_loads % 100 == 0:
                    print(f"Processed {successful_loads} medicines...")
                    
            except Exception as e:
                failed_loads += 1
                print(f"Failed to process medicine {medicine}: {e}")
        
        print(f"✓ Successfully loaded {successful_loads} medicines")
        print(f"✗ Failed to load {failed_loads} medicines")
        
        # Verify lung cancer data
        self.verify_lung_cancer_data()
    
    def verify_lung_cancer_data(self):
        """Verify that lung cancer data is properly loaded"""
        print("\n=== Verifying Lung Cancer Data ===")
        
        # Check for lung cancer variations
        lung_cancer_queries = [
            "MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'lung cancer' RETURN d.name as disease",
            "MATCH (d:Disease) WHERE toLower(d.name) CONTAINS 'lung' RETURN d.name as disease",
            "MATCH (m:Medicine)-[:USED_FOR]->(d:Disease) WHERE toLower(d.name) CONTAINS 'lung' RETURN m.name as medicine, d.name as disease"
        ]
        
        for i, query in enumerate(lung_cancer_queries, 1):
            print(f"\nQuery {i}: {query}")
            results = self.db.query(query)
            if results:
                for result in results[:5]:  # Show first 5 results
                    print(f"  Result: {result}")
            else:
                print("  No results found")
    
    def semantic_search_diseases(self, query: str, limit: int = 10) -> List[Dict[str, Any]]:
        """Perform semantic search for diseases using fuzzy matching"""
        search_query = """
        MATCH (d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($query)
           OR toLower(d.normalized_name) CONTAINS toLower($query)
        RETURN d.name as disease, 
               size((d)<-[:USED_FOR]-()) as medicine_count
        ORDER BY medicine_count DESC
        LIMIT $limit
        """
        
        results = self.db.query(search_query, {'query': query, 'limit': limit})
        return [dict(result) for result in results]
    
    def get_medicines_for_disease(self, disease_query: str) -> List[Dict[str, Any]]:
        """Get medicines that treat a specific disease with fuzzy matching"""
        medicine_query = """
        MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($disease_query)
           OR toLower(d.normalized_name) CONTAINS toLower($disease_query)
        OPTIONAL MATCH (m)-[:HAS_COMPOSITION]->(c:Composition)
        OPTIONAL MATCH (m)-[:MANUFACTURED_BY]->(man:Manufacturer)
        OPTIONAL MATCH (m)-[:HAS_SIDE_EFFECT]->(s:SideEffect)
        RETURN m.name as medicine,
               d.name as disease_treated,
               c.name as composition,
               man.name as manufacturer,
               collect(DISTINCT s.name) as side_effects
        ORDER BY m.name
        """
        
        results = self.db.query(medicine_query, {'disease_query': disease_query})
        return [dict(result) for result in results]
    
    def enhanced_disease_search(self, query: str) -> Dict[str, Any]:
        """Enhanced search that combines exact and fuzzy matching"""
        # Try exact match first
        exact_query = """
        MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
        WHERE toLower(d.name) = toLower($query)
        RETURN m.name as medicine, d.name as disease
        """
        
        exact_results = self.db.query(exact_query, {'query': query})
        
        # Try fuzzy match
        fuzzy_query = """
        MATCH (m:Medicine)-[:USED_FOR]->(d:Disease)
        WHERE toLower(d.name) CONTAINS toLower($query)
        RETURN m.name as medicine, d.name as disease
        ORDER BY length(d.name) ASC
        """
        
        fuzzy_results = self.db.query(fuzzy_query, {'query': query})
        
        return {
            'exact_matches': [dict(r) for r in exact_results],
            'fuzzy_matches': [dict(r) for r in fuzzy_results],
            'total_medicines': len(set([r['medicine'] for r in exact_results + fuzzy_results]))
        }

if __name__ == "__main__":
    # Test the loader
    loader = GraphRAGLoader()
    loader.load_csv_to_graph("Medicine_Details.csv")
    
    # Test lung cancer search
    print("\n=== Testing Lung Cancer Search ===")
    results = loader.enhanced_disease_search("lung cancer")
    print(f"Exact matches: {len(results['exact_matches'])}")
    print(f"Fuzzy matches: {len(results['fuzzy_matches'])}")
    print(f"Total medicines: {results['total_medicines']}")
    
    for match in results['fuzzy_matches'][:5]:
        print(f"  {match['medicine']} -> {match['disease']}")
