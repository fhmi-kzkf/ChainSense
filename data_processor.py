"""
Data processing utilities for ChainSense Supply Chain Risk Analyzer
"""

import pandas as pd
import networkx as nx
import streamlit as st
from typing import Dict, List, Tuple, Optional
import numpy as np


class SupplyChainData:
    """Class to handle supply chain data processing and graph construction"""
    
    def __init__(self):
        self.df = None
        self.graph = None
        self.node_types = {}
        
    def load_csv(self, uploaded_file) -> bool:
        """Load CSV data from uploaded file"""
        try:
            self.df = pd.read_csv(uploaded_file)
            return True
        except Exception as e:
            st.error(f"Error loading CSV: {str(e)}")
            return False
    
    def validate_data(self) -> Tuple[bool, str]:
        """Validate if the data has the required columns for graph construction"""
        if self.df is None:
            return False, "No data loaded"
        
        # Check if we already have the required columns
        required_columns = ['supplier', 'customer']
        if all(col in self.df.columns for col in required_columns):
            # Remove rows with null values in required columns
            initial_rows = len(self.df)
            self.df = self.df.dropna(subset=required_columns)
            final_rows = len(self.df)
            
            if final_rows == 0:
                return False, "No valid data rows after removing null values"
            
            if initial_rows != final_rows:
                print(f"Removed {initial_rows - final_rows} rows with null values in required columns")
            
            return True, f"Data validated successfully. {final_rows} valid rows found."
        
        # If not, return False but don't fail - let the UI handle column mapping
        return False, f"Missing required columns. Available columns: {list(self.df.columns)}. Please use column mapping interface."
    
    def auto_detect_relationships(self) -> bool:
        """Automatically detect and create supplier-customer relationships from any dataset"""
        try:
            if self.df is None or len(self.df) == 0:
                return False
            
            # Strategy 1: Look for explicit supplier/customer columns
            supplier_cols = self._find_columns_by_keywords(['supplier', 'vendor', 'manufacturer', 'provider', 'source'])
            customer_cols = self._find_columns_by_keywords(['customer', 'buyer', 'client', 'retailer', 'distributor', 'market'])
            
            # Strategy 2: If we have supplier column but no customer, create logical customers
            if supplier_cols and not customer_cols:
                supplier_col = supplier_cols[0]
                relationships = []
                
                # Option A: Use location as customer
                location_cols = self._find_columns_by_keywords(['location', 'region', 'city', 'country', 'area'])
                if location_cols:
                    location_col = location_cols[0]
                    for _, row in self.df.iterrows():
                        supplier = self._clean_value(row.get(supplier_col))
                        location = self._clean_value(row.get(location_col))
                        if supplier and location:
                            relationships.append({
                                'supplier': supplier,
                                'customer': f"Market_{location}"
                            })
                
                # Option B: Use product type as customer
                elif 'Product type' in self.df.columns or 'product' in [c.lower() for c in self.df.columns]:
                    product_col = 'Product type' if 'Product type' in self.df.columns else next((c for c in self.df.columns if 'product' in c.lower()), None)
                    if product_col:
                        for _, row in self.df.iterrows():
                            supplier = self._clean_value(row.get(supplier_col))
                            product = self._clean_value(row.get(product_col))
                            if supplier and product:
                                relationships.append({
                                    'supplier': supplier,
                                    'customer': f"ProductLine_{product}"
                                })
                
                # Option C: Create generic market segments
                if not relationships:
                    unique_suppliers = self.df[supplier_col].dropna().unique()
                    for supplier in unique_suppliers:
                        supplier_clean = self._clean_value(supplier)
                        if supplier_clean:
                            relationships.append({
                                'supplier': supplier_clean,
                                'customer': f"Market_Segment_{len(relationships) % 5 + 1}"
                            })
                
                if relationships:
                    self.df = pd.DataFrame(relationships).drop_duplicates()
                    return True
            
            # Strategy 3: If we have both supplier and customer columns
            elif supplier_cols and customer_cols:
                supplier_col = supplier_cols[0]
                customer_col = customer_cols[0]
                
                # Create relationships
                relationships = []
                for _, row in self.df.iterrows():
                    supplier = self._clean_value(row.get(supplier_col))
                    customer = self._clean_value(row.get(customer_col))
                    if supplier and customer and supplier != customer:
                        relationships.append({
                            'supplier': supplier,
                            'customer': customer
                        })
                
                if relationships:
                    self.df = pd.DataFrame(relationships).drop_duplicates()
                    return True
            
            # Strategy 4: Create relationships from any two text columns with reasonable diversity
            text_columns = [col for col in self.df.columns if self.df[col].dtype == 'object']
            suitable_pairs = []
            
            for i, col1 in enumerate(text_columns):
                for col2 in text_columns[i+1:]:
                    # Check if columns have reasonable diversity
                    unique_ratio1 = self.df[col1].nunique() / len(self.df)
                    unique_ratio2 = self.df[col2].nunique() / len(self.df)
                    
                    if 0.1 <= unique_ratio1 <= 0.8 and 0.1 <= unique_ratio2 <= 0.8:
                        suitable_pairs.append((col1, col2, unique_ratio1 + unique_ratio2))
            
            if suitable_pairs:
                # Use the pair with best diversity balance
                suitable_pairs.sort(key=lambda x: abs(x[2] - 1.0))  # Prefer pairs closest to 50-50 split
                col1, col2, _ = suitable_pairs[0]
                
                relationships = []
                for _, row in self.df.iterrows():
                    val1 = self._clean_value(row.get(col1))
                    val2 = self._clean_value(row.get(col2))
                    if val1 and val2 and val1 != val2:
                        relationships.append({
                            'supplier': val1,
                            'customer': val2
                        })
                
                if relationships:
                    self.df = pd.DataFrame(relationships).drop_duplicates()
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error in auto-detection: {str(e)}")
            return False
    
    def _find_columns_by_keywords(self, keywords: List[str]) -> List[str]:
        """Find columns containing any of the given keywords"""
        found_columns = []
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in keywords):
                found_columns.append(col)
        return found_columns
    
    def _clean_value(self, value) -> str:
        """Clean and validate a value for use as node name"""
        if pd.isna(value) or value is None:
            return None
        
        cleaned = str(value).strip()
        if cleaned.lower() in ['', 'nan', 'null', 'none']:
            return None
        
        return cleaned
    
    def build_graph(self) -> bool:
        """Build NetworkX graph from the loaded data"""
        if self.df is None:
            return False
        
        try:
            self.graph = nx.DiGraph()
            
            # Add edges from supplier to customer
            for _, row in self.df.iterrows():
                supplier = str(row['supplier']).strip()
                customer = str(row['customer']).strip()
                
                # Skip if either is empty or NaN
                if not supplier or not customer or supplier == 'nan' or customer == 'nan':
                    continue
                
                # Add edge attributes if available
                edge_attrs = {}
                if 'quantity' in self.df.columns and pd.notna(row['quantity']):
                    edge_attrs['quantity'] = row['quantity']
                if 'price' in self.df.columns and pd.notna(row['price']):
                    edge_attrs['price'] = row['price']
                if 'product' in self.df.columns and pd.notna(row['product']):
                    edge_attrs['product'] = str(row['product'])
                if 'date' in self.df.columns and pd.notna(row['date']):
                    edge_attrs['date'] = str(row['date'])
                
                self.graph.add_edge(supplier, customer, **edge_attrs)
            
            # Classify nodes by type
            self._classify_nodes()
            
            return True
            
        except Exception as e:
            st.error(f"Error building graph: {str(e)}")
            return False
    
    def _classify_nodes(self):
        """Classify nodes as suppliers, customers, or intermediaries"""
        if self.graph is None:
            return
        
        self.node_types = {}
        
        for node in self.graph.nodes():
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            
            if in_degree == 0 and out_degree > 0:
                self.node_types[node] = 'supplier'
            elif out_degree == 0 and in_degree > 0:
                self.node_types[node] = 'customer'
            elif in_degree > 0 and out_degree > 0:
                self.node_types[node] = 'intermediary'
            else:
                self.node_types[node] = 'isolated'
    
    def get_data_summary(self) -> Dict:
        """Get summary statistics of the loaded data"""
        if self.df is None:
            return {}
        
        summary = {
            'total_rows': len(self.df),
            'unique_suppliers': self.df['supplier'].nunique() if 'supplier' in self.df.columns else 0,
            'unique_customers': self.df['customer'].nunique() if 'customer' in self.df.columns else 0,
            'date_range': None,
            'columns': list(self.df.columns)
        }
        
        if 'date' in self.df.columns:
            try:
                date_col = pd.to_datetime(self.df['date'], errors='coerce')
                summary['date_range'] = {
                    'start': date_col.min(),
                    'end': date_col.max()
                }
            except:
                pass
        
        return summary
        
    def suggest_column_mapping(self) -> Dict[str, List[str]]:
        """Suggest column mappings based on available columns"""
        if self.df is None:
            return {}
        
        available_cols = [col.lower() for col in self.df.columns]
        suggestions = {
            'supplier': [],
            'customer': []
        }
        
        # Look for supplier-like columns
        supplier_keywords = ['supplier', 'vendor', 'manufacturer', 'source', 'provider']
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in supplier_keywords):
                suggestions['supplier'].append(col)
        
        # Look for customer-like columns  
        customer_keywords = ['customer', 'buyer', 'client', 'retailer', 'distributor', 'destination']
        for col in self.df.columns:
            col_lower = col.lower()
            if any(keyword in col_lower for keyword in customer_keywords):
                suggestions['customer'].append(col)
        
        # If no direct matches, suggest based on data patterns
        if not suggestions['supplier'] and not suggestions['customer']:
            # Look for columns that might represent entities
            entity_like_cols = []
            for col in self.df.columns:
                if self.df[col].dtype == 'object':  # Text columns
                    unique_ratio = self.df[col].nunique() / len(self.df)
                    if 0.1 < unique_ratio < 0.8:  # Not too unique, not too repetitive
                        entity_like_cols.append(col)
            
            # Suggest the first few as potential supplier/customer columns
            if len(entity_like_cols) >= 2:
                suggestions['supplier'] = [entity_like_cols[0]]
                suggestions['customer'] = [entity_like_cols[1]]
            elif len(entity_like_cols) == 1:
                suggestions['supplier'] = [entity_like_cols[0]]
        
        return suggestions
        
    def create_supply_chain_from_products(self, supplier_col: str, location_col: str = None) -> bool:
        """Create supply chain relationships from product-focused data"""
        try:
            if supplier_col not in self.df.columns:
                return False
            
            # If we have location data, use it to create geographic relationships
            if location_col and location_col in self.df.columns:
                # Create supplier -> location relationships
                relationships = []
                
                for _, row in self.df.iterrows():
                    supplier = str(row[supplier_col]).strip()
                    location = str(row[location_col]).strip()
                    
                    if supplier and location and supplier != 'nan' and location != 'nan':
                        relationships.append({
                            'supplier': supplier,
                            'customer': f"Market_{location}",
                            'relationship_type': 'geographic'
                        })
                
                # Convert to DataFrame and assign
                if relationships:
                    new_df = pd.DataFrame(relationships)
                    self.df = new_df
                    return True
            
            # Alternative: Create supplier -> product type relationships
            if 'Product type' in self.df.columns:
                relationships = []
                
                for _, row in self.df.iterrows():
                    supplier = str(row[supplier_col]).strip()
                    product_type = str(row['Product type']).strip()
                    
                    if supplier and product_type and supplier != 'nan' and product_type != 'nan':
                        relationships.append({
                            'supplier': supplier,
                            'customer': f"ProductLine_{product_type}",
                            'relationship_type': 'product_supply'
                        })
                
                if relationships:
                    new_df = pd.DataFrame(relationships)
                    self.df = new_df
                    return True
            
            return False
            
        except Exception as e:
            print(f"Error creating supply chain relationships: {str(e)}")
            return False
    
    def get_graph_summary(self) -> Dict:
        """Get summary statistics of the graph"""
        if self.graph is None:
            return {}
        
        # Count nodes by type
        type_counts = {}
        for node_type in ['supplier', 'customer', 'intermediary', 'isolated']:
            type_counts[node_type] = sum(1 for nt in self.node_types.values() if nt == node_type)
        
        return {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'node_types': type_counts,
            'is_connected': nx.is_weakly_connected(self.graph),
            'connected_components': nx.number_weakly_connected_components(self.graph),
            'density': nx.density(self.graph)
        }
    
    def create_sample_data(self) -> pd.DataFrame:
        """Create sample supply chain data for testing"""
        np.random.seed(42)
        
        suppliers = ['Supplier_A', 'Supplier_B', 'Supplier_C', 'Supplier_D']
        distributors = ['Distributor_X', 'Distributor_Y', 'Distributor_Z']
        retailers = ['Retailer_1', 'Retailer_2', 'Retailer_3', 'Retailer_4', 'Retailer_5']
        products = ['Product_Alpha', 'Product_Beta', 'Product_Gamma']
        
        data = []
        
        # Supplier to Distributor connections
        for supplier in suppliers:
            for distributor in np.random.choice(distributors, size=np.random.randint(1, 3), replace=False):
                data.append({
                    'supplier': supplier,
                    'customer': distributor,
                    'product': np.random.choice(products),
                    'quantity': np.random.randint(100, 1000),
                    'price': np.random.uniform(10, 100),
                    'date': f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
                })
        
        # Distributor to Retailer connections
        for distributor in distributors:
            for retailer in np.random.choice(retailers, size=np.random.randint(2, 4), replace=False):
                data.append({
                    'supplier': distributor,
                    'customer': retailer,
                    'product': np.random.choice(products),
                    'quantity': np.random.randint(50, 500),
                    'price': np.random.uniform(15, 120),
                    'date': f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
                })
        
        # Add some direct supplier to retailer connections
        for _ in range(5):
            supplier = np.random.choice(suppliers)
            retailer = np.random.choice(retailers)
            data.append({
                'supplier': supplier,
                'customer': retailer,
                'product': np.random.choice(products),
                'quantity': np.random.randint(20, 200),
                'price': np.random.uniform(12, 90),
                'date': f"2024-{np.random.randint(1, 13):02d}-{np.random.randint(1, 29):02d}"
            })
        
        return pd.DataFrame(data)