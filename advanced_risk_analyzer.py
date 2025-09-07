"""
Advanced risk analysis utilities for ChainSense Supply Chain Risk Analyzer
Level 2 features: Advanced risk scoring, predictive analytics, and scenario analysis
"""

import pandas as pd
import numpy as np
import networkx as nx
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class AdvancedRiskAnalyzer:
    """Advanced risk analysis for supply chain networks"""
    
    def __init__(self, graph: nx.DiGraph, node_types: Dict[str, str], 
                 metrics: Dict = None, historical_data: pd.DataFrame = None):
        self.graph = graph
        self.node_types = node_types
        self.metrics = metrics or {}
        self.historical_data = historical_data
        self.risk_factors = {}
        self.scenario_results = {}
        
    def calculate_advanced_risk_scores(self) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive risk scores with multiple dimensions"""
        risk_dimensions = {}
        
        try:
            # 1. Structural Risk (network position)
            structural_risk = self._calculate_structural_risk()
            
            # 2. Operational Risk (based on transaction patterns)
            operational_risk = self._calculate_operational_risk()
            
            # 3. Concentration Risk (dependency analysis)
            concentration_risk = self._calculate_concentration_risk()
            
            # 4. Connectivity Risk (isolation and redundancy)
            connectivity_risk = self._calculate_connectivity_risk()
            
            # 5. Composite Risk Score
            composite_risk = self._calculate_composite_risk(
                structural_risk, operational_risk, concentration_risk, connectivity_risk
            )
            
            risk_dimensions = {
                'structural': structural_risk,
                'operational': operational_risk,
                'concentration': concentration_risk,
                'connectivity': connectivity_risk,
                'composite': composite_risk
            }
            
            self.risk_factors = risk_dimensions
            return risk_dimensions
            
        except Exception as e:
            print(f"Error calculating advanced risk scores: {str(e)}")
            return {}
    
    def _calculate_structural_risk(self) -> Dict[str, float]:
        """Calculate risk based on network structural properties"""
        structural_risk = {}
        
        for node in self.graph.nodes():
            risk_score = 0.0
            
            # Centrality-based risk
            if 'betweenness_centrality' in self.metrics:
                betweenness = self.metrics['betweenness_centrality'].get(node, 0)
                risk_score += betweenness * 0.4  # High betweenness = bottleneck risk
            
            if 'degree_centrality' in self.metrics:
                degree_cent = self.metrics['degree_centrality'].get(node, 0)
                risk_score += degree_cent * 0.3  # High degree = critical node
            
            # Clustering coefficient (low clustering = vulnerable)
            if 'clustering' in self.metrics:
                clustering = self.metrics['clustering'].get(node, 0)
                risk_score += (1 - clustering) * 0.2  # Low clustering increases risk
            
            # Node type specific risk
            node_type = self.node_types.get(node, 'unknown')
            if node_type == 'supplier':
                risk_score += 0.1  # Suppliers have inherent risk
            elif node_type == 'intermediary':
                risk_score += 0.15  # Intermediaries can be bottlenecks
            
            structural_risk[node] = min(risk_score, 1.0)
        
        return structural_risk
    
    def _calculate_operational_risk(self) -> Dict[str, float]:
        """Calculate risk based on operational patterns"""
        operational_risk = {}
        
        # If no historical data, use current graph structure
        for node in self.graph.nodes():
            risk_score = 0.0
            
            # Transaction volume volatility (if data available)
            in_degree = self.graph.in_degree(node)
            out_degree = self.graph.out_degree(node)
            total_degree = in_degree + out_degree
            
            # Risk increases with degree imbalance
            if total_degree > 0:
                imbalance = abs(in_degree - out_degree) / total_degree
                risk_score += imbalance * 0.3
            
            # Single customer/supplier dependency
            if out_degree == 1:
                risk_score += 0.4  # High dependency on single customer
            if in_degree == 1:
                risk_score += 0.3  # High dependency on single supplier
            
            # Overload risk (too many connections)
            if total_degree > 20:
                risk_score += 0.2
            
            operational_risk[node] = min(risk_score, 1.0)
        
        return operational_risk
    
    def _calculate_concentration_risk(self) -> Dict[str, float]:
        """Calculate risk based on market concentration"""
        concentration_risk = {}
        
        for node in self.graph.nodes():
            risk_score = 0.0
            
            # Calculate Herfindahl-Hirschman Index for neighbors
            neighbors = list(self.graph.neighbors(node))
            if neighbors:
                # Market share approximation (equal weight)
                market_shares = [1/len(neighbors) for _ in neighbors]
                hhi = sum(share**2 for share in market_shares)
                
                # High HHI indicates concentration risk
                if hhi > 0.25:  # Concentrated market
                    risk_score += 0.4
                elif hhi > 0.15:  # Moderately concentrated
                    risk_score += 0.2
            
            # Geographic concentration (simplified - based on node clustering)
            local_clustering = 0
            if 'clustering' in self.metrics:
                local_clustering = self.metrics['clustering'].get(node, 0)
            
            # High clustering might indicate geographic concentration
            if local_clustering > 0.7:
                risk_score += 0.3
            
            concentration_risk[node] = min(risk_score, 1.0)
        
        return concentration_risk
    
    def _calculate_connectivity_risk(self) -> Dict[str, float]:
        """Calculate risk based on connectivity patterns"""
        connectivity_risk = {}
        
        # Find articulation points and bridges
        undirected_graph = self.graph.to_undirected()
        articulation_points = set(nx.articulation_points(undirected_graph))
        bridges = set(nx.bridges(undirected_graph))
        
        for node in self.graph.nodes():
            risk_score = 0.0
            
            # Articulation point risk
            if node in articulation_points:
                risk_score += 0.5
            
            # Bridge involvement risk
            node_bridges = [bridge for bridge in bridges if node in bridge]
            if node_bridges:
                risk_score += len(node_bridges) * 0.2
            
            # Redundancy analysis
            degree = self.graph.degree(node)
            if degree <= 2:
                risk_score += 0.3  # Low redundancy
            
            # Distance to critical nodes
            try:
                # Calculate average distance to high-centrality nodes
                high_centrality_nodes = []
                if 'betweenness_centrality' in self.metrics:
                    sorted_nodes = sorted(
                        self.metrics['betweenness_centrality'].items(),
                        key=lambda x: x[1], reverse=True
                    )[:5]  # Top 5 most central nodes
                    high_centrality_nodes = [n for n, _ in sorted_nodes]
                
                if high_centrality_nodes:
                    distances = []
                    for central_node in high_centrality_nodes:
                        if central_node != node:
                            try:
                                dist = nx.shortest_path_length(self.graph, node, central_node)
                                distances.append(dist)
                            except nx.NetworkXNoPath:
                                distances.append(float('inf'))
                    
                    if distances:
                        avg_distance = np.mean([d for d in distances if d != float('inf')])
                        if avg_distance > 3:  # Far from critical nodes
                            risk_score += 0.2
                            
            except:
                pass
            
            connectivity_risk[node] = min(risk_score, 1.0)
        
        return connectivity_risk
    
    def _calculate_composite_risk(self, structural: Dict, operational: Dict, 
                                concentration: Dict, connectivity: Dict) -> Dict[str, float]:
        """Calculate weighted composite risk score"""
        composite_risk = {}
        
        # Risk dimension weights
        weights = {
            'structural': 0.3,
            'operational': 0.25,
            'concentration': 0.25,
            'connectivity': 0.2
        }
        
        for node in self.graph.nodes():
            score = (
                structural.get(node, 0) * weights['structural'] +
                operational.get(node, 0) * weights['operational'] +
                concentration.get(node, 0) * weights['concentration'] +
                connectivity.get(node, 0) * weights['connectivity']
            )
            
            composite_risk[node] = min(score, 1.0)
        
        return composite_risk
    
    def simulate_disruption_scenarios(self, scenarios: List[Dict]) -> Dict:
        """Simulate various disruption scenarios"""
        scenario_results = {}
        
        for i, scenario in enumerate(scenarios):
            scenario_name = scenario.get('name', f'Scenario_{i+1}')
            disrupted_nodes = scenario.get('nodes', [])
            disruption_type = scenario.get('type', 'removal')  # 'removal' or 'capacity_reduction'
            
            result = self._simulate_single_scenario(disrupted_nodes, disruption_type)
            scenario_results[scenario_name] = result
        
        self.scenario_results = scenario_results
        return scenario_results
    
    def _simulate_single_scenario(self, disrupted_nodes: List[str], 
                                disruption_type: str) -> Dict:
        """Simulate a single disruption scenario"""
        # Create a copy of the graph
        graph_copy = self.graph.copy()
        
        # Apply disruption
        if disruption_type == 'removal':
            graph_copy.remove_nodes_from(disrupted_nodes)
        
        # Analyze impact
        original_nodes = self.graph.number_of_nodes()
        original_edges = self.graph.number_of_edges()
        original_components = nx.number_weakly_connected_components(self.graph)
        
        new_nodes = graph_copy.number_of_nodes()
        new_edges = graph_copy.number_of_edges()
        new_components = nx.number_weakly_connected_components(graph_copy)
        
        # Calculate impact metrics
        impact = {
            'nodes_removed': len(disrupted_nodes),
            'edges_lost': original_edges - new_edges,
            'connectivity_loss': (original_edges - new_edges) / original_edges if original_edges > 0 else 0,
            'fragmentation': new_components - original_components,
            'largest_component_size': len(max(nx.weakly_connected_components(graph_copy), key=len)) if new_nodes > 0 else 0,
            'isolation_impact': self._calculate_isolation_impact(graph_copy),
            'criticality_score': self._calculate_scenario_criticality(disrupted_nodes)
        }
        
        return impact
    
    def _calculate_isolation_impact(self, disrupted_graph: nx.DiGraph) -> float:
        """Calculate the impact of node isolation"""
        isolated_nodes = [node for node in disrupted_graph.nodes() 
                         if disrupted_graph.degree(node) == 0]
        
        total_nodes = self.graph.number_of_nodes()
        isolation_ratio = len(isolated_nodes) / total_nodes if total_nodes > 0 else 0
        
        return isolation_ratio
    
    def _calculate_scenario_criticality(self, disrupted_nodes: List[str]) -> float:
        """Calculate criticality score for the scenario"""
        criticality = 0.0
        
        for node in disrupted_nodes:
            # Add centrality-based criticality
            if 'betweenness_centrality' in self.metrics:
                criticality += self.metrics['betweenness_centrality'].get(node, 0)
            
            if 'degree_centrality' in self.metrics:
                criticality += self.metrics['degree_centrality'].get(node, 0)
            
            # Add type-based criticality
            node_type = self.node_types.get(node, 'unknown')
            if node_type == 'supplier':
                criticality += 0.3
            elif node_type == 'intermediary':
                criticality += 0.4
        
        return min(criticality, 1.0)
    
    def identify_critical_paths(self, max_paths: int = 10) -> List[Dict]:
        """Identify critical paths in the supply chain"""
        critical_paths = []
        
        try:
            # Find all suppliers (source nodes)
            suppliers = [node for node in self.graph.nodes() 
                        if self.graph.in_degree(node) == 0]
            
            # Find all customers (sink nodes)  
            customers = [node for node in self.graph.nodes() 
                        if self.graph.out_degree(node) == 0]
            
            path_criticalities = []
            
            for supplier in suppliers[:5]:  # Limit to top 5 suppliers
                for customer in customers[:5]:  # Limit to top 5 customers
                    try:
                        # Find shortest path
                        path = nx.shortest_path(self.graph, supplier, customer)
                        
                        # Calculate path criticality
                        path_criticality = self._calculate_path_criticality(path)
                        
                        path_info = {
                            'path': path,
                            'length': len(path) - 1,
                            'criticality': path_criticality,
                            'supplier': supplier,
                            'customer': customer
                        }
                        
                        path_criticalities.append(path_info)
                        
                    except nx.NetworkXNoPath:
                        continue
            
            # Sort by criticality and return top paths
            path_criticalities.sort(key=lambda x: x['criticality'], reverse=True)
            critical_paths = path_criticalities[:max_paths]
            
        except Exception as e:
            print(f"Error identifying critical paths: {str(e)}")
        
        return critical_paths
    
    def _calculate_path_criticality(self, path: List[str]) -> float:
        """Calculate criticality score for a path"""
        criticality = 0.0
        
        for node in path:
            # Node criticality based on centrality
            if 'betweenness_centrality' in self.metrics:
                criticality += self.metrics['betweenness_centrality'].get(node, 0)
            
            # Degree-based criticality
            degree = self.graph.degree(node)
            if degree <= 2:  # Bottleneck nodes
                criticality += 0.3
        
        # Path length penalty (longer paths are more vulnerable)
        path_length = len(path) - 1
        length_penalty = path_length * 0.1
        
        return criticality + length_penalty
    
    def generate_resilience_recommendations(self) -> List[Dict]:
        """Generate recommendations to improve supply chain resilience"""
        recommendations = []
        
        try:
            # 1. Diversification recommendations
            diversification_recs = self._analyze_diversification_opportunities()
            recommendations.extend(diversification_recs)
            
            # 2. Redundancy recommendations
            redundancy_recs = self._analyze_redundancy_opportunities()
            recommendations.extend(redundancy_recs)
            
            # 3. Critical node protection
            protection_recs = self._analyze_protection_priorities()
            recommendations.extend(protection_recs)
            
            # 4. Network optimization
            optimization_recs = self._analyze_network_optimization()
            recommendations.extend(optimization_recs)
            
        except Exception as e:
            print(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _analyze_diversification_opportunities(self) -> List[Dict]:
        """Analyze opportunities for supplier/customer diversification"""
        recommendations = []
        
        # Find nodes with high concentration risk
        if 'concentration' in self.risk_factors:
            high_concentration_nodes = [
                node for node, risk in self.risk_factors['concentration'].items()
                if risk > 0.6
            ]
            
            for node in high_concentration_nodes[:5]:
                node_type = self.node_types.get(node, 'unknown')
                degree = self.graph.degree(node)
                
                recommendation = {
                    'type': 'diversification',
                    'priority': 'high' if self.risk_factors['concentration'][node] > 0.8 else 'medium',
                    'node': node,
                    'node_type': node_type,
                    'current_connections': degree,
                    'description': f"Diversify {node_type} base for {node} (currently {degree} connections)",
                    'impact': 'Reduces concentration risk and single point of failure'
                }
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_redundancy_opportunities(self) -> List[Dict]:
        """Analyze opportunities for adding redundant connections"""
        recommendations = []
        
        # Find articulation points
        undirected_graph = self.graph.to_undirected()
        articulation_points = list(nx.articulation_points(undirected_graph))
        
        for node in articulation_points[:3]:
            node_type = self.node_types.get(node, 'unknown')
            
            recommendation = {
                'type': 'redundancy',
                'priority': 'high',
                'node': node,
                'node_type': node_type,
                'description': f"Add redundant connections around critical {node_type} {node}",
                'impact': 'Eliminates single point of failure in network connectivity'
            }
            
            recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_protection_priorities(self) -> List[Dict]:
        """Analyze which nodes need priority protection"""
        recommendations = []
        
        if 'composite' in self.risk_factors:
            # Find highest risk nodes
            high_risk_nodes = sorted(
                self.risk_factors['composite'].items(),
                key=lambda x: x[1], reverse=True
            )[:3]
            
            for node, risk_score in high_risk_nodes:
                node_type = self.node_types.get(node, 'unknown')
                
                recommendation = {
                    'type': 'protection',
                    'priority': 'critical' if risk_score > 0.8 else 'high',
                    'node': node,
                    'node_type': node_type,
                    'risk_score': risk_score,
                    'description': f"Implement enhanced monitoring and protection for {node_type} {node}",
                    'impact': 'Reduces likelihood of disruption at critical network points'
                }
                
                recommendations.append(recommendation)
        
        return recommendations
    
    def _analyze_network_optimization(self) -> List[Dict]:
        """Analyze opportunities for network structure optimization"""
        recommendations = []
        
        # Analyze network density
        density = nx.density(self.graph)
        
        if density < 0.1:  # Sparse network
            recommendation = {
                'type': 'optimization',
                'priority': 'medium',
                'description': f"Network is sparse (density: {density:.3f}). Consider adding strategic connections",
                'impact': 'Improves overall network resilience and reduces path dependencies'
            }
            recommendations.append(recommendation)
        
        elif density > 0.5:  # Dense network
            recommendation = {
                'type': 'optimization',
                'priority': 'low',
                'description': f"Network is dense (density: {density:.3f}). Consider optimizing for efficiency",
                'impact': 'Reduces complexity while maintaining resilience'
            }
            recommendations.append(recommendation)
        
        return recommendations