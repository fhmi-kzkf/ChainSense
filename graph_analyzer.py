"""
Graph analysis and metrics calculation for ChainSense Supply Chain Risk Analyzer
"""

import networkx as nx
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import streamlit as st


class GraphAnalyzer:
    """Class to perform graph analysis and calculate supply chain metrics"""
    
    def __init__(self, graph: nx.DiGraph, node_types: Dict[str, str]):
        self.graph = graph
        self.node_types = node_types
        self.metrics = {}
        self.risk_scores = {}
        self.communities = {}
        self.anomalies = {}
    
    def calculate_basic_metrics(self) -> Dict:
        """Calculate basic graph metrics"""
        if self.graph is None or self.graph.number_of_nodes() == 0:
            return {}
        
        try:
            # Basic graph properties
            metrics = {
                'nodes': self.graph.number_of_nodes(),
                'edges': self.graph.number_of_edges(),
                'density': nx.density(self.graph),
                'is_weakly_connected': nx.is_weakly_connected(self.graph),
                'connected_components': nx.number_weakly_connected_components(self.graph)
            }
            
            # Centrality measures
            metrics['degree_centrality'] = nx.degree_centrality(self.graph)
            metrics['in_degree_centrality'] = nx.in_degree_centrality(self.graph)
            metrics['out_degree_centrality'] = nx.out_degree_centrality(self.graph)
            
            # Betweenness centrality (computationally expensive for large graphs)
            if self.graph.number_of_nodes() <= 1000:
                metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph)
            else:
                # Use approximation for large graphs
                k = min(100, self.graph.number_of_nodes() // 10)
                metrics['betweenness_centrality'] = nx.betweenness_centrality(self.graph, k=k)
            
            # Closeness centrality
            if nx.is_weakly_connected(self.graph) and self.graph.number_of_nodes() <= 1000:
                metrics['closeness_centrality'] = nx.closeness_centrality(self.graph)
            
            # PageRank
            try:
                metrics['pagerank'] = nx.pagerank(self.graph)
            except:
                pass
            
            # Clustering coefficient
            undirected_graph = self.graph.to_undirected()
            metrics['clustering'] = nx.clustering(undirected_graph)
            metrics['average_clustering'] = nx.average_clustering(undirected_graph)
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            st.error(f"Error calculating basic metrics: {str(e)}")
            return {}
    
    def get_top_nodes(self, metric: str, n: int = 10) -> List[Tuple[str, float]]:
        """Get top N nodes by a specific metric"""
        if metric not in self.metrics:
            return []
        
        metric_values = self.metrics[metric]
        if isinstance(metric_values, dict):
            sorted_nodes = sorted(metric_values.items(), key=lambda x: x[1], reverse=True)
            return sorted_nodes[:n]
        return []
    
    def calculate_risk_scores(self) -> Dict[str, float]:
        """Calculate risk scores for each node based on multiple factors"""
        if not self.metrics:
            self.calculate_basic_metrics()
        
        risk_scores = {}
        
        try:
            for node in self.graph.nodes():
                score = 0.0
                
                # High centrality = higher risk (critical nodes)
                if 'degree_centrality' in self.metrics:
                    score += self.metrics['degree_centrality'].get(node, 0) * 0.3
                
                if 'betweenness_centrality' in self.metrics:
                    score += self.metrics['betweenness_centrality'].get(node, 0) * 0.4
                
                # High PageRank = higher risk
                if 'pagerank' in self.metrics:
                    score += self.metrics['pagerank'].get(node, 0) * 10 * 0.2
                
                # Node type considerations
                node_type = self.node_types.get(node, 'unknown')
                if node_type == 'supplier':
                    score += 0.1  # Suppliers are inherently more risky
                elif node_type == 'intermediary':
                    score += 0.15  # Intermediaries can be bottlenecks
                elif node_type == 'isolated':
                    score += 0.2  # Isolated nodes are vulnerable
                
                # Connectivity risk (too few or too many connections)
                degree = self.graph.degree(node)
                if degree == 1:
                    score += 0.1  # Single point of failure
                elif degree > 20:
                    score += 0.05  # Overloaded node
                
                risk_scores[node] = min(score, 1.0)  # Cap at 1.0
            
            self.risk_scores = risk_scores
            return risk_scores
            
        except Exception as e:
            st.error(f"Error calculating risk scores: {str(e)}")
            return {}
    
    def detect_communities(self, algorithm: str = 'louvain') -> Dict:
        """Detect communities in the supply chain network"""
        try:
            undirected_graph = self.graph.to_undirected()
            
            if algorithm == 'louvain':
                try:
                    import community as community_louvain
                    communities = community_louvain.best_partition(undirected_graph)
                except ImportError:
                    # Fallback to networkx community detection
                    from networkx.algorithms import community
                    communities_generator = community.greedy_modularity_communities(undirected_graph)
                    communities = {}
                    for i, comm in enumerate(communities_generator):
                        for node in comm:
                            communities[node] = i
            
            elif algorithm == 'label_propagation':
                from networkx.algorithms import community
                communities_generator = community.label_propagation_communities(undirected_graph)
                communities = {}
                for i, comm in enumerate(communities_generator):
                    for node in comm:
                        communities[node] = i
            
            else:
                # Default to connected components
                communities = {}
                for i, component in enumerate(nx.weakly_connected_components(self.graph)):
                    for node in component:
                        communities[node] = i
            
            # Calculate community statistics
            community_stats = {}
            for comm_id in set(communities.values()):
                nodes_in_comm = [node for node, comm in communities.items() if comm == comm_id]
                subgraph = self.graph.subgraph(nodes_in_comm)
                
                community_stats[comm_id] = {
                    'size': len(nodes_in_comm),
                    'nodes': nodes_in_comm,
                    'density': nx.density(subgraph),
                    'node_types': {node_type: sum(1 for node in nodes_in_comm 
                                                if self.node_types.get(node) == node_type)
                                 for node_type in ['supplier', 'customer', 'intermediary', 'isolated']}
                }
            
            self.communities = {
                'assignments': communities,
                'stats': community_stats,
                'num_communities': len(set(communities.values()))
            }
            
            return self.communities
            
        except Exception as e:
            st.error(f"Error detecting communities: {str(e)}")
            return {}
    
    def detect_anomalies(self, method: str = 'isolation_forest') -> Dict:
        """Detect anomalous nodes in the supply chain"""
        try:
            # Prepare feature matrix
            features = []
            node_names = []
            
            for node in self.graph.nodes():
                node_features = [
                    self.graph.degree(node),
                    self.graph.in_degree(node),
                    self.graph.out_degree(node),
                    self.metrics.get('degree_centrality', {}).get(node, 0),
                    self.metrics.get('betweenness_centrality', {}).get(node, 0),
                    self.metrics.get('pagerank', {}).get(node, 0) * 100,
                    self.metrics.get('clustering', {}).get(node, 0)
                ]
                
                features.append(node_features)
                node_names.append(node)
            
            features_array = np.array(features)
            
            # Standardize features
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features_array)
            
            anomalies = {}
            
            if method == 'isolation_forest':
                clf = IsolationForest(contamination=0.1, random_state=42)
                anomaly_labels = clf.fit_predict(features_scaled)
                anomaly_scores = clf.score_samples(features_scaled)
                
                for i, (node, label, score) in enumerate(zip(node_names, anomaly_labels, anomaly_scores)):
                    anomalies[node] = {
                        'is_anomaly': label == -1,
                        'anomaly_score': abs(score),  # Convert to positive for easier interpretation
                        'method': 'isolation_forest'
                    }
            
            elif method == 'lof':
                clf = LocalOutlierFactor(contamination=0.1)
                anomaly_labels = clf.fit_predict(features_scaled)
                anomaly_scores = clf.negative_outlier_factor_
                
                for i, (node, label, score) in enumerate(zip(node_names, anomaly_labels, anomaly_scores)):
                    anomalies[node] = {
                        'is_anomaly': label == -1,
                        'anomaly_score': abs(score),
                        'method': 'local_outlier_factor'
                    }
            
            self.anomalies = anomalies
            return anomalies
            
        except Exception as e:
            st.error(f"Error detecting anomalies: {str(e)}")
            return {}
    
    def get_vulnerability_analysis(self) -> Dict:
        """Analyze network vulnerability and critical paths"""
        try:
            vulnerabilities = {}
            
            # Single points of failure (articulation points)
            undirected_graph = self.graph.to_undirected()
            articulation_points = list(nx.articulation_points(undirected_graph))
            
            # Bridges (critical edges)
            bridges = list(nx.bridges(undirected_graph))
            
            # Nodes with highest betweenness (bottlenecks)
            if 'betweenness_centrality' in self.metrics:
                bottlenecks = self.get_top_nodes('betweenness_centrality', 5)
            else:
                bottlenecks = []
            
            # Critical suppliers (high out-degree, low redundancy)
            critical_suppliers = []
            for node in self.graph.nodes():
                if self.node_types.get(node) == 'supplier':
                    out_degree = self.graph.out_degree(node)
                    if out_degree >= 3:  # Serving multiple customers
                        critical_suppliers.append((node, out_degree))
            
            critical_suppliers.sort(key=lambda x: x[1], reverse=True)
            
            vulnerabilities = {
                'articulation_points': articulation_points,
                'bridges': bridges,
                'bottlenecks': bottlenecks,
                'critical_suppliers': critical_suppliers[:10],
                'vulnerability_score': self._calculate_vulnerability_score(
                    len(articulation_points), len(bridges), len(bottlenecks)
                )
            }
            
            return vulnerabilities
            
        except Exception as e:
            st.error(f"Error in vulnerability analysis: {str(e)}")
            return {}
    
    def _calculate_vulnerability_score(self, articulation_points: int, bridges: int, bottlenecks: int) -> float:
        """Calculate overall network vulnerability score"""
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()
        
        if total_nodes == 0:
            return 0.0
        
        # Normalize by network size
        ap_ratio = articulation_points / total_nodes
        bridge_ratio = bridges / max(total_edges, 1)
        bottleneck_ratio = bottlenecks / total_nodes
        
        # Weighted vulnerability score
        vulnerability = (ap_ratio * 0.4 + bridge_ratio * 0.3 + bottleneck_ratio * 0.3)
        
        return min(vulnerability, 1.0)
    
    def get_supply_chain_insights(self) -> Dict:
        """Generate high-level insights about the supply chain"""
        insights = {}
        
        try:
            # Network structure insights
            num_suppliers = sum(1 for nt in self.node_types.values() if nt == 'supplier')
            num_customers = sum(1 for nt in self.node_types.values() if nt == 'customer')
            num_intermediaries = sum(1 for nt in self.node_types.values() if nt == 'intermediary')
            
            insights['structure'] = {
                'supplier_ratio': num_suppliers / self.graph.number_of_nodes(),
                'customer_ratio': num_customers / self.graph.number_of_nodes(),
                'intermediary_ratio': num_intermediaries / self.graph.number_of_nodes(),
                'supply_chain_depth': self._estimate_supply_chain_depth()
            }
            
            # Risk distribution
            if self.risk_scores:
                risk_values = list(self.risk_scores.values())
                insights['risk_distribution'] = {
                    'mean_risk': np.mean(risk_values),
                    'std_risk': np.std(risk_values),
                    'high_risk_nodes': sum(1 for r in risk_values if r > 0.7),
                    'low_risk_nodes': sum(1 for r in risk_values if r < 0.3)
                }
            
            # Connectivity patterns
            degrees = [self.graph.degree(node) for node in self.graph.nodes()]
            insights['connectivity'] = {
                'mean_degree': np.mean(degrees),
                'degree_variance': np.var(degrees),
                'isolated_nodes': sum(1 for d in degrees if d == 0),
                'highly_connected_nodes': sum(1 for d in degrees if d > np.mean(degrees) + 2 * np.std(degrees))
            }
            
            return insights
            
        except Exception as e:
            st.error(f"Error generating insights: {str(e)}")
            return {}
    
    def _estimate_supply_chain_depth(self) -> int:
        """Estimate the depth/tiers of the supply chain"""
        try:
            # Find all suppliers (nodes with no incoming edges)
            suppliers = [node for node in self.graph.nodes() if self.graph.in_degree(node) == 0]
            
            if not suppliers:
                return 0
            
            # Calculate shortest path lengths from suppliers to all other nodes
            max_depth = 0
            for supplier in suppliers:
                try:
                    lengths = nx.single_source_shortest_path_length(self.graph, supplier)
                    supplier_max_depth = max(lengths.values()) if lengths else 0
                    max_depth = max(max_depth, supplier_max_depth)
                except:
                    continue
            
            return max_depth + 1  # Add 1 because depth is 0-indexed
            
        except:
            return 0