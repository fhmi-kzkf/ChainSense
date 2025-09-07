"""
Visualization utilities for ChainSense Supply Chain Risk Analyzer
"""

import networkx as nx
import pandas as pd
import numpy as np
try:
    from pyvis.network import Network
except ImportError:
    Network = None
    print("Warning: PyVis not available, using fallback visualization")
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from typing import Dict, List, Tuple, Optional
import tempfile
import os


class SupplyChainVisualizer:
    """Class to handle all visualization tasks for supply chain analysis"""
    
    def __init__(self, graph: nx.DiGraph, node_types: Dict[str, str], 
                 metrics: Dict = None, risk_scores: Dict = None):
        self.graph = graph
        self.node_types = node_types
        self.metrics = metrics or {}
        self.risk_scores = risk_scores or {}
        
        # Color schemes
        self.node_colors = {
            'supplier': '#FF6B6B',      # Red
            'customer': '#4ECDC4',       # Teal
            'intermediary': '#45B7D1',   # Blue
            'isolated': '#96CEB4'        # Green
        }
        
        self.risk_colorscale = [
            [0, '#00CC96'],    # Low risk - Green
            [0.5, '#FFA15A'],  # Medium risk - Orange
            [1, '#EF553B']     # High risk - Red
        ]
    
    def create_interactive_graph(self, layout: str = 'spring', 
                                node_size_metric: str = 'degree_centrality',
                                show_risk: bool = False,
                                height: str = '600px') -> str:
        """Create interactive graph visualization using PyVis"""
        
        # Check if PyVis is available
        if Network is None:
            st.warning("PyVis is not available. Using fallback visualization.")
            return self._create_fallback_graph(layout, node_size_metric, show_risk, height)
        
        try:
            # Validate input parameters
            if self.graph is None or self.graph.number_of_nodes() == 0:
                st.warning("No graph data available for visualization")
                return None
            
            # Create PyVis network with version-compatible parameters
            try:
                # Try creating network with minimal parameters first
                net = Network()
                net.height = height
                net.width = '100%'
                net.bgcolor = '#ffffff'
                net.font_color = 'black'
            except Exception as net_error:
                st.warning(f"Could not create PyVis network with custom parameters: {str(net_error)}")
                try:
                    # Fallback to most basic network creation
                    net = Network(height=height)
                except Exception as basic_error:
                    st.error(f"Could not create basic PyVis network: {str(basic_error)}")
                    return self._create_fallback_graph(layout, node_size_metric, show_risk, height)
            
            # Verify network object was created properly
            if not hasattr(net, 'add_node') or not hasattr(net, 'add_edge'):
                st.error("PyVis Network object not created properly")
                return self._create_fallback_graph(layout, node_size_metric, show_risk, height)
            
            # Calculate node sizes
            node_sizes = self._calculate_node_sizes(node_size_metric)
            
            # Add nodes
            nodes_added = 0
            for node in self.graph.nodes():
                node_type = self.node_types.get(node, 'isolated')
                
                # Node color based on type or risk
                if show_risk and node in self.risk_scores:
                    risk_score = self.risk_scores[node]
                    color = self._risk_to_color(risk_score)
                else:
                    color = self.node_colors.get(node_type, '#CCCCCC')
                
                # Node size
                size = node_sizes.get(node, 10)
                
                # Tooltip with information
                tooltip = self._create_node_tooltip(node, node_type)
                
                try:
                    net.add_node(
                        node,
                        label=str(node),  # Ensure label is string
                        color=color,
                        size=size,
                        title=tooltip,
                        group=node_type
                    )
                    nodes_added += 1
                except Exception as node_error:
                    st.warning(f"Could not add node {node}: {str(node_error)}")
            
            # Verify we have nodes
            if nodes_added == 0:
                st.error("No nodes were successfully added to the visualization")
                return self._create_fallback_graph(layout, node_size_metric, show_risk, height)
            
            # Add edges
            edges_added = 0
            for edge in self.graph.edges(data=True):
                source, target, data = edge
                
                # Edge tooltip
                edge_tooltip = self._create_edge_tooltip(source, target, data)
                
                # Edge width based on weight if available
                width = 1
                if 'quantity' in data:
                    try:
                        width = max(1, min(5, data['quantity'] / 100))
                    except:
                        width = 1
                
                try:
                    net.add_edge(source, target, title=edge_tooltip, width=width)
                    edges_added += 1
                except Exception as edge_error:
                    st.warning(f"Could not add edge {source} -> {target}: {str(edge_error)}")
            
            # Verify we have a valid network
            if edges_added == 0 and self.graph.number_of_edges() > 0:
                st.warning("No edges were successfully added to the visualization")
            
            # Apply layout with simplified approach
            try:
                if layout == 'hierarchical':
                    # Simple hierarchical configuration
                    net.set_options("""
                    {
                      "layout": {
                        "hierarchical": {
                          "enabled": true,
                          "direction": "LR"
                        }
                      },
                      "physics": false
                    }
                    """)
                elif layout == 'circular':
                    # Enable physics for circular layout
                    net.set_options("""
                    {
                      "layout": {
                        "randomSeed": 2
                      },
                      "physics": {
                        "forceAtlas2Based": {
                          "gravitationalConstant": -26,
                          "centralGravity": 0.005,
                          "springLength": 230,
                          "springConstant": 0.18
                        },
                        "maxVelocity": 146,
                        "solver": "forceAtlas2Based"
                      }
                    }
                    """)
                else:
                    # Default spring layout with basic physics
                    net.set_options("""
                    {
                      "physics": {
                        "enabled": true,
                        "stabilization": {"iterations": 100}
                      }
                    }
                    """)
            except Exception as layout_error:
                # If any layout fails, use minimal configuration
                st.warning(f"Layout configuration failed: {str(layout_error)}. Using default layout.")
                pass  # Continue with default PyVis behavior
            
            # Save to temporary file with comprehensive error handling
            return self._save_network_safely(net)
            
        except Exception as e:
            st.error(f"Error creating interactive graph: {str(e)}")
            if 'bool' in str(e) and 'iterable' in str(e):
                st.info("Detected PyVis compatibility issue, using direct visualization method...")
                return self._create_direct_html_fallback()
            else:
                st.info("Trying fallback visualization method...")
                return self._create_fallback_graph(layout, node_size_metric, show_risk, height)
    
    def _create_fallback_graph(self, layout: str = 'spring', 
                              node_size_metric: str = 'degree_centrality',
                              show_risk: bool = False,
                              height: str = '600px') -> str:
        """Create a simple HTML graph as fallback when PyVis fails"""
        try:
            # Create a simple HTML visualization using networkx and matplotlib
            import matplotlib.pyplot as plt
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            
            # Calculate positions
            if layout == 'hierarchical':
                # Try to create a hierarchical layout using networkx
                try:
                    pos = nx.multipartite_layout(self.graph)
                except:
                    pos = nx.spring_layout(self.graph, k=1, iterations=50)
            elif layout == 'circular':
                pos = nx.circular_layout(self.graph)
            else:
                pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 8))
            fig.patch.set_facecolor('white')
            
            # Draw edges
            nx.draw_networkx_edges(self.graph, pos, ax=ax, edge_color='#888888', alpha=0.6, arrows=True)
            
            # Prepare node colors and sizes
            node_colors = []
            node_sizes = []
            
            for node in self.graph.nodes():
                node_type = self.node_types.get(node, 'isolated')
                
                # Node color
                if show_risk and node in self.risk_scores:
                    risk_score = self.risk_scores[node]
                    if risk_score < 0.3:
                        color = '#00CC96'
                    elif risk_score < 0.7:
                        color = '#FFA15A'
                    else:
                        color = '#EF553B'
                else:
                    color = self.node_colors.get(node_type, '#CCCCCC')
                
                node_colors.append(color)
                
                # Node size
                if node_size_metric == 'degree':
                    size = max(100, self.graph.degree(node) * 50)
                elif node_size_metric in self.metrics and isinstance(self.metrics[node_size_metric], dict):
                    metric_value = self.metrics[node_size_metric].get(node, 0)
                    size = max(100, metric_value * 500)
                else:
                    size = 200
                
                node_sizes.append(size)
            
            # Draw nodes
            nx.draw_networkx_nodes(self.graph, pos, ax=ax, 
                                 node_color=node_colors, 
                                 node_size=node_sizes,
                                 alpha=0.8)
            
            # Draw labels
            nx.draw_networkx_labels(self.graph, pos, ax=ax, font_size=8)
            
            # Set title and clean up axes
            ax.set_title(f"Supply Chain Network ({layout.title()} Layout)", fontsize=16, pad=20)
            ax.axis('off')
            
            # Save to temporary file
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w')
            
            # Convert matplotlib figure to HTML
            import io
            import base64
            
            img_buffer = io.BytesIO()
            plt.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            img_buffer.seek(0)
            img_data = base64.b64encode(img_buffer.read()).decode()
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ChainSense Network Visualization</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; text-align: center; }}
                    .graph-container {{ margin: 20px auto; }}
                    .info {{ background: #f0f0f0; padding: 10px; margin: 10px; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h2>ChainSense Network Visualization (Fallback Mode)</h2>
                <div class="info">
                    <p><strong>Layout:</strong> {layout.title()} | <strong>Node Size:</strong> {node_size_metric.replace('_', ' ').title()}</p>
                    <p><strong>Nodes:</strong> {self.graph.number_of_nodes()} | <strong>Edges:</strong> {self.graph.number_of_edges()}</p>
                </div>
                <div class="graph-container">
                    <img src="data:image/png;base64,{img_data}" alt="Network Graph" style="max-width: 100%; height: auto;">
                </div>
                <div class="info">
                    <p><em>This is a fallback visualization. For interactive features, please check your PyVis installation.</em></p>
                </div>
            </body>
            </html>
            """
            
            temp_file.write(html_content)
            temp_file.close()
            plt.close(fig)  # Clean up matplotlib figure
            
            return temp_file.name
            
        except Exception as fallback_error:
            st.error(f"Fallback visualization also failed: {str(fallback_error)}")
            return None
    
    def _save_network_safely(self, net) -> str:
        """Safely save PyVis network with immediate fallback on bool iteration error"""
        import uuid
        
        # Check for the specific bool iteration error early
        try:
            # Quick test to see if PyVis object is corrupted
            test_nodes = getattr(net, 'nodes', [])
            test_edges = getattr(net, 'edges', [])
            
            # If we can't access basic properties, skip PyVis methods entirely
            if not hasattr(net, 'save_graph') or 'bool' in str(type(test_nodes)):
                st.warning("PyVis object appears corrupted, using direct fallback")
                return self._create_direct_html_fallback()
                
        except Exception as test_error:
            if 'bool' in str(test_error) and 'iterable' in str(test_error):
                st.warning("Detected PyVis bool iteration error, using direct fallback")
                return self._create_direct_html_fallback()
        
        # Method 1: Standard save_graph (only if no bool error detected)
        try:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.html')
            temp_filename = temp_file.name
            temp_file.close()
            
            net.save_graph(temp_filename)
            
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 100:
                return temp_filename
            else:
                if os.path.exists(temp_filename):
                    os.unlink(temp_filename)
                    
        except Exception as e1:
            if 'bool' in str(e1) and 'iterable' in str(e1):
                st.warning("PyVis bool iteration error detected, using direct fallback")
                return self._create_direct_html_fallback()
            st.warning(f"Standard save method failed: {str(e1)}")
        
        # Method 2: Try to extract data manually and create HTML
        return self._create_direct_html_fallback()
    
    def _create_direct_html_fallback(self) -> str:
        """Create HTML visualization directly from NetworkX graph, bypassing PyVis entirely"""
        try:
            import uuid
            temp_path = os.path.join(tempfile.gettempdir(), f"chainsense_direct_{uuid.uuid4().hex[:8]}.html")
            
            # Extract data directly from NetworkX graph
            nodes_data = []
            edges_data = []
            
            # Prepare nodes with colors and sizes
            for i, node in enumerate(self.graph.nodes()):
                node_type = self.node_types.get(node, 'unknown')
                color = self.node_colors.get(node_type, '#CCCCCC')
                
                # Calculate size based on degree
                size = max(10, min(50, self.graph.degree(node) * 3))
                
                node_data = {
                    'id': str(node),
                    'label': str(node),
                    'color': color,
                    'size': size,
                    'title': f"Type: {node_type}\\nDegree: {self.graph.degree(node)}"
                }
                nodes_data.append(node_data)
            
            # Prepare edges
            for edge in self.graph.edges():
                edge_data = {
                    'from': str(edge[0]),
                    'to': str(edge[1]),
                    'arrows': 'to'
                }
                edges_data.append(edge_data)
            
            # Create interactive HTML with vis.js
            html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>ChainSense Supply Chain Network</title>
    <script src="https://unpkg.com/vis-network@9.1.0/standalone/umd/vis-network.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            border-radius: 10px;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            overflow: hidden;
        }}
        .header {{
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            text-align: center;
        }}
        .header h1 {{
            margin: 0;
            font-size: 1.8rem;
        }}
        .stats {{
            display: flex;
            justify-content: center;
            gap: 30px;
            margin-top: 10px;
            font-size: 0.9rem;
        }}
        .network-container {{
            width: 100%;
            height: 600px;
            border-bottom: 1px solid #e1e5e9;
        }}
        .controls {{
            padding: 15px 20px;
            background: #f8f9fa;
            display: flex;
            justify-content: center;
            gap: 15px;
            flex-wrap: wrap;
        }}
        .control-btn {{
            background: #667eea;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 5px;
            cursor: pointer;
            font-size: 0.9rem;
            transition: background 0.3s;
        }}
        .control-btn:hover {{
            background: #5a6fd8;
        }}
        .legend {{
            padding: 15px 20px;
            display: flex;
            justify-content: center;
            gap: 20px;
            flex-wrap: wrap;
            font-size: 0.85rem;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 8px;
        }}
        .legend-color {{
            width: 16px;
            height: 16px;
            border-radius: 50%;
            border: 2px solid #fff;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>🔗 ChainSense Supply Chain Network</h1>
            <div class="stats">
                <span><strong>{len(nodes_data)}</strong> Nodes</span>
                <span><strong>{len(edges_data)}</strong> Connections</span>
                <span><strong>{len(set(self.node_types.values()))}</strong> Node Types</span>
            </div>
        </div>
        
        <div id="network" class="network-container"></div>
        
        <div class="controls">
            <button class="control-btn" onclick="network.fit()">🔍 Fit View</button>
            <button class="control-btn" onclick="togglePhysics()">⚡ Toggle Physics</button>
            <button class="control-btn" onclick="randomSeed()">🎲 Randomize Layout</button>
        </div>
        
        <div class="legend">
            <div class="legend-item">
                <div class="legend-color" style="background-color: #FF6B6B;"></div>
                <span>Suppliers</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #4ECDC4;"></div>
                <span>Customers</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #45B7D1;"></div>
                <span>Intermediaries</span>
            </div>
            <div class="legend-item">
                <div class="legend-color" style="background-color: #96CEB4;"></div>
                <span>Isolated</span>
            </div>
        </div>
    </div>

    <script>
        // Network data
        var nodes = new vis.DataSet({nodes_data});
        var edges = new vis.DataSet({edges_data});
        
        // Network options
        var options = {{
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -26,
                    centralGravity: 0.005,
                    springLength: 230,
                    springConstant: 0.18
                }},
                maxVelocity: 146,
                timestep: 0.35,
                stabilization: {{iterations: 150}}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 200,
                hideEdgesOnDrag: false
            }},
            nodes: {{
                borderWidth: 2,
                borderWidthSelected: 4,
                font: {{
                    size: 12,
                    color: '#2c3e50'
                }}
            }},
            edges: {{
                width: 2,
                color: {{
                    color: '#848484',
                    highlight: '#667eea',
                    hover: '#667eea'
                }},
                smooth: {{
                    type: 'continuous'
                }}
            }}
        }};
        
        // Create network
        var container = document.getElementById('network');
        var data = {{nodes: nodes, edges: edges}};
        var network = new vis.Network(container, data, options);
        
        // Control functions
        var physicsEnabled = true;
        function togglePhysics() {{
            physicsEnabled = !physicsEnabled;
            network.setOptions({{physics: {{enabled: physicsEnabled}}}});
        }}
        
        function randomSeed() {{
            var newSeed = Math.floor(Math.random() * 1000000);
            network.setOptions({{layout: {{randomSeed: newSeed}}}});
        }}
        
        // Network events
        network.on('click', function(params) {{
            if (params.nodes.length > 0) {{
                var nodeId = params.nodes[0];
                var node = nodes.get(nodeId);
                console.log('Clicked node:', node);
            }}
        }});
        
        network.on('hoverNode', function(params) {{
            document.body.style.cursor = 'pointer';
        }});
        
        network.on('blurNode', function(params) {{
            document.body.style.cursor = 'default';
        }});
        
        // Initial fit
        network.once('stabilizationIterationsDone', function() {{
            network.fit();
        }});
    </script>
</body>
</html>
            """
            
            with open(temp_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            if os.path.exists(temp_path) and os.path.getsize(temp_path) > 1000:
                return temp_path
            else:
                st.error("Direct HTML fallback failed to create valid file")
                return None
                
        except Exception as e:
            st.error(f"Direct HTML fallback failed: {str(e)}")
            return None
    
    def _generate_minimal_html(self, net) -> str:
        """Generate minimal HTML when PyVis methods fail"""
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>ChainSense Network</title>
            <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
        </head>
        <body>
            <div id="mynetworkid" style="width: 100%; height: 600px;"></div>
            <script>
                var nodes = new vis.DataSet({list(net.nodes) if hasattr(net, 'nodes') else []});
                var edges = new vis.DataSet({list(net.edges) if hasattr(net, 'edges') else []});
                var container = document.getElementById('mynetworkid');
                var data = {{nodes: nodes, edges: edges}};
                var options = {{}};
                var network = new vis.Network(container, data, options);
            </script>
        </body>
        </html>
        """
    
    def _create_manual_html(self, net) -> str:
        """Create manual HTML representation of the network"""
        try:
            # Extract nodes and edges from PyVis network
            nodes_data = []
            edges_data = []
            
            if hasattr(net, 'nodes'):
                for node in net.nodes:
                    if isinstance(node, dict):
                        nodes_data.append(node)
                    else:
                        nodes_data.append({'id': str(node), 'label': str(node)})
            
            if hasattr(net, 'edges'):
                for edge in net.edges:
                    if isinstance(edge, dict):
                        edges_data.append(edge)
                    else:
                        edges_data.append({'from': str(edge[0]), 'to': str(edge[1])})
            
            # Create simple HTML visualization
            nodes_json = str(nodes_data).replace("'", '"')
            edges_json = str(edges_data).replace("'", '"')
            
            return f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>ChainSense Network Visualization</title>
                <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 20px; }}
                    #network {{ width: 100%; height: 600px; border: 1px solid #ccc; }}
                    .info {{ background: #f0f0f0; padding: 10px; margin: 10px 0; border-radius: 5px; }}
                </style>
            </head>
            <body>
                <h2>ChainSense Supply Chain Network</h2>
                <div class="info">
                    <strong>Nodes:</strong> {len(nodes_data)} | <strong>Edges:</strong> {len(edges_data)}
                </div>
                <div id="network"></div>
                <script>
                    var nodes = new vis.DataSet({nodes_json});
                    var edges = new vis.DataSet({edges_json});
                    var container = document.getElementById('network');
                    var data = {{ nodes: nodes, edges: edges }};
                    var options = {{
                        physics: {{ enabled: true }},
                        interaction: {{ hover: true }}
                    }};
                    var network = new vis.Network(container, data, options);
                </script>
            </body>
            </html>
            """
            
        except Exception as e:
            # Ultimate fallback - simple message
            return f"""
            <!DOCTYPE html>
            <html>
            <head><title>ChainSense Network</title></head>
            <body>
                <h2>ChainSense Network Visualization</h2>
                <p>Network visualization temporarily unavailable.</p>
                <p>Error: {str(e)}</p>
                <p>Please try refreshing or use a different layout option.</p>
            </body>
            </html>
            """
    
    def _calculate_node_sizes(self, metric: str = 'degree_centrality') -> Dict[str, float]:
        """Calculate node sizes based on a metric"""
        node_sizes = {}
        
        if metric == 'degree':
            sizes = {node: self.graph.degree(node) for node in self.graph.nodes()}
        elif metric in self.metrics and isinstance(self.metrics[metric], dict):
            sizes = self.metrics[metric]
        elif metric == 'risk_score' and self.risk_scores:
            sizes = self.risk_scores
        else:
            # Default to equal sizes
            sizes = {node: 1 for node in self.graph.nodes()}
        
        # Normalize sizes to reasonable range (10-30)
        if sizes:
            min_size, max_size = min(sizes.values()), max(sizes.values())
            size_range = max_size - min_size
            
            if size_range > 0:
                for node in sizes:
                    normalized = (sizes[node] - min_size) / size_range
                    node_sizes[node] = 10 + normalized * 20
            else:
                node_sizes = {node: 15 for node in sizes}
        
        return node_sizes
    
    def _risk_to_color(self, risk_score: float) -> str:
        """Convert risk score to color"""
        if risk_score < 0.3:
            return '#00CC96'  # Green
        elif risk_score < 0.7:
            return '#FFA15A'  # Orange
        else:
            return '#EF553B'  # Red
    
    def _create_node_tooltip(self, node: str, node_type: str) -> str:
        """Create detailed tooltip for a node"""
        tooltip = f"<b>{node}</b><br>"
        tooltip += f"Type: {node_type}<br>"
        tooltip += f"Degree: {self.graph.degree(node)}<br>"
        tooltip += f"In-degree: {self.graph.in_degree(node)}<br>"
        tooltip += f"Out-degree: {self.graph.out_degree(node)}<br>"
        
        if self.metrics:
            if 'degree_centrality' in self.metrics:
                tooltip += f"Degree Centrality: {self.metrics['degree_centrality'].get(node, 0):.3f}<br>"
            if 'betweenness_centrality' in self.metrics:
                tooltip += f"Betweenness Centrality: {self.metrics['betweenness_centrality'].get(node, 0):.3f}<br>"
        
        if node in self.risk_scores:
            tooltip += f"Risk Score: {self.risk_scores[node]:.3f}<br>"
        
        return tooltip
    
    def _create_edge_tooltip(self, source: str, target: str, data: Dict) -> str:
        """Create tooltip for an edge"""
        tooltip = f"<b>{source} → {target}</b><br>"
        
        for key, value in data.items():
            if key not in ['weight']:
                tooltip += f"{key.title()}: {value}<br>"
        
        return tooltip
    
    def create_metrics_dashboard(self) -> List[go.Figure]:
        """Create dashboard charts for metrics visualization"""
        figures = []
        
        try:
            # 1. Node type distribution
            if self.node_types:
                type_counts = {}
                for node_type in self.node_types.values():
                    type_counts[node_type] = type_counts.get(node_type, 0) + 1
                
                fig_pie = px.pie(
                    values=list(type_counts.values()),
                    names=list(type_counts.keys()),
                    title="Node Type Distribution",
                    color_discrete_map=self.node_colors
                )
                figures.append(fig_pie)
            
            # 2. Degree distribution
            degrees = [self.graph.degree(node) for node in self.graph.nodes()]
            fig_hist = px.histogram(
                x=degrees,
                nbins=20,
                title="Degree Distribution",
                labels={'x': 'Node Degree', 'y': 'Count'}
            )
            figures.append(fig_hist)
            
            # 3. Centrality comparison (top 10 nodes)
            if 'degree_centrality' in self.metrics and 'betweenness_centrality' in self.metrics:
                top_nodes = sorted(self.metrics['degree_centrality'].items(), 
                                 key=lambda x: x[1], reverse=True)[:10]
                
                node_names = [node for node, _ in top_nodes]
                degree_centrality = [self.metrics['degree_centrality'][node] for node in node_names]
                betweenness_centrality = [self.metrics['betweenness_centrality'].get(node, 0) for node in node_names]
                
                fig_centrality = go.Figure()
                fig_centrality.add_trace(go.Bar(
                    x=node_names,
                    y=degree_centrality,
                    name='Degree Centrality',
                    marker_color='lightblue'
                ))
                fig_centrality.add_trace(go.Bar(
                    x=node_names,
                    y=betweenness_centrality,
                    name='Betweenness Centrality',
                    marker_color='salmon'
                ))
                fig_centrality.update_layout(
                    title="Top 10 Nodes - Centrality Comparison",
                    xaxis_title="Nodes",
                    yaxis_title="Centrality Value",
                    barmode='group'
                )
                figures.append(fig_centrality)
            
            # 4. Risk score distribution
            if self.risk_scores:
                risk_values = list(self.risk_scores.values())
                fig_risk = px.histogram(
                    x=risk_values,
                    nbins=20,
                    title="Risk Score Distribution",
                    labels={'x': 'Risk Score', 'y': 'Count'},
                    color_discrete_sequence=['#EF553B']
                )
                figures.append(fig_risk)
            
            return figures
            
        except Exception as e:
            st.error(f"Error creating metrics dashboard: {str(e)}")
            return []
    
    def create_risk_heatmap(self, communities: Dict = None) -> go.Figure:
        """Create risk heatmap visualization"""
        try:
            if not self.risk_scores:
                return None
            
            # Prepare data for heatmap
            nodes = list(self.risk_scores.keys())
            risk_values = list(self.risk_scores.values())
            node_types = [self.node_types.get(node, 'unknown') for node in nodes]
            
            # Create DataFrame for easier handling
            df = pd.DataFrame({
                'node': nodes,
                'risk_score': risk_values,
                'node_type': node_types
            })
            
            # Sort by risk score
            df = df.sort_values('risk_score', ascending=False)
            
            # Create heatmap
            fig = px.bar(
                df.head(20),  # Top 20 risky nodes
                x='risk_score',
                y='node',
                color='risk_score',
                color_continuous_scale=self.risk_colorscale,
                title="Top 20 Risky Nodes",
                labels={'risk_score': 'Risk Score', 'node': 'Node'},
                orientation='h'
            )
            
            fig.update_layout(
                height=600,
                yaxis={'categoryorder': 'total ascending'}
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating risk heatmap: {str(e)}")
            return None
    
    def create_network_overview(self) -> go.Figure:
        """Create network overview visualization"""
        try:
            # Calculate positions using spring layout
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Prepare node traces
            node_traces = {}
            for node_type in self.node_colors:
                node_traces[node_type] = {
                    'x': [], 'y': [], 'text': [], 'size': []
                }
            
            for node in self.graph.nodes():
                node_type = self.node_types.get(node, 'isolated')
                x, y = pos[node]
                
                node_traces[node_type]['x'].append(x)
                node_traces[node_type]['y'].append(y)
                node_traces[node_type]['text'].append(node)
                
                # Size based on degree
                size = max(5, min(20, self.graph.degree(node) * 2))
                node_traces[node_type]['size'].append(size)
            
            # Create figure
            fig = go.Figure()
            
            # Add edge traces
            edge_x = []
            edge_y = []
            for edge in self.graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Edges'
            ))
            
            # Add node traces
            for node_type, trace_data in node_traces.items():
                if trace_data['x']:  # Only add if there are nodes of this type
                    fig.add_trace(go.Scatter(
                        x=trace_data['x'],
                        y=trace_data['y'],
                        mode='markers',
                        hoverinfo='text',
                        text=trace_data['text'],
                        marker=dict(
                            size=trace_data['size'],
                            color=self.node_colors[node_type],
                            line=dict(width=2, color='white')
                        ),
                        name=node_type.title()
                    ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': "Supply Chain Network Overview",
                    'font': {'size': 16}
                },
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                annotations=[
                    dict(
                        text="Interactive network visualization",
                        showarrow=False,
                        xref="paper", yref="paper",
                        x=0.005, y=-0.002,
                        xanchor='left', yanchor='bottom',
                        font=dict(color="#888", size=12)
                    )
                ],
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating network overview: {str(e)}")
            return None
    
    def create_community_visualization(self, communities: Dict) -> go.Figure:
        """Create community detection visualization"""
        try:
            if not communities or 'assignments' not in communities:
                return None
            
            # Calculate positions
            pos = nx.spring_layout(self.graph, k=1, iterations=50)
            
            # Color map for communities
            community_colors = px.colors.qualitative.Set3
            
            fig = go.Figure()
            
            # Add edge traces
            edge_x = []
            edge_y = []
            for edge in self.graph.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            fig.add_trace(go.Scatter(
                x=edge_x, y=edge_y,
                line=dict(width=0.5, color='#888'),
                hoverinfo='none',
                mode='lines',
                name='Edges'
            ))
            
            # Add nodes by community
            community_assignments = communities['assignments']
            unique_communities = set(community_assignments.values())
            
            for i, community_id in enumerate(unique_communities):
                nodes_in_community = [node for node, comm in community_assignments.items() 
                                    if comm == community_id]
                
                if nodes_in_community:
                    x_coords = [pos[node][0] for node in nodes_in_community]
                    y_coords = [pos[node][1] for node in nodes_in_community]
                    
                    color = community_colors[i % len(community_colors)]
                    
                    fig.add_trace(go.Scatter(
                        x=x_coords,
                        y=y_coords,
                        mode='markers',
                        hoverinfo='text',
                        text=nodes_in_community,
                        marker=dict(
                            size=10,
                            color=color,
                            line=dict(width=2, color='white')
                        ),
                        name=f'Community {community_id}'
                    ))
            
            # Update layout
            fig.update_layout(
                title={
                    'text': "Community Detection Visualization",
                    'font': {'size': 16}
                },
                showlegend=True,
                hovermode='closest',
                margin=dict(b=20,l=5,r=5,t=40),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor='white'
            )
            
            return fig
            
        except Exception as e:
            st.error(f"Error creating community visualization: {str(e)}")
            return None
    
    def create_advanced_risk_dashboard(self, risk_factors: Dict) -> List[go.Figure]:
        """Create advanced risk analysis dashboard"""
        figures = []
        
        try:
            if not risk_factors:
                return figures
            
            # 1. Risk dimensions comparison
            if 'structural' in risk_factors and 'operational' in risk_factors:
                nodes = list(risk_factors['structural'].keys())[:20]  # Top 20 nodes
                
                structural_scores = [risk_factors['structural'].get(node, 0) for node in nodes]
                operational_scores = [risk_factors['operational'].get(node, 0) for node in nodes]
                concentration_scores = [risk_factors.get('concentration', {}).get(node, 0) for node in nodes]
                connectivity_scores = [risk_factors.get('connectivity', {}).get(node, 0) for node in nodes]
                
                fig_radar = go.Figure()
                
                # Add traces for different risk dimensions
                fig_radar.add_trace(go.Scatterpolar(
                    r=structural_scores[:10],
                    theta=nodes[:10],
                    fill='toself',
                    name='Structural Risk',
                    line_color='red'
                ))
                
                fig_radar.add_trace(go.Scatterpolar(
                    r=operational_scores[:10],
                    theta=nodes[:10],
                    fill='toself',
                    name='Operational Risk',
                    line_color='blue'
                ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(
                            visible=True,
                            range=[0, 1]
                        )
                    ),
                    title="Risk Dimensions Comparison (Top 10 Nodes)",
                    showlegend=True
                )
                
                figures.append(fig_radar)
            
            # 2. Risk correlation matrix
            if len(risk_factors) > 1:
                risk_data = []
                node_list = list(next(iter(risk_factors.values())).keys())
                
                for risk_type in risk_factors:
                    risk_data.append([risk_factors[risk_type].get(node, 0) for node in node_list])
                
                correlation_matrix = np.corrcoef(risk_data)
                
                fig_corr = go.Figure(data=go.Heatmap(
                    z=correlation_matrix,
                    x=list(risk_factors.keys()),
                    y=list(risk_factors.keys()),
                    colorscale='RdBu',
                    zmid=0
                ))
                
                fig_corr.update_layout(
                    title="Risk Factors Correlation Matrix",
                    xaxis_title="Risk Factors",
                    yaxis_title="Risk Factors"
                )
                
                figures.append(fig_corr)
            
            # 3. Composite risk distribution by node type
            if 'composite' in risk_factors:
                composite_risks = risk_factors['composite']
                
                # Group by node type
                risk_by_type = {}
                for node, risk in composite_risks.items():
                    node_type = self.node_types.get(node, 'unknown')
                    if node_type not in risk_by_type:
                        risk_by_type[node_type] = []
                    risk_by_type[node_type].append(risk)
                
                fig_box = go.Figure()
                
                for node_type, risks in risk_by_type.items():
                    fig_box.add_trace(go.Box(
                        y=risks,
                        name=node_type.title(),
                        boxpoints='outliers',
                        marker_color=self.node_colors.get(node_type, '#CCCCCC')
                    ))
                
                fig_box.update_layout(
                    title="Risk Distribution by Node Type",
                    yaxis_title="Composite Risk Score",
                    xaxis_title="Node Type"
                )
                
                figures.append(fig_box)
            
            return figures
            
        except Exception as e:
            st.error(f"Error creating advanced risk dashboard: {str(e)}")
            return []