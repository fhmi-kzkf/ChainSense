#!/usr/bin/env python3
"""
Test script to verify all modules can be imported successfully
"""

def test_imports():
    """Test all module imports"""
    try:
        from data_processor import SupplyChainData
        print("✓ Data processor imported successfully")
    except ImportError as e:
        print(f"✗ Data processor import failed: {e}")
    
    try:
        from graph_analyzer import GraphAnalyzer
        print("✓ Graph analyzer imported successfully")
    except ImportError as e:
        print(f"✗ Graph analyzer import failed: {e}")
    
    try:
        from visualizer import SupplyChainVisualizer
        print("✓ Visualizer imported successfully")
    except ImportError as e:
        print(f"✗ Visualizer import failed: {e}")
    
    try:
        from advanced_risk_analyzer import AdvancedRiskAnalyzer
        print("✓ Advanced risk analyzer imported successfully")
    except ImportError as e:
        print(f"✗ Advanced risk analyzer import failed: {e}")
    
    # Test basic functionality
    print("\n🧪 Testing basic functionality...")
    
    try:
        # Create sample data
        data_processor = SupplyChainData()
        sample_df = data_processor.create_sample_data()
        print(f"✓ Sample data created: {len(sample_df)} rows")
        
        # Load data
        data_processor.df = sample_df
        is_valid, message = data_processor.validate_data()
        print(f"✓ Data validation: {is_valid} - {message}")
        
        # Build graph
        if data_processor.build_graph():
            print(f"✓ Graph built: {data_processor.graph.number_of_nodes()} nodes, {data_processor.graph.number_of_edges()} edges")
            
            # Test analyzer
            analyzer = GraphAnalyzer(data_processor.graph, data_processor.node_types)
            metrics = analyzer.calculate_basic_metrics()
            print(f"✓ Basic metrics calculated: {len(metrics)} metrics")
            
            # Test visualizer
            visualizer = SupplyChainVisualizer(
                data_processor.graph, 
                data_processor.node_types, 
                metrics
            )
            print("✓ Visualizer initialized successfully")
            
        else:
            print("✗ Failed to build graph")
            
    except Exception as e:
        print(f"✗ Basic functionality test failed: {e}")
    
    print("\n🎉 Module testing completed!")

if __name__ == "__main__":
    test_imports()