# Changelog

All notable changes to ChainSense will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2024-01-15

### Added
- Initial release of ChainSense Supply Chain Risk Analyzer
- Level 1 (MVP) Exploratory Dashboard
  - CSV data upload and validation
  - Automatic graph construction from supply chain data
  - Interactive network visualization with PyVis
  - Basic graph metrics calculation (centrality, components)
  - Node classification (suppliers, customers, intermediaries)
- Level 2 Advanced Risk Analysis
  - Multi-dimensional risk scoring system
  - Community detection using Louvain algorithm
  - Anomaly detection with Isolation Forest and LOF
  - Scenario planning and disruption simulation
  - AI-powered resilience recommendations
- Professional Streamlit UI with modern styling
- Comprehensive error handling and fallback mechanisms
- Sample data generation for testing

### Features
- **Data Processing**: Robust CSV handling with automatic validation
- **Graph Analysis**: NetworkX-based network analysis with 13+ metrics
- **Visualization**: Multiple visualization backends (PyVis, vis.js, matplotlib)
- **Risk Assessment**: Four-dimensional risk scoring framework
- **Machine Learning**: Anomaly detection and community clustering
- **User Interface**: Professional gradient design with responsive layout
- **Error Recovery**: Automatic fallback systems for reliable operation

### Technical Specifications
- **Framework**: Streamlit 1.28+
- **Graph Engine**: NetworkX 3.2+
- **Visualization**: PyVis 0.3.2, vis.js, Plotly 5.17+
- **Machine Learning**: scikit-learn 1.3.2
- **Data Processing**: Pandas 2.1.3
- **Python Version**: 3.8+

### Supported Platforms
- Windows 10/11
- macOS 10.15+
- Linux (Ubuntu 18.04+)

### Installation Methods
- Automated setup scripts (setup.bat, run_app.bat)
- Manual pip installation
- Virtual environment support

## [0.9.0] - 2024-01-10 (Beta)

### Added
- Core graph analysis functionality
- Basic Streamlit interface
- PyVis integration for network visualization

### Fixed
- Initial PyVis compatibility issues
- Data validation edge cases

## [0.8.0] - 2024-01-05 (Alpha)

### Added
- Basic data processing capabilities
- NetworkX graph construction
- Initial risk calculation algorithms

---

## Future Releases

### [1.1.0] - Planned Features
- Export functionality (PDF reports, CSV results)
- Advanced layout algorithms
- Real-time data streaming support
- API integration capabilities
- Enhanced mobile responsiveness

### [1.2.0] - Advanced Analytics
- Time-series analysis for supply chain evolution
- Predictive modeling for disruption forecasting
- Integration with external data sources
- Advanced machine learning models

### [2.0.0] - Enterprise Features
- Multi-user support and authentication
- Database integration (PostgreSQL, MongoDB)
- Advanced security features
- Custom branding and white-labeling
- REST API for integration