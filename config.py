# config.py - Add imputation configuration

CONFIG = {
    # Phase 1: Data Health Check Settings
    'phase1': {
        'min_companies':35,
        'min_kpis': 0, 
        'min_completeness': 0,
        'output_folder': 'data/results/data_check',
        'sort_priority': ['Companies', 'Data_Points_With_Values', 'KPIs']
    },
    
    # Phase 2: Clustering Settings - UPDATED WITH IMPUTATION OPTIONS
    'phase2': {
        'n_clusters': 3,  # Will be overridden by elbow method
        'random_state': 42,
        'min_kpi_completeness': 15,      # Configurable filtering
        'min_company_completeness': 5,  # Configurable filtering
        'imputation_method': 'median',   # Options: 'knn', 'median', 'mean'
        'knn_neighbors': 5,              # Only used if imputation_method = 'knn'
        'standardization_method': 'zscore',
        'max_clusters_to_test': 4
    },
    
    # Phase 3: Visualization Settings
    'phase3': {
        'chart_style': 'seaborn',
        'dpi': 300,
        'color_palette': 'viridis'
    },
    
    # General Settings
    'general': {
        'sector_name': 'Industrials',  #ENV Technology Healthcare Real Estate Consumer Defensive Industrialss # GOV Real Estate
        'esg_pillars': ['Environmental', ], # 'Social', 'Governance'
        'data_file': 'data/esg_data.csv'
    }
}