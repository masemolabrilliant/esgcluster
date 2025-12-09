# phase22.py - Enhanced Clustering Analysis for Research

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import silhouette_score, silhouette_samples
from scipy.stats import f_oneway
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import textwrap
from config import CONFIG

# Consistent color theme across charts
THEME = {
    'perf_pos': '#27AE60',          # positive performance bars
    'perf_neg': '#E74C3C',          # negative performance bars
    'pillar_importance': '#2E86AB', # pillar-wide importance
    'cluster_importance': '#E67E22',# cluster-specific importance
    'anova': {
        '***': '#2E8B57',
        '**': '#7CB518',
        '*': '#C5E384',
        'NS': '#6C757D'
    }
}

def analyze_sector_enhanced():
    """Enhanced clustering analysis with research-justified components"""
    
    sector_name = CONFIG['general']['sector_name']
    print(f"🚀 ENHANCED CLUSTERING ANALYSIS - {sector_name}")
    print("=" * 70)
    print("🔬 Research Components: Elbow Method, Silhouette Analysis, ANOVA")
    print("=" * 70)
    
    # Setup folder structure
    base_dir = setup_sector_folders(sector_name)
    
    # Analyze each ESG pillar
    pillars = CONFIG['general']['esg_pillars']
    all_results = {}
    
    for pillar in pillars:
        print(f"\n{'='*70}")
        print(f"ANALYZING {sector_name.upper()} - {pillar.upper()}")
        print(f"{'='*70}")
        
        # Load and analyze data
        raw_matrix = load_sector_data(sector_name, pillar)
        if raw_matrix is None:
            continue
            
        # Data preparation
        prepared_data = prepare_data_enhanced(raw_matrix, pillar, base_dir)
        if prepared_data is None:
            continue
            
        standardized_data, filtered_data, imputed_data, imputation_stats = prepared_data
        
        # Determine optimal clusters using elbow method
        optimal_clusters = determine_optimal_clusters(standardized_data, pillar, base_dir)
        if optimal_clusters is None:
            continue
        
        # Perform clustering with optimal cluster count
        CONFIG['phase2']['n_clusters'] = optimal_clusters
        clustering_results = perform_clustering_enhanced(standardized_data, pillar)
        if clustering_results is None:
            continue
            
        cluster_labels, kmeans_model, silhouette_avg = clustering_results
        
        # Comprehensive feature importance with ANOVA
        statistical_analysis = compute_comprehensive_feature_importance(
            standardized_data, cluster_labels, filtered_data.columns, pillar
        )
        
        # Add cluster labels
        clustered_data = standardized_data.copy()
        clustered_data['Cluster'] = cluster_labels
        
        # Enhanced cluster analysis
        cluster_analysis = analyze_clusters_comprehensive(
            clustered_data, kmeans_model, statistical_analysis, pillar
        )
        
        # Generate research reports and visualizations
        generate_research_reports(
            clustered_data, cluster_analysis, statistical_analysis, 
            pillar, sector_name, base_dir
        )
        
        all_results[pillar] = {
            'clustered_data': clustered_data,
            'cluster_analysis': cluster_analysis,
            'statistical_analysis': statistical_analysis
        }
    
    print(f"\n🎉 RESEARCH ANALYSIS COMPLETED FOR {sector_name.upper()}")
    print(f"📁 All results saved in: {base_dir}/")
    
    return all_results

def setup_sector_folders(sector_name):
    """Create organized folder structure for the sector"""
    sector_dir = Path(f"data/results/{sector_name.replace(' ', '_')}")
    pillars = CONFIG['general']['esg_pillars']
    
    for pillar in pillars:
        (sector_dir / pillar / 'visualizations').mkdir(parents=True, exist_ok=True)
        (sector_dir / pillar / 'cluster_data').mkdir(parents=True, exist_ok=True)
    
    (sector_dir / 'Weight_Analysis').mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Research folder structure created for {sector_name}")
    return sector_dir

def load_sector_data(sector_name, pillar):
    """Load sector data for a specific pillar"""
    matrix_path = f"data/results/data_check/matrices/matrix_{sector_name}_{pillar}.csv"
    
    try:
        df = pd.read_csv(matrix_path, index_col=0)
        print(f"✅ Loaded: {df.shape[0]} companies × {df.shape[1]} KPIs")
        return df
    except FileNotFoundError:
        print(f"❌ No data found for {sector_name} - {pillar}")
        return None

def prepare_data_enhanced(raw_matrix, pillar, base_dir):
    """Enhanced data preparation with research standards"""
    print(f"\n🔄 DATA PREPARATION")
    print("-" * 50)
    
    # Filter data
    filtered_data = filter_data(raw_matrix)
    if filtered_data is None:
        return None
    
    # Save filtered matrix
    filtered_file = base_dir / pillar / 'cluster_data' / 'filtered_matrix.csv'
    filtered_data.to_csv(filtered_file)
    
    # Impute missing values using KNN
    print("  Imputing missing values (KNN)...")
    imputer = KNNImputer(n_neighbors=5)
    imputed_data = imputer.fit_transform(filtered_data)
    imputed_df = pd.DataFrame(imputed_data, index=filtered_data.index, columns=filtered_data.columns)
    
    # Save imputed matrix
    imputed_file = base_dir / pillar / 'cluster_data' / 'imputed_matrix.csv'
    imputed_df.to_csv(imputed_file)
    
    # Standardize using Z-score
    print("  Standardizing data (Z-score)...")
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(imputed_df)
    standardized_df = pd.DataFrame(standardized_data, index=imputed_df.index, columns=imputed_df.columns)
    
    # Calculate imputation statistics
    total_cells = filtered_data.size
    original_data = filtered_data.notnull().sum().sum()
    imputed_data_count = total_cells - original_data
    imputation_stats = {
        'total_cells': total_cells,
        'original_data': original_data,
        'imputed_data': imputed_data_count,
        'imputed_pct': (imputed_data_count / total_cells) * 100
    }
    
    print(f"✅ Final data: {standardized_df.shape[0]} companies × {standardized_df.shape[1]} KPIs")
    print(f"📊 Imputation: {imputation_stats['imputed_pct']:.1f}% of data imputed")
    
    return standardized_df, filtered_data, imputed_df, imputation_stats

def filter_data(raw_matrix):
    """Apply filtering based on configuration thresholds"""
    filtered_data = raw_matrix.copy()
    
    # Filter KPIs with low completeness
    if CONFIG['phase2']['min_kpi_completeness'] > 0:
        kpi_completeness = (raw_matrix.notnull().sum(axis=0) / len(raw_matrix)) * 100
        viable_kpis = kpi_completeness[kpi_completeness >= CONFIG['phase2']['min_kpi_completeness']].index
        filtered_data = filtered_data[viable_kpis]
        print(f"  KPIs after filtering: {len(viable_kpis)}/{raw_matrix.shape[1]}")
    
    # Filter companies with low data
    if CONFIG['phase2']['min_company_completeness'] > 0:
        company_completeness = (filtered_data.notnull().sum(axis=1) / len(filtered_data.columns)) * 100
        viable_companies = company_completeness[company_completeness >= CONFIG['phase2']['min_company_completeness']].index
        filtered_data = filtered_data.loc[viable_companies]
        print(f"  Companies after filtering: {len(viable_companies)}/{raw_matrix.shape[0]}")
    
    if len(filtered_data) < 10:
        print("❌ Too few companies after filtering")
        return None
    
    completeness_after = (filtered_data.notnull().sum().sum() / filtered_data.size) * 100
    print(f"  Completeness after filtering: {completeness_after:.1f}%")
    
    return filtered_data

def determine_optimal_clusters(data, pillar, base_dir):
    """Determine optimal number of clusters using elbow method and silhouette analysis"""
    print(f"\n🎯 DETERMINING OPTIMAL CLUSTER COUNT")
    print("-" * 50)
    
    # Test different numbers of clusters
    max_clusters_setting = CONFIG['phase2'].get('max_clusters_to_test', len(data) - 1)
    max_clusters = max(2, min(max_clusters_setting, len(data) - 1))
    cluster_range = range(2, max_clusters + 1)
    
    wcss = []  # Within-cluster sum of squares
    silhouette_scores = []
    
    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG['phase2']['random_state'], n_init=15)
        cluster_labels = kmeans.fit_predict(data)
        
        # Calculate WCSS (elbow method)
        wcss.append(kmeans.inertia_)
        
        # Calculate silhouette score
        if n_clusters > 1:
            silhouette_avg = silhouette_score(data, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        else:
            silhouette_scores.append(0)
    
    # Find optimal clusters (elbow point)
    optimal_clusters = find_elbow_point(wcss, cluster_range)
    
    # Create elbow chart
    create_elbow_chart(cluster_range, wcss, silhouette_scores, optimal_clusters, pillar, base_dir)
    
    print(f"  Optimal clusters determined: {optimal_clusters}")
    print(f"  Silhouette score at optimal: {silhouette_scores[optimal_clusters-2]:.3f}")
    
    return optimal_clusters

def find_elbow_point(wcss, cluster_range):
    """Find the elbow point in WCSS curve"""
    # Simple elbow detection: find point with maximum curvature
    differences = []
    for i in range(1, len(wcss)):
        diff = wcss[i-1] - wcss[i]
        differences.append(diff)
    
    # Find the point where the decrease slows down significantly
    if len(differences) > 1:
        second_diff = [differences[i-1] - differences[i] for i in range(1, len(differences))]
        optimal_index = np.argmax(second_diff) + 2  # +2 because we start from 2 clusters
    else:
        optimal_index = 2
    
    return min(optimal_index, len(cluster_range))

def create_elbow_chart(cluster_range, wcss, silhouette_scores, optimal_clusters, pillar, base_dir):
    """Create elbow method visualization with silhouette scores"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    cluster_values = list(cluster_range)
    optimal_idx = cluster_values.index(optimal_clusters)
    
    # Elbow plot (WCSS)
    ax1.plot(cluster_range, wcss, 'bo-', linewidth=2, markersize=8)
    ax1.axvline(x=optimal_clusters, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('Within-Cluster Sum of Squares (WCSS)')
    ax1.set_title(f'{pillar} Pillar: Elbow Method\nOptimal Clusters: {optimal_clusters}')
    ax1.grid(True, alpha=0.3)
    
    # Silhouette score plot
    ax2.plot(cluster_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    ax2.axvline(x=optimal_clusters, color='red', linestyle='--', alpha=0.7)
    optimal_silhouette = silhouette_scores[optimal_idx]
    # Annotate the optimal silhouette value directly on the chart for quick reference
    ax2.scatter([optimal_clusters], [optimal_silhouette], color='red', zorder=5)
    ax2.annotate(
        f"Optimal k={optimal_clusters}\nSilhouette={optimal_silhouette:.3f}",
        xy=(optimal_clusters, optimal_silhouette),
        xytext=(optimal_clusters + 0.3, optimal_silhouette),
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.9),
        arrowprops=dict(arrowstyle="->", connectionstyle="arc3", color="gray", alpha=0.8)
    )
    ax2.set_xlabel('Number of Clusters')
    ax2.set_ylabel('Silhouette Score')
    ax2.set_title(f'{pillar} Pillar: Silhouette Analysis\nOptimal Clusters: {optimal_clusters}')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    elbow_path = base_dir / pillar / 'visualizations' / 'elbow_analysis.png'
    plt.savefig(elbow_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Elbow chart saved: {elbow_path}")

def perform_clustering_enhanced(data, pillar):
    """Enhanced clustering with validation"""
    print(f"\n🎯 CLUSTERING ANALYSIS")
    print("-" * 50)
    
    n_clusters = CONFIG['phase2']['n_clusters']
    kmeans = KMeans(n_clusters=n_clusters, random_state=CONFIG['phase2']['random_state'], n_init=15)
    cluster_labels = kmeans.fit_predict(data)
    
    silhouette_avg = silhouette_score(data, cluster_labels)
    
    # Calculate silhouette samples for detailed analysis
    silhouette_vals = silhouette_samples(data, cluster_labels)
    
    # Validate cluster sizes
    cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()
    
    print(f"  Clusters: {n_clusters}")
    print(f"  Silhouette Score: {silhouette_avg:.3f}")
    print(f"  Cluster sizes: {dict(cluster_sizes)}")
    
    # Check for balanced clusters
    balanced = all(size >= 3 for size in cluster_sizes)
    if not balanced:
        print("  ⚠️  Warning: Unbalanced clusters detected")
    
    return cluster_labels, kmeans, silhouette_avg

def compute_comprehensive_feature_importance(data, cluster_labels, feature_names, pillar):
    """Compute both cluster-specific and pillar-wide feature importance with ANOVA"""
    
    print(f"\n🔍 COMPUTING COMPREHENSIVE FEATURE IMPORTANCE")
    print("-" * 50)
    
    statistical_results = {}
    data_array = data.values if hasattr(data, 'values') else data
    
    # 1. PILLAR-WIDE FEATURE IMPORTANCE (across all clusters)
    print("  Computing pillar-wide importance...")
    rf_pillar = RandomForestClassifier(n_estimators=100, random_state=CONFIG['phase2']['random_state'])
    rf_pillar.fit(data_array, cluster_labels)
    
    pillar_importance = pd.DataFrame({
        'KPI': feature_names,
        'Pillar_Importance': rf_pillar.feature_importances_
    }).sort_values('Pillar_Importance', ascending=False)
    
    statistical_results['pillar_importance'] = pillar_importance
    
    # 2. CLUSTER-SPECIFIC FEATURE IMPORTANCE (for each cluster)
    print("  Computing cluster-specific importance...")
    cluster_importance_results = {}
    
    for cluster_id in range(CONFIG['phase2']['n_clusters']):
        # Create binary classification: this cluster vs all others
        binary_labels = (cluster_labels == cluster_id).astype(int)
        size_in = int(binary_labels.sum())
        size_out = len(binary_labels) - size_in

        # Only compute if both classes have enough samples (>=3 each) so RF is meaningful
        if size_in >= 3 and size_out >= 3:
            rf_cluster = RandomForestClassifier(n_estimators=100, random_state=CONFIG['phase2']['random_state'])
            rf_cluster.fit(data_array, binary_labels)
            
            cluster_importance = pd.DataFrame({
                'KPI': feature_names,
                'Cluster_Specific_Importance': rf_cluster.feature_importances_
            }).sort_values('Cluster_Specific_Importance', ascending=False)
            
            cluster_importance_results[cluster_id] = cluster_importance
        else:
            cluster_importance_results[cluster_id] = None
    
    statistical_results['cluster_importance'] = cluster_importance_results
    
    # 3. STATISTICAL SIGNIFICANCE (ANOVA)
    print("  Computing ANOVA statistical significance...")
    anova_results = []
    
    for i, feature in enumerate(feature_names):
        feature_data = []
        for cluster_id in range(CONFIG['phase2']['n_clusters']):
            cluster_mask = cluster_labels == cluster_id
            feature_data.append(data_array[cluster_mask, i])
        
        try:
            f_stat, p_value = f_oneway(*feature_data)
            anova_results.append({
                'KPI': feature,
                'F_Statistic': f_stat,
                'P_Value': p_value
            })
        except:
            anova_results.append({
                'KPI': feature,
                'F_Statistic': 0,
                'P_Value': 1
            })
    
    anova_df = pd.DataFrame(anova_results)
    anova_df['Significance'] = anova_df['P_Value'].apply(
        lambda p: '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'NS'
    )
    statistical_results['anova_results'] = anova_df.sort_values('P_Value')
    
    print(f"  ✅ Feature importance computed:")
    print(f"     - Pillar-wide: {len(pillar_importance)} KPIs")
    print(f"     - Cluster-specific: {len([x for x in cluster_importance_results.values() if x is not None])} clusters")
    print(f"     - ANOVA: {len(anova_df)} KPIs tested")
    
    return statistical_results

def analyze_clusters_comprehensive(clustered_data, kmeans_model, statistical_analysis, pillar):
    """Comprehensive cluster analysis with both weight types"""
    
    cluster_centers = kmeans_model.cluster_centers_
    feature_names = clustered_data.drop('Cluster', axis=1).columns
    
    cluster_profiles = []
    
    for cluster_id in range(len(cluster_centers)):
        cluster_companies = clustered_data[clustered_data['Cluster'] == cluster_id]
        cluster_center = cluster_centers[cluster_id]
        
        # Performance profile (standardized averages)
        performance_profile = pd.Series(cluster_center, index=feature_names)
        
        # Get pillar-wide importance
        pillar_importance = None
        if 'pillar_importance' in statistical_analysis:
            pillar_importance = statistical_analysis['pillar_importance'].set_index('KPI')['Pillar_Importance']
        
        # Get cluster-specific importance
        cluster_specific_importance = None
        if ('cluster_importance' in statistical_analysis and 
            statistical_analysis['cluster_importance'].get(cluster_id) is not None):
            cluster_specific_importance = statistical_analysis['cluster_importance'][cluster_id].set_index('KPI')['Cluster_Specific_Importance']
        
        profile = {
            'cluster_id': cluster_id,
            'size': len(cluster_companies),
            'companies': cluster_companies.index.tolist(),
            'performance_profile': performance_profile,
            'pillar_importance': pillar_importance,
            'cluster_specific_importance': cluster_specific_importance,
            'avg_performance': cluster_center.mean()
        }
        cluster_profiles.append(profile)
        
        print(f"  Cluster {cluster_id}: {len(cluster_companies)} companies")
    
    return cluster_profiles

def generate_research_reports(clustered_data, cluster_analysis, statistical_analysis, pillar, sector_name, base_dir):
    """Generate research reports and visualizations"""
    
    print(f"\n📋 GENERATING RESEARCH REPORTS FOR {pillar}")
    print("-" * 50)
    
    # Save clustered data
    clustered_file = base_dir / pillar / 'cluster_data' / 'clustered_companies.csv'
    clustered_data.to_csv(clustered_file)
    
    # Save cluster KPI values (standardized)
    cluster_values_file = base_dir / pillar / 'cluster_data' / 'cluster_kpi_values.csv'
    clustered_data.groupby('Cluster').mean().to_csv(cluster_values_file)
    
    # Generate cluster-specific reports
    generate_cluster_reports(cluster_analysis, pillar, base_dir)
    
    # Save statistical analysis results
    weight_dir = base_dir / 'Weight_Analysis'
    
    if 'pillar_importance' in statistical_analysis:
        statistical_analysis['pillar_importance'].to_csv(
            weight_dir / f'{pillar.lower()}_pillar_importance.csv', index=False
        )
    
    if 'anova_results' in statistical_analysis:
        statistical_analysis['anova_results'].to_csv(
            weight_dir / f'{pillar.lower()}_statistical_significance.csv', index=False
        )
    
    # Save cluster-specific importance
    if 'cluster_importance' in statistical_analysis:
        for cluster_id, importance_df in statistical_analysis['cluster_importance'].items():
            if importance_df is not None:
                importance_df.to_csv(
                    weight_dir / f'{pillar.lower()}_cluster_{cluster_id}_importance.csv', 
                    index=False
                )
    
    # Create comprehensive visualizations
    create_research_visualizations(clustered_data, cluster_analysis, statistical_analysis, pillar, sector_name, base_dir)

def generate_cluster_reports(cluster_analysis, pillar, base_dir):
    """Generate cluster-specific reports"""
    
    for cluster_profile in cluster_analysis:
        cluster_id = cluster_profile['cluster_id']
        
        # Create comprehensive report
        report_data = []
        
        for kpi in cluster_profile['performance_profile'].index:
            performance_coef = cluster_profile['performance_profile'][kpi]
            
            # Get pillar-wide importance
            pillar_imp = (cluster_profile['pillar_importance'].get(kpi, 0) 
                         if cluster_profile['pillar_importance'] is not None else None)
            
            # Get cluster-specific importance
            cluster_imp = (cluster_profile['cluster_specific_importance'].get(kpi, 0) 
                          if cluster_profile['cluster_specific_importance'] is not None else None)
            
            report_data.append({
                'KPI': kpi,
                'Performance_Coefficient': performance_coef,
                'Pillar_Wide_Importance': pillar_imp,
                'Cluster_Specific_Importance': cluster_imp,
                'Absolute_Performance': abs(performance_coef)
            })
        
        report_df = pd.DataFrame(report_data)
        
        # Save cluster defining KPIs (sorted by cluster-specific importance)
        if any(report_df['Cluster_Specific_Importance'].notna()):
            cluster_sorted = report_df.sort_values('Cluster_Specific_Importance', ascending=False)
            cluster_sorted.to_csv(base_dir / pillar / 'cluster_data' / f'cluster_{cluster_id}_defining_kpis.csv', index=False)
        
        print(f"  Cluster {cluster_id}: {cluster_profile['size']} companies analyzed")

def create_research_visualizations(clustered_data, cluster_analysis, statistical_analysis, pillar, sector_name, base_dir):
    """Create all research visualizations"""
    
    viz_dir = base_dir / pillar / 'visualizations'
    
    # Pillar-wide Importance Chart
    create_pillar_importance_chart(statistical_analysis, pillar, sector_name, viz_dir)
    
    # ANOVA Statistical Significance Chart
    create_anova_significance_chart(statistical_analysis, pillar, sector_name, viz_dir)
    
    # Cluster KPI Profile Charts
    create_cluster_profile_charts(cluster_analysis, pillar, sector_name, viz_dir)
    
    # Cluster Importance Charts
    create_cluster_importance_charts(cluster_analysis, pillar, sector_name, viz_dir)

def create_pillar_importance_chart(statistical_analysis, pillar, sector_name, viz_dir):
    """Create pillar-wide importance chart"""
    if 'pillar_importance' not in statistical_analysis:
        return

    pillar_importance = statistical_analysis['pillar_importance'].head(20)

    plt.figure(figsize=(14, 10))
    y_pos = np.arange(len(pillar_importance))

    plt.barh(y_pos, pillar_importance['Pillar_Importance'], color=THEME['pillar_importance'], alpha=0.82)
    plt.yticks(y_pos, [textwrap.fill(kpi, 50) for kpi in pillar_importance['KPI']])
    plt.xlabel('Pillar-Wide Importance Weight')
    plt.title(f'{sector_name} - {pillar} Pillar\nTop 20 Most Important KPIs (All Clusters)',
             fontsize=14, fontweight='bold', pad=20)

    # Add value labels at the end of bars
    max_importance = pillar_importance['Pillar_Importance'].max()
    for i, (importance, kpi) in enumerate(zip(pillar_importance['Pillar_Importance'], pillar_importance['KPI'])):
        plt.text(importance + 0.004, i, f'{importance:.3f}',
                va='center', fontweight='bold', fontsize=10, color='black')

    plt.xlim(0, max_importance + 0.03)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()

    importance_path = viz_dir / 'pillar_importance_chart.png'
    plt.savefig(importance_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ Pillar importance chart saved: {importance_path}")

def create_anova_significance_chart(statistical_analysis, pillar, sector_name, viz_dir):
    """Create ANOVA statistical significance chart"""
    if 'anova_results' not in statistical_analysis:
        return

    # Get top 20 KPIs by F-statistic
    anova_data = statistical_analysis['anova_results'].head(20)

    plt.figure(figsize=(16, 12))
    y_pos = np.arange(len(anova_data))

    # Color bars by significance level
    colors = []
    for sig in anova_data['Significance']:
        colors.append(THEME['anova'].get(sig, THEME['anova']['NS']))

    bars = plt.barh(y_pos, anova_data['F_Statistic'], color=colors, alpha=0.8)
    plt.yticks(y_pos, [textwrap.fill(kpi, 50) for kpi in anova_data['KPI']])
    plt.xlabel('F-Statistic (ANOVA)')
    plt.title(f'{sector_name} - {pillar} Pillar\nStatistical Significance of KPI Differences Between Clusters',
             fontsize=14, fontweight='bold', pad=20)

    # Add value labels and significance codes
    max_f = anova_data['F_Statistic'].max()
    pad = max(max_f * 0.05, 0.8)  # small offset to sit just outside the bar
    for i, (f_stat, p_val, sig) in enumerate(zip(anova_data['F_Statistic'],
                                                anova_data['P_Value'],
                                                anova_data['Significance'])):
        label_text = f'F={f_stat:.1f} | p={p_val:.3f} | {sig}'
        plt.text(f_stat + pad, i, label_text,
                va='center', fontweight='bold', fontsize=10,
                bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9),
                ha='left', clip_on=False)

    plt.xlim(0, max_f + pad * 3)
    plt.gca().invert_yaxis()
    plt.grid(True, axis='x', alpha=0.3)

    # Add legend for significance codes
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=THEME['anova']['***'], label='*** p < 0.001'),
        Patch(facecolor=THEME['anova']['**'], label='** p < 0.01'),
        Patch(facecolor=THEME['anova']['*'], label='* p < 0.05'),
        Patch(facecolor=THEME['anova']['NS'], label='NS p >= 0.05')
    ]
    plt.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()

    anova_path = viz_dir / 'anova_significance_chart.png'
    plt.savefig(anova_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  ✅ ANOVA significance chart saved: {anova_path}")

def create_cluster_profile_charts(cluster_analysis, pillar, sector_name, viz_dir):
    """Create KPI profile charts for each cluster"""
    for cluster_profile in cluster_analysis:
        cluster_id = cluster_profile['cluster_id']
        # Get top 15 KPIs by absolute performance
        performance_data = cluster_profile['performance_profile'].abs().nlargest(15)
        kpis = performance_data.index
        values = cluster_profile['performance_profile'].loc[kpis]

        plt.figure(figsize=(12, 8))
        y_pos = np.arange(len(kpis))

        colors = [THEME['perf_neg'] if x < 0 else THEME['perf_pos'] for x in values]
        bars = plt.barh(y_pos, values, color=colors, alpha=0.82)

        plt.yticks(y_pos, [textwrap.fill(kpi, 50) for kpi in kpis])
        plt.xlabel('Standardized Z-Score')
        plt.title(
            f'{sector_name} - {pillar} Pillar\nCluster {cluster_id}: KPI Performance Profile',
            fontsize=14,
            fontweight='bold',
            pad=18
        )

        # Set appropriate x-axis limits with cushion
        max_abs = max(abs(values.min()), abs(values.max()))
        cushion = 0.35
        plt.xlim(-max_abs - cushion, max_abs + cushion)

        # Add value labels at the end of bars
        span = plt.xlim()[1] - plt.xlim()[0]
        pad = 0.025 * span  # small offset outside the bar
        for i, value in enumerate(values):
            x_pos = value + pad if value >= 0 else value - pad
            ha = 'left' if value >= 0 else 'right'
            plt.text(
                x_pos,
                i,
                f'{value:+.2f}',
                va='center',
                ha=ha,
                fontweight='bold',
                fontsize=9,
                color='black',
                clip_on=False
            )

        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        profile_path = viz_dir / f'cluster_{cluster_id}_kpi_profile.png'
        plt.savefig(profile_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Cluster {cluster_id} profile chart saved: {profile_path}")

def create_cluster_importance_charts(cluster_analysis, pillar, sector_name, viz_dir):
    """Create importance charts for each cluster"""
    for cluster_profile in cluster_analysis:
        cluster_id = cluster_profile['cluster_id']

        if cluster_profile['cluster_specific_importance'] is None:
            continue

        # Get top 15 KPIs by cluster-specific importance
        importance_data = pd.Series(cluster_profile['cluster_specific_importance']).nlargest(15)

        plt.figure(figsize=(14, 10))
        y_pos = np.arange(len(importance_data))

        plt.barh(y_pos, importance_data.values, color=THEME['cluster_importance'], alpha=0.82)  # Orange for importance
        plt.yticks(y_pos, [textwrap.fill(kpi, 50) for kpi in importance_data.index])
        plt.xlabel('Cluster-Specific Importance Weight')
        plt.title(f'{sector_name} - {pillar} Pillar\nCluster {cluster_id}: Defining KPIs',
                 fontsize=14, fontweight='bold', pad=20)

        # Add value labels at the end of bars
        max_importance = importance_data.max()
        for i, (importance, kpi) in enumerate(zip(importance_data.values, importance_data.index)):
            plt.text(importance + 0.005, i, f'{importance:.3f}',
                    va='center', fontweight='bold', fontsize=10, color='black')

        plt.xlim(0, max_importance + 0.02)
        plt.gca().invert_yaxis()
        plt.grid(True, axis='x', alpha=0.3)
        plt.tight_layout()

        importance_path = viz_dir / f'cluster_{cluster_id}_importance_chart.png'
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        plt.close()

        print(f"  ✅ Cluster {cluster_id} importance chart saved: {importance_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == "__main__":
    # Run the comprehensive research analysis
    results = analyze_sector_enhanced()
    
    print(f"\n{'='*70}")
    print("RESEARCH ANALYSIS COMPLETE!")
    print(f"{'='*70}")
    print("\n🎯 RESEARCH COMPONENTS INCLUDED:")
    print("1. Elbow Method + Silhouette Analysis for optimal clusters")
    print("2. Z-score standardization only")
    print("3. KNN imputation only") 
    print("4. Pillar-wide + Cluster-specific feature importance")
    print("5. ANOVA statistical significance testing")
    print("6. Comprehensive visualizations for research presentation")
    print("\n📁 Check pillar folders for:")
    print("   - Elbow analysis charts")
    print("   - ANOVA significance charts")
    print("   - KPI performance profiles per cluster")
    print("   - Importance charts (pillar-wide + cluster-specific)")




