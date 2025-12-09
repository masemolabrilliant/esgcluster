
# phase1.py - Complete with triple filter system

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from pathlib import Path
import json
from config import CONFIG

# Palette aligned with phase C visuals
THEME = {
    'companies': '#2E86AB',    # pillar importance blue
    'kpis': '#E67E22',         # cluster importance orange
    'completeness': '#27AE60'  # positive performance green
}
def setup_directories():
    """Create organized folder structure for data check phase"""
    base_dir = Path(CONFIG['phase1']['output_folder'])
    folders = ['matrices', 'sector_stats', 'visualizations']
    
    for folder in folders:
        (base_dir / folder).mkdir(parents=True, exist_ok=True)
    
    print("📁 Data Check Phase Folders Created:")
    for folder in folders:
        print(f"   - {base_dir / folder}/")
    return base_dir

def get_latest_company_data(df):
    """Get the latest available data for each company"""
    print("Getting latest available data for each company...")
    
    df_sorted = df.sort_values(['Company', 'year'], ascending=[True, False])
    latest_data = df_sorted.drop_duplicates(['Company', 'KPI']).copy()
    
    print(f"Original data: {len(df):,} records")
    print(f"After taking latest per company-KPI: {len(latest_data):,} records")
    print(f"Unique companies: {latest_data['Company'].nunique()}")
    
    return latest_data

def analyze_sector_pillars(latest_data, base_dir):
    """Analyze each sector's data for each ESG pillar separately"""
    print(f"\n{'='*60}")
    print("PILLAR-SEPARATED SECTOR ANALYSIS")
    print(f"{'='*60}")
    
    sector_results = []
    
    for sector in latest_data['Sector'].unique():
        print(f"\n📊 Analyzing {sector}...")
        sector_data = latest_data[latest_data['Sector'] == sector]
        
        for pillar in CONFIG['general']['esg_pillars']:
            pillar_data = sector_data[sector_data['indicator_type'] == pillar]
            
            if len(pillar_data) == 0:
                continue
                
            pivot_matrix = pillar_data.pivot_table(
                index='Company', 
                columns='KPI', 
                values='value'
            )
            
            companies = len(pivot_matrix)
            kpis = len(pivot_matrix.columns)
            total_cells = pivot_matrix.size
            data_points_with_values = pivot_matrix.notnull().sum().sum()
            completeness_pct = (data_points_with_values / total_cells) * 100 if total_cells > 0 else 0
            
            matrix_filename = base_dir / 'matrices' / f'matrix_{sector}_{pillar}.csv'
            pivot_matrix.to_csv(matrix_filename)
            
            pillar_info = {
                'Sector': sector,
                'Pillar': pillar,
                'Companies': companies,
                'KPIs': kpis,
                'Total_Cells': total_cells,
                'Data_Points_With_Values': data_points_with_values,
                'Completeness_Percentage': completeness_pct,
                'Data_Points_Per_Company': data_points_with_values / companies if companies > 0 else 0
            }
            sector_results.append(pillar_info)
            
            print(f"   {pillar}: {companies} companies × {kpis} KPIs = {data_points_with_values:,} data points ({completeness_pct:.1f}%)")
    
    results_df = pd.DataFrame(sector_results)
    stats_file = base_dir / 'sector_stats' / 'sector_pillar_analysis.csv'
    results_df.to_csv(stats_file, index=False)
    
    return results_df

def identify_qualifying_sectors(results_df, base_dir):
    """Identify sectors with sufficient data using triple filter"""
    print(f"\n{'='*60}")
    print("QUALIFYING SECTORS IDENTIFICATION")
    print(f"{'='*60}")

    MIN_COMPANIES = CONFIG['phase1']['min_companies']
    MIN_KPIS = CONFIG['phase1']['min_kpis']
    MIN_COMPLETENESS = CONFIG['phase1']['min_completeness']

    print(f"TRIPLE FILTER CRITERIA:")
    print(f"  • Minimum {MIN_COMPANIES} companies per pillar")
    print(f"  • Minimum {MIN_KPIS} KPIs per pillar")
    print(f"  • Minimum {MIN_COMPLETENESS}% completeness per pillar")
    print()

    qualifying_sectors = {pillar: [] for pillar in CONFIG['general']['esg_pillars']}
    
    for pillar in CONFIG['general']['esg_pillars']:
        pillar_data = results_df[results_df['Pillar'] == pillar]
        
        for _, row in pillar_data.iterrows():
            if (row['Companies'] >= MIN_COMPANIES and 
                row['KPIs'] >= MIN_KPIS and
                row['Completeness_Percentage'] >= MIN_COMPLETENESS):
                
                qualifying_sectors[pillar].append({
                    'Sector': row['Sector'],
                    'Companies': row['Companies'],
                    'KPIs': row['KPIs'],
                    'Data_Points_With_Values': row['Data_Points_With_Values'],
                    'Completeness_Percentage': row['Completeness_Percentage'],
                    'Data_Per_Company': row['Data_Points_Per_Company']
                })

    for pillar, sectors in qualifying_sectors.items():
        if sectors:
            # FIX: Sort the list before creating DataFrame AND before printing
            sectors_sorted = sorted(sectors, 
                                  key=lambda x: (x['Completeness_Percentage'], x['Companies'], x['Data_Points_With_Values']), 
                                  reverse=True)
            
            pillar_df = pd.DataFrame(sectors_sorted)
            qual_file = base_dir / 'sector_stats' / f'qualifying_sectors_{pillar.lower()}.csv'
            pillar_df.to_csv(qual_file, index=False)
            
            # FIX: Update the qualifying_sectors with sorted list
            qualifying_sectors[pillar] = sectors_sorted

    return qualifying_sectors
def create_pillar_comparison_charts(results_df, base_dir):
    """Create comprehensive comparison charts for each pillar with triple filter"""
    print(f"\n📊 CREATING PILLAR COMPARISON CHARTS")
    print(f"{'='*60}")
    
    for pillar in CONFIG['general']['esg_pillars']:
        print(f"Creating chart for {pillar} pillar...")
        
        pillar_data = results_df[results_df['Pillar'] == pillar].copy()
        
        if len(pillar_data) == 0:
            continue
            
        # Apply triple filter
        min_companies = CONFIG['phase1']['min_companies']
        min_kpis = CONFIG['phase1']['min_kpis'] 
        min_completeness = CONFIG['phase1']['min_completeness']
        
        filtered_data = pillar_data[
            (pillar_data['Companies'] >= min_companies) &
            (pillar_data['KPIs'] >= min_kpis) &
            (pillar_data['Completeness_Percentage'] >= min_completeness)
        ]
        
        # Sort by priority: highest completeness first, then companies, data points, KPIs
        filtered_data = filtered_data.sort_values(
            ['Completeness_Percentage', 'Companies', 'Data_Points_With_Values', 'KPIs'],
            ascending=[False, False, False, False]
        )
        
        if len(filtered_data) == 0:
            print(f"  No sectors meet the triple filter criteria for {pillar}")
            continue
        
        # Create the comprehensive comparison chart - FIXED SIZE
        fig, ax = plt.subplots(figsize=(12, 8))  # Reasonable size
        
        sectors = filtered_data['Sector']
        y_pos = np.arange(len(sectors))
        bar_width = 0.25  # Reasonable bar width
        
        # Create only 3 bars - FIXED: Companies, KPIs, Completeness
        companies_bars = ax.barh(y_pos - bar_width, filtered_data['Companies'], 
                               bar_width, label='Companies', color=THEME['companies'], alpha=0.82)
        
        kpis_bars = ax.barh(y_pos, filtered_data['KPIs'], 
                          bar_width, label='KPIs', color=THEME['kpis'], alpha=0.82)
        
        completeness_bars = ax.barh(y_pos + bar_width, filtered_data['Completeness_Percentage'], 
                                  bar_width, label='Completeness %', color=THEME['completeness'], alpha=0.82)
        
        # Customize the chart
        ax.set_xlabel('Values', fontsize=12, fontweight='bold')
        ax.set_ylabel('Sectors', fontsize=12, fontweight='bold')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sectors, fontsize=10)
        ax.set_title(f'{pillar} Pillar Qualified Sector Comparison',
                    fontsize=14, fontweight='bold', pad=18)
        ax.invert_yaxis()  # show highest-ranked at the top
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, axis='x', alpha=0.3)
        
        # Add value labels on bars - FIXED: Only for the 3 metrics we're showing
        max_x_value = max(filtered_data['Companies'].max(), 
                         filtered_data['KPIs'].max(), 
                         filtered_data['Completeness_Percentage'].max())
        label_offset = max_x_value * 0.02
        
        for i, (company, kpi, completeness) in enumerate(zip(
            filtered_data['Companies'], filtered_data['KPIs'], 
            filtered_data['Completeness_Percentage'])):
            
            ax.text(company + label_offset, y_pos[i] - bar_width, f'{company}', 
                   va='center', fontsize=9, fontweight='bold')
            ax.text(kpi + label_offset, y_pos[i], f'{kpi}', 
                   va='center', fontsize=9, fontweight='bold')
            ax.text(completeness + label_offset, y_pos[i] + bar_width, f'{completeness:.1f}%', 
                   va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = base_dir / 'visualizations' / f'{pillar.lower()}_qualified_sectors.png'
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ {pillar} chart saved: {chart_path}")
        print(f"  📈 Showing {len(filtered_data)} sectors that meet all criteria")

def main():
    """Phase 1: Comprehensive Data Health Check Analysis"""
    
    try:
        base_dir = setup_directories()
        
        print("🔍 PHASE 1: COMPREHENSIVE DATA HEALTH CHECK")
        print("="*60)
        print("TRIPLE FILTER CONFIGURATION:")
        print(f"  • Minimum Companies: {CONFIG['phase1']['min_companies']}")
        print(f"  • Minimum KPIs: {CONFIG['phase1']['min_kpis']}")
        print(f"  • Minimum Completeness: {CONFIG['phase1']['min_completeness']}%")
        print("="*60)
        
        print("Loading ESG data...")
        df = pd.read_csv(CONFIG['general']['data_file'])
        
        latest_data = get_latest_company_data(df)
        
        print(f"Analyzing {len(latest_data):,} records")
        print(f"Found {latest_data['Sector'].nunique()} sectors")
        
        results_df = analyze_sector_pillars(latest_data, base_dir)
        qualifying_sectors = identify_qualifying_sectors(results_df, base_dir)
        create_pillar_comparison_charts(results_df, base_dir)
        
        print(f"\n🎯 QUALIFYING SECTORS SUMMARY BY PILLAR:")
        print("=" * 60)
        
        total_qualified = 0
        for pillar in CONFIG['general']['esg_pillars']:
            sectors = qualifying_sectors[pillar]  # This should now be sorted
            total_qualified += len(sectors)
            print(f"\n{pillar.upper()} PILLAR: {len(sectors)} qualifying sectors")
            print("-" * 40)
            
            if sectors:
                for i, sector in enumerate(sectors, 1):
                    print(f"{i}. {sector['Sector']}")
                    print(f"   Companies: {sector['Companies']:,}")
                    print(f"   KPIs: {sector['KPIs']:,}")
                    print(f"   Data Points: {sector['Data_Points_With_Values']:,}")
                    print(f"   Completeness: {sector['Completeness_Percentage']:.1f}%")
            else:
                print("   No sectors meet all criteria")
        
        print(f"\n📊 TOTAL QUALIFIED SECTORS: {total_qualified} across all pillars")
        print(f"📁 All Phase 1 results saved in: {base_dir}/")
        print(f"💡 Adjust thresholds in config.py if too many/few sectors qualify")
        
        return results_df, qualifying_sectors
        
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return None, None
if __name__ == "__main__":
    results, qualifying = main()



