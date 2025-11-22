# --- Core packages ---
import os
import pandas as pd
import numpy as np

# --- Visualization ---
import plotly.graph_objects as go
import plotly.express as px

# --- Geospatial ---

import requests
import json



# BASIN_CONFIG = {
#     'Gunnison': {
#         'streamflow_col': 'Gunnison',
#         'outflow_id': 'gunn_end',
#         'parquet_path': 'DATA/PARQUET_FILES/gm/gm2015B.parquet',
#         'structure_files': {
#             'industry': 'ind_gm.txt',
#             'irrigation': 'irrigation_gm.txt',
#             'municipality': 'mun_gm.txt'
#         },
#         'drought_path': 'DATA/PARQUET_FILES/gm/gm2015B.parquet',
#         'sdis_path': 'DATA/sdis.csv',
#         'districts_path': 'DATA/div4_districts.geojson',
#         'map_center': {"lat": 38.5, "lon": -107.8},
#         'reservoir_path': 'DATA/EXCEL/BM_hist_xre_data.csv'
#     },
#     'White': {
#         'streamflow_col': 'White',
#         'outflow_id': 'white_end',
#         'parquet_path': 'DATA/PARQUET_FILES/wm/wm2015B.parquet',
#         'structure_files': {
#             'industry': 'ind_wm.txt',
#             'irrigation': 'irrigation_wm.txt',
#             'municipality': 'mun_wm.txt'
#         },
#         'drought_path': 'DATA/PARQUET_FILES/wm/wm2015B.parquet',
#         'sdis_path': 'DATA/boundaries_wm_ym/sdis_white_full.csv',
#         'districts_path': 'DATA/boundaries_wm_ym/div6_districts_wm_ym.geojson',
#         'map_center': {"lat": 39.9, "lon": -107.5},
#         'reservoir_path': 'DATA/EXCEL/BM_hist_xre_data.csv'
#     },
#     'Yampa': {
#         'streamflow_col': 'Yampa',
#         'outflow_id': 'yampa_end',
#         'parquet_path': 'DATA/PARQUET_FILES/ym/ym2015B.parquet',
#         'structure_files': {
#             'industry': 'ind_ym.txt',
#             'irrigation': 'irrigation_ym.txt',
#             'municipality': 'mun_ym.txt'
#         },
#         'drought_path': 'DATA/PARQUET_FILES/ym/ym2015B.parquet',
#         'sdis_path': 'DATA/boundaries_wm_ym/sdis_white_full.csv',
#         'districts_path': 'DATA/boundaries_wm_ym/div6_districts_wm_ym.geojson',
#         'map_center': {"lat": 40.0, "lon": -106.9},
#         'reservoir_path': 'DATA/EXCEL/BM_hist_xre_data.csv'
#     },
#     'Upper Colorado': {
#         'streamflow_col': 'Upper_Colorado', 
#         'outflow_id': 'cm_end',
#         'parquet_path': 'DATA/PARQUET_FILES/cm/cm2015B.parquet',
#         'structure_files': {
#             'industry': 'ind_cm.txt',
#             'irrigation': 'irrigation_cm.txt',
#             'municipality': 'mun_cm.txt'
#         },
#         'drought_path': 'DATA/PARQUET_FILES/cm/cm2015B.parquet',
#         'sdis_path': 'DATA/boundaries_colorado/cm_sdis.csv',
#         'districts_path': 'DATA/boundaries_colorado/Div5_districts_cm.geojson',
#         'map_center': {"lat": 39.5, "lon": -106.8},
#         'reservoir_path': 'DATA/EXCEL/LG_xre_data_hist.csv'
#     },
#     'San Juan & Dolores': {
#         'streamflow_col': 'SanJuan_Dolores',  
#         'outflow_id': 'sj_end',
#         'parquet_path': 'DATA/PARQUET_FILES/sj/sj2015B.parquet',
#         'structure_files': {
#             'industry': 'ind_sj.txt',
#             'irrigation': 'irrigation_sj.txt',
#             'municipality': 'mun_sj.txt'
#         },
#         'drought_path': 'DATA/PARQUET_FILES/sj/sj2015B.parquet',
#         'sdis_path': 'DATA/boundaries_sanjuan/sj_sdis.csv',
#         'districts_path': 'DATA/boundaries_sanjuan/div7_districts_sj.geojson',
#         'map_center': {"lat": 37.5, "lon": -108.5},
#         'reservoir_path': 'DATA/EXCEL/MR_xre_data_hist.csv'
#     }
# }

BASIN_CONFIG = {
    'Gunnison': {
        'streamflow_col': 'Gunnison',
        'outflow_id': 'gunn_end',
        'parquet_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/gm/gm2015B.parquet',
        'structure_files': {
            'industry': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/ind_gm.txt',
            'irrigation': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/irrigation_gm.txt',
            'municipality': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/mun_gm.txt'
        },
        'drought_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/gm/gm2015B.parquet',
        'sdis_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/sdis.csv',
        'districts_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/div4_districts.geojson',
        'map_center': {"lat": 38.5, "lon": -107.8},
        'reservoir_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/EXCEL/BM_hist_xre_data.csv'
    },
    'White': {
        'streamflow_col': 'White',
        'outflow_id': 'white_end',
        'parquet_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/wm/wm2015B.parquet',
        'structure_files': {
            'industry': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/ind_wm.txt',
            'irrigation': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/irrigation_wm.txt',
            'municipality': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/mun_wm.txt'
        },
        'drought_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/wm/wm2015B.parquet',
        'sdis_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_wm_ym/sdis_white_full.csv',
        'districts_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_wm_ym/div6_districts_wm_ym.geojson',
        'map_center': {"lat": 39.9, "lon": -107.5},
        'reservoir_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/EXCEL/BM_hist_xre_data.csv'
    },
    'Yampa': {
        'streamflow_col': 'Yampa',
        'outflow_id': 'yampa_end',
        'parquet_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/ym/ym2015B.parquet',
        'structure_files': {
            'industry': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/ind_ym.txt',
            'irrigation': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/irrigation_ym.txt',
            'municipality': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/mun_ym.txt'
        },
        'drought_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/ym/ym2015B.parquet',
        'sdis_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_wm_ym/sdis_white_full.csv',
        'districts_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_wm_ym/div6_districts_wm_ym.geojson',
        'map_center': {"lat": 40.0, "lon": -106.9},
        'reservoir_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/EXCEL/BM_hist_xre_data.csv'
    },
    'Upper Colorado': {
        'streamflow_col': 'Upper_Colorado',
        'outflow_id': 'cm_end',
        'parquet_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/cm/cm2015B.parquet',
        'structure_files': {
            'industry': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/ind_cm.txt',
            'irrigation': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/irrigation_cm.txt',
            'municipality': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/mun_cm.txt'
        },
        'drought_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/cm/cm2015B.parquet',
        'sdis_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_colorado/cm_sdis.csv',
        'districts_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_colorado/Div5_districts_cm.geojson',
        'map_center': {"lat": 39.5, "lon": -106.8},
        'reservoir_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/EXCEL/LG_xre_data_hist.csv'
    },
    'San Juan & Dolores': {
        'streamflow_col': 'SanJuan_Dolores',
        'outflow_id': 'sj_end',
        'parquet_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/sj/sj2015B.parquet',
        'structure_files': {
            'industry': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/ind_sj.txt',
            'irrigation': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/irrigation_sj.txt',
            'municipality': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/mun_sj.txt'
        },
        'drought_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/PARQUET_FILES/sj/sj2015B.parquet',
        'sdis_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_sanjuan/sj_sdis.csv',
        'districts_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/boundaries_sanjuan/div7_districts_sj.geojson',
        'map_center': {"lat": 37.5, "lon": -108.5},
        'reservoir_path': 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/EXCEL/MR_xre_data_hist.csv'
    }
}


####################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################

# Path to the full streamflow dataset
# STREAMFLOW_PATH = 'DATA/EXCEL/HistoricalData_all_basins.csv'

STREAMFLOW_PATH = 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/EXCEL/HistoricalData_all_basins.csv'

def streamflow(basin: str, path: str = STREAMFLOW_PATH) -> pd.DataFrame:
    """Loads and processes streamflow data for the selected basin."""
    if basin not in BASIN_CONFIG:
        raise ValueError(f"Basin '{basin}' not recognized in BASIN_CONFIG.")

    col = BASIN_CONFIG[basin]['streamflow_col']
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df.set_index('Date', inplace=True)

    if col not in df.columns:
        raise ValueError(f"Streamflow column '{col}' not found in dataset.")

    df_basin = df[[col]].copy()
    df_basin.rename(columns={col: 'annual'}, inplace=True)
    df_basin['basin'] = basin
    return df_basin

def streamflow_fig(selected_year: int = None, basin: str = "Gunnison") -> go.Figure:
    """Generates a streamflow plot for the selected basin and year."""
    df = streamflow(basin)

    fig = go.Figure()

    # Plot the streamflow line
    fig.add_trace(go.Scatter(
        x=df.index,
        y=df['annual'],
        mode='lines',
        name='Annual Flow',
        line=dict(color='royalblue')
    ))

    # Highlight drought years (below 25th percentile)
    threshold = df['annual'].quantile(0.25)
    drought_years = df[df['annual'] < threshold].index.year.unique()

    for year in drought_years:
        fig.add_vrect(
            x0=f"{year}-01-01", x1=f"{year}-12-31",
            fillcolor="lightgray", opacity=0.3,
            layer="below", line_width=0
        )

    # Add dummy trace for legend
    fig.add_trace(go.Scatter(
        x=[None], y=[None],
        mode='markers',
        marker=dict(color='lightgray'),
        name='Drought Years (<25th percentile)'
    ))

    # Add vertical line for selected year
    if selected_year is not None and selected_year in df.index.year:
        y_min = df['annual'].min()
        y_max = df['annual'].max()
        fig.add_shape(
            type='line',
            x0=f"{selected_year}-01-01", x1=f"{selected_year}-01-01",
            y0=y_min, y1=y_max,
            line=dict(color='firebrick', width=2, dash='dot')
        )
        fig.add_annotation(
            x=f"{selected_year}-01-01",
            y=y_max,
            text=f'Selected Year: {selected_year}',
            showarrow=False,
            font=dict(color='firebrick', size=12),
            yshift=10,
            xanchor='left'
        )

    # Final layout
    fig.update_layout(
        title=f'{basin} Annual Streamflow (1909–2013)',
        xaxis_title='Year',
        yaxis_title='Annual Flow (ft³)',
        template='plotly_white',
        height=400,
        autosize=True,
        margin=dict(l=30, r=70, t=60, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        font=dict(family="Arial", size=12)
    )

    return fig

####################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################



# --- File paths ---
# streamflow_path = 'DATA/EXCEL/HistoricalData_all_basins.csv'
# demand_base_path = 'DATA/Demand/Demand-IDs'

streamflow_path = 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/EXCEL/HistoricalData_all_basins.csv'
demand_base_path = 'https://raw.githubusercontent.com/WillemPoelma/colorado_river_DATA/main/Demand/Demand-IDs'

# --- Data loaders ---
def streamflow(basin, path=streamflow_path):
    col = BASIN_CONFIG[basin]['streamflow_col']
    df = pd.read_csv(path, sep=';', usecols=['Year', col])
    df['Year'] = pd.to_datetime(df['Year'], format='%Y', errors='coerce')
    df = df.set_index('Year').rename(columns={col: 'annual'})
    return df

def statemod(path):
    colnames = [
        'structure_name', 'structure_id', 'river_id', 'year', 'month',
        'demand_total', 'demand_cu', 'supply_total',
        'shortage_total', 'shortage_cu', 'station_balance_river_outflow', 'water_use_cu'
    ]
    df = pd.read_parquet(path, columns=colnames)
    df['structure_id'] = df['structure_id'].astype(str).str.replace('_D', '', regex=False).str.replace('_I', '', regex=False)
    df['month'] = df['month'].astype(str)
    df['year'] = df['year'].astype(int)
    return df[df['month'] == 'TOT'].copy()

def load_structure_ids(basin, base_path=demand_base_path):
    files = BASIN_CONFIG[basin]['structure_files']
    dfs = {k: pd.read_csv(f"{base_path}/{v}", header=None, sep=r'\s+') for k, v in files.items()}
    return (
        set(dfs['industry'].iloc[:, 0].astype(str)),
        set(dfs['irrigation'].iloc[:, 0].astype(str)),
        set(dfs['municipality'].iloc[:, 0].astype(str))
    )

# --- Generalized Sankey Generator ---
def make_sankey_fig(year, basin):
    if basin not in BASIN_CONFIG:
        raise ValueError(f"Basin '{basin}' not recognized.")

    # Load data
    inflow_df = streamflow(basin)
    df_annual = statemod(BASIN_CONFIG[basin]['parquet_path'])
    ind_set, irr_set, mun_set = load_structure_ids(basin)

    # Filter year
    df_year = df_annual[df_annual['year'] == int(year)].copy()
    if df_year.empty:
        return go.Figure().update_layout(title_text=f"No Sankey data available for {basin} in {year}")

    # Inflow
    inflow = inflow_df.loc[inflow_df.index.year == int(year), 'annual'].sum()
    if pd.isna(inflow) or inflow == 0:
        return go.Figure().update_layout(title_text=f"No inflow data for {basin} in {year}")

    # Outflow
    outflow_id = BASIN_CONFIG[basin]['outflow_id']
    outflow_raw = df_year[df_year['river_id'] == outflow_id]['station_balance_river_outflow'].sum()
    outflow = float(pd.to_numeric(outflow_raw, errors='coerce'))

    # Water use
    df_year['structure_id'] = df_year['structure_id'].astype(str)
    df_year['water_use_cu'] = pd.to_numeric(df_year['water_use_cu'], errors='coerce')

    industry = df_year[df_year['structure_id'].isin(ind_set)]['water_use_cu'].sum()
    irrigation = df_year[df_year['structure_id'].isin(irr_set)]['water_use_cu'].sum()
    municipality = df_year[df_year['structure_id'].isin(mun_set)]['water_use_cu'].sum()
    other = max(0, inflow - (industry + irrigation + municipality + outflow))

    values = [industry, irrigation, municipality, other, outflow]
    values = [0 if pd.isna(v) else v for v in values]

    if sum(values) == 0:
        return go.Figure().update_layout(title_text=f"No Sankey data available for {basin} in {year}")

    labels = ["Total Inflow", "Industry", "Irrigation", "Municipality", "Other", "Outflow"]
    sources = [0, 0, 0, 0, 0]
    targets = [1, 2, 3, 4, 5]

    fig = go.Figure(go.Sankey(
        node=dict(label=labels),
        link=dict(source=sources, target=targets, value=values)
    ))

    fig.update_layout(
        title_text=f"Water Flow in {basin} Basin ({year})",
        font_size=12,
        height=400,
        margin=dict(l=30, r=30, t=40, b=30)
    )
    return fig

####################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################

# def sector_water_use_shortage_plot(year, basin, demand_base_path='DATA/Demand/Demand-IDs'):
#     if basin not in BASIN_CONFIG:
#         raise ValueError(f"Basin '{basin}' not recognized.")

#     # Load StateMod data
#     parquet_path = BASIN_CONFIG[basin]['parquet_path']
#     cols = ['structure_id', 'year', 'month', 'shortage_cu', 'demand_cu']
#     df_raw = pd.read_parquet(parquet_path, columns=cols)
#     df_raw['structure_id'] = df_raw['structure_id'].astype(str)
#     df_raw['year'] = df_raw['year'].astype(int)

#     # Filter by selected year
#     df = df_raw[(df_raw['year'] == int(year)) & (df_raw['month'] == 'TOT')].copy()
#     if df.empty:
#         df = df_raw[df_raw['year'] == int(year)].copy()

#     # Load sector structure ID lists
#     files = BASIN_CONFIG[basin]['structure_files']
#     ind_ids = pd.read_csv(f"{demand_base_path}/{files['industry']}", header=None, sep=r'\s+')[0].astype(str)
#     irr_ids = pd.read_csv(f"{demand_base_path}/{files['irrigation']}", header=None, sep=r'\s+')[0].astype(str)
#     mun_ids = pd.read_csv(f"{demand_base_path}/{files['municipality']}", header=None, sep=r'\s+')[0].astype(str)

#     # Detect suffixes in filtered data
#     all_ids = set(df['structure_id'].dropna())
#     def detect_suffixes(raw_list, fallback):
#         matched = []
#         for raw in raw_list:
#             candidates = [sid for sid in all_ids if sid.startswith(raw)]
#             matched.append(candidates[0] if candidates else raw + fallback)
#         return set(matched)

#     ind_set = detect_suffixes(ind_ids, '_D')
#     irr_set = detect_suffixes(irr_ids, '_I')
#     mun_set = detect_suffixes(mun_ids, '_M')

#     # Clean numeric fields
#     df['shortage_cu'] = pd.to_numeric(df['shortage_cu'], errors='coerce')
#     df['demand_cu'] = pd.to_numeric(df['demand_cu'], errors='coerce')

#     # Sector filtering
#     ind_df = df[df['structure_id'].isin(ind_set)]
#     irr_df = df[df['structure_id'].isin(irr_set)]
#     mun_df = df[df['structure_id'].isin(mun_set)]

#     # Compute relative shortage values
#     def compute_ratio(df_sector):
#         total_shortage = df_sector['shortage_cu'].sum(skipna=True)
#         total_demand = df_sector['demand_cu'].sum(skipna=True)
#         return total_shortage / total_demand if total_demand > 0 else 0

#     shortage_ratios = [
#         compute_ratio(ind_df),
#         compute_ratio(irr_df),
#         compute_ratio(mun_df)
#     ]

#     # Build plot
#     labels = ['Industry', 'Irrigation', 'Municipality']
#     fig = go.Figure()

#     fig.add_trace(go.Bar(
#         x=labels,
#         y=shortage_ratios,
#         name='Shortage Fraction',
#         marker_color='indianred',
#         text=[f"{v:.2%}" if v > 0 else "" for v in shortage_ratios],
#         textposition='auto'
#     ))

#     fig.update_layout(
#         title=f"Sectoral Shortage Fraction in {basin} Basin ({year})",
#         yaxis_title='Shortage / Demand [%]',
#         template='plotly_white',
#         height=400,
#         width=700,
#         margin=dict(l=30, r=30, t=50, b=40),
#         legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
#         font=dict(family="Arial", size=12)
#     )


#     return fig



def sector_water_use_shortage_plot(year, basin):
    if basin not in BASIN_CONFIG:
        raise ValueError(f"Basin '{basin}' not recognized.")

    # --- Load StateMod data ---
    parquet_path = BASIN_CONFIG[basin]['parquet_path']
    cols = ['structure_id', 'year', 'month', 'shortage_cu', 'demand_cu']
    df_raw = pd.read_parquet(parquet_path, columns=cols)
    df_raw['structure_id'] = df_raw['structure_id'].astype(str)
    df_raw['year'] = df_raw['year'].astype(int)

    # --- Filter by selected year ---
    df = df_raw[(df_raw['year'] == int(year)) & (df_raw['month'] == 'TOT')].copy()
    if df.empty:
        df = df_raw[df_raw['year'] == int(year)].copy()

    # --- Load sector structure ID lists directly from BASIN_CONFIG URLs ---
    files = BASIN_CONFIG[basin]['structure_files']
    ind_ids = pd.read_csv(files['industry'], header=None, sep=r'\s+')[0].astype(str)
    irr_ids = pd.read_csv(files['irrigation'], header=None, sep=r'\s+')[0].astype(str)
    mun_ids = pd.read_csv(files['municipality'], header=None, sep=r'\s+')[0].astype(str)

    # --- Detect suffixes in filtered data ---
    all_ids = set(df['structure_id'].dropna())

    def detect_suffixes(raw_list, fallback):
        matched = []
        for raw in raw_list:
            candidates = [sid for sid in all_ids if sid.startswith(raw)]
            matched.append(candidates[0] if candidates else raw + fallback)
        return set(matched)

    ind_set = detect_suffixes(ind_ids, '_D')
    irr_set = detect_suffixes(irr_ids, '_I')
    mun_set = detect_suffixes(mun_ids, '_M')

    # --- Clean numeric fields ---
    df['shortage_cu'] = pd.to_numeric(df['shortage_cu'], errors='coerce')
    df['demand_cu'] = pd.to_numeric(df['demand_cu'], errors='coerce')

    # --- Sector filtering ---
    ind_df = df[df['structure_id'].isin(ind_set)]
    irr_df = df[df['structure_id'].isin(irr_set)]
    mun_df = df[df['structure_id'].isin(mun_set)]

    # --- Compute relative shortage values ---
    def compute_ratio(df_sector):
        total_shortage = df_sector['shortage_cu'].sum(skipna=True)
        total_demand = df_sector['demand_cu'].sum(skipna=True)
        return total_shortage / total_demand if total_demand > 0 else 0

    shortage_ratios = [
        compute_ratio(ind_df),
        compute_ratio(irr_df),
        compute_ratio(mun_df)
    ]

    # --- Build plot ---
    labels = ['Industry', 'Irrigation', 'Municipality']
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=labels,
        y=shortage_ratios,
        name='Shortage Fraction',
        marker_color='indianred',
        text=[f"{v:.2%}" if v > 0 else "" for v in shortage_ratios],
        textposition='auto'
    ))

    fig.update_layout(
        title=f"Sectoral Shortage Fraction in {basin} Basin ({year})",
        yaxis_title='Shortage / Demand [%]',
        template='plotly_white',
        height=400,
        width=700,
        margin=dict(l=30, r=30, t=50, b=40),
        legend=dict(orientation='h', y=1.05, x=0.5, xanchor='center'),
        font=dict(family="Arial", size=12)
    )

    return fig

####################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################


# def plot_drought_map(year, basin):
#     if basin not in BASIN_CONFIG:
#         raise ValueError(f"Basin '{basin}' not recognized.")

#     config = BASIN_CONFIG[basin]

#     # --- Load drought data ---
#     drought_df = pd.read_parquet(config['drought_path'])
#     drought_df["year"] = pd.to_numeric(drought_df["year"], errors="coerce")
#     drought_df["shortage_cu"] = pd.to_numeric(drought_df["shortage_cu"], errors="coerce").fillna(0)
#     drought_df["structure_id"] = drought_df["structure_id"].astype(str).str.strip()
#     drought_df["structure_id_clean"] = drought_df["structure_id"].str.replace(r'_[A-Z]+$', '', regex=True)

#     # --- Load structure-to-district mapping ---
#     sdis_df = pd.read_csv(config['sdis_path'])
#     sdis_df.columns = sdis_df.columns.str.strip().str.lower()

#     # Detect structure ID column
#     wdid_col = next((col for col in sdis_df.columns if col in ['wdid', 'sw_wdid1', 'structure_id', 'id']), None)
#     if wdid_col is None:
#         raise ValueError(f"No structure ID column found in {config['sdis_path']}. Available columns: {sdis_df.columns.tolist()}")

#     sdis_df[wdid_col] = sdis_df[wdid_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
#     sdis_df["wdid_clean"] = sdis_df[wdid_col].str.replace(r'_[A-Z]+$', '', regex=True)

#     # Detect district column in sdis_df
#     dist_col_sdis = next((col for col in sdis_df.columns if col in ['dist', 'district', 'district_id']), None)
#     if dist_col_sdis is None:
#         raise ValueError(f"No district column found in {config['sdis_path']}. Available columns: {sdis_df.columns.tolist()}")

#     sdis_df[dist_col_sdis] = sdis_df[dist_col_sdis].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

#     # --- Merge drought data with district info ---
#     drought_df = drought_df.merge(
#         sdis_df[["wdid_clean", dist_col_sdis]],
#         left_on="structure_id_clean",
#         right_on="wdid_clean",
#         how="left"
#     )

#     drought_df[dist_col_sdis] = drought_df[dist_col_sdis].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

#     # --- Filter for selected year ---
#     drought_year = drought_df[drought_df["year"] == int(year)].copy()
#     if drought_year.empty:
#         return go.Figure().update_layout(title_text=f"No drought data available for {basin} in {year}")

#     # --- Aggregate shortage by district ---
#     district_shortage = (
#         drought_year.groupby(dist_col_sdis, observed=False)["shortage_cu"]
#         .sum()
#         .reset_index()
#     )

#     # --- Load district boundaries ---
#     gdf = gpd.read_file(config['districts_path']).to_crs(epsg=4326)
#     gdf.columns = gdf.columns.str.strip().str.lower()

#     # Detect district column in gdf
#     dist_col_gdf = next((col for col in gdf.columns if col in ['dist', 'district', 'district_id']), None)
#     if dist_col_gdf is None:
#         raise ValueError(f"No district column found in {config['districts_path']}. Available columns: {gdf.columns.tolist()}")

#     gdf[dist_col_gdf] = gdf[dist_col_gdf].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

#     # --- Merge with spatial data ---
#     map_df = gdf.merge(district_shortage, left_on=dist_col_gdf, right_on=dist_col_sdis, how="left")
#     map_df["shortage_cu"] = map_df["shortage_cu"].fillna(0)

#     # --- Plot the map ---
#     fig = px.choropleth_mapbox(
#         map_df,
#         geojson=map_df.__geo_interface__,
#         locations=map_df[dist_col_gdf],
#         featureidkey=f"properties.{dist_col_gdf}",
#         color="shortage_cu",
#         color_continuous_scale="YlOrRd",
#         mapbox_style="carto-positron",
#         zoom=7,
#         center=config['map_center'],
#         opacity=0.7,
#         hover_name=dist_col_gdf,
#         hover_data={"shortage_cu": True}
#     )

#     fig.update_layout(
#         title=f"District-Level Drought Map in {basin} Basin ({year})",
#         margin={"r": 0, "t": 30, "l": 0, "b": 0},
#         font=dict(family="Arial", size=12)
#     )

#     return fig



def plot_drought_map(year, basin):
    if basin not in BASIN_CONFIG:
        raise ValueError(f"Basin '{basin}' not recognized.")

    config = BASIN_CONFIG[basin]

    # --- Load drought data ---
    drought_df = pd.read_parquet(config['drought_path'])
    drought_df["year"] = pd.to_numeric(drought_df["year"], errors="coerce")
    drought_df["shortage_cu"] = pd.to_numeric(drought_df["shortage_cu"], errors="coerce").fillna(0)
    drought_df["structure_id"] = drought_df["structure_id"].astype(str).str.strip()
    drought_df["structure_id_clean"] = drought_df["structure_id"].str.replace(r'_[A-Z]+$', '', regex=True)

    # --- Load structure-to-district mapping ---
    sdis_df = pd.read_csv(config['sdis_path'])
    sdis_df.columns = sdis_df.columns.str.strip().str.lower()

    # Detect structure ID column
    wdid_col = next((col for col in sdis_df.columns if col in ['wdid', 'sw_wdid1', 'structure_id', 'id']), None)
    if wdid_col is None:
        raise ValueError(f"No structure ID column found in {config['sdis_path']}. Available columns: {sdis_df.columns.tolist()}")

    sdis_df[wdid_col] = sdis_df[wdid_col].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)
    sdis_df["wdid_clean"] = sdis_df[wdid_col].str.replace(r'_[A-Z]+$', '', regex=True)

    # Detect district column in sdis_df
    dist_col_sdis = next((col for col in sdis_df.columns if col in ['dist', 'district', 'district_id']), None)
    if dist_col_sdis is None:
        raise ValueError(f"No district column found in {config['sdis_path']}. Available columns: {sdis_df.columns.tolist()}")

    sdis_df[dist_col_sdis] = sdis_df[dist_col_sdis].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    # --- Merge drought data with district info ---
    drought_df = drought_df.merge(
        sdis_df[["wdid_clean", dist_col_sdis]],
        left_on="structure_id_clean",
        right_on="wdid_clean",
        how="left"
    )

    drought_df[dist_col_sdis] = drought_df[dist_col_sdis].astype(str).str.strip().str.replace(r'\.0$', '', regex=True)

    # --- Filter for selected year ---
    drought_year = drought_df[drought_df["year"] == int(year)].copy()
    if drought_year.empty:
        return go.Figure().update_layout(title_text=f"No drought data available for {basin} in {year}")

    # --- Aggregate shortage by district ---
    district_shortage = (
        drought_year.groupby(dist_col_sdis, observed=False)["shortage_cu"]
        .sum()
        .reset_index()
    )

    # --- Load district boundaries (GeoJSON) ---
    resp = requests.get(config['districts_path'])
    geojson_data = resp.json()

    # --- Plot the map ---
    fig = px.choropleth_mapbox(
        district_shortage,
        geojson=geojson_data,
        locations=dist_col_sdis,
        featureidkey=f"properties.{dist_col_sdis}",
        color="shortage_cu",
        color_continuous_scale="YlOrRd",
        mapbox_style="carto-positron",
        zoom=7,
        center=config['map_center'],
        opacity=0.7,
        hover_name=dist_col_sdis,
        hover_data={"shortage_cu": True}
    )

    fig.update_layout(
        title=f"District-Level Drought Map in {basin} Basin ({year})",
        margin={"r": 0, "t": 30, "l": 0, "b": 0},
        font=dict(family="Arial", size=12)
    )

    return fig



####################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################



def load_reservoir_data(file_path):
    """Loads reservoir data from the specified CSV file using semicolon delimiter, skipping malformed rows."""
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    try:
        df = pd.read_csv(file_path, delimiter=';', on_bad_lines='skip')
    except Exception as e:
        raise ValueError(f"Failed to load reservoir data from {file_path}: {e}")
    
    df.columns = df.columns.str.strip().str.lower()
    return df

def plot_storage(year, basin):
    """Returns a Plotly figure showing Init. Storage per month for selected year and basin."""
    if basin not in BASIN_CONFIG:
        raise ValueError(f"Basin '{basin}' not recognized.")

    file_path = BASIN_CONFIG[basin].get('reservoir_path')
    if not file_path:
        raise ValueError(f"No reservoir data path defined for basin: {basin}")

    df = load_reservoir_data(file_path)

    # Validate required columns
    required_cols = ['year', 'mo', 'init. storage']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {file_path}: {missing}\nAvailable columns: {df.columns.tolist()}")

    # Clean and prepare data
    df['year'] = pd.to_datetime(df['year'], format='%Y', errors='coerce')
    df['init. storage'] = pd.to_numeric(df['init. storage'], errors='coerce')
    df['mo'] = df['mo'].str.upper().str.strip()

    # Define month order
    month_order = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN',
                   'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
    df = df[df['mo'].isin(month_order)]
    df['mo'] = pd.Categorical(df['mo'], categories=month_order, ordered=True)

    # Pivot to create Year x Month matrix
    df['year_num'] = df['year'].dt.year
    pivot_df = df.pivot_table(index='year_num', columns='mo', values='init. storage', aggfunc='mean')
    pivot_df = pivot_df.dropna(how='all')

    # Extract stats across years
    min_vals = pivot_df.min(axis=0)
    max_vals = pivot_df.max(axis=0)
    mean_vals = pivot_df.mean(axis=0)
    p10_vals = pivot_df.quantile(0.10, axis=0)

    # Extract selected year row
    year = int(year)
    if year not in pivot_df.index:
        return go.Figure().update_layout(title_text=f"No reservoir data available for {basin} in {year}")

    year_vals = pivot_df.loc[year]

    # Create figure
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=month_order, y=year_vals, mode='lines+markers', name=str(year), line=dict(color='teal', width=3)))
    fig.add_trace(go.Scatter(x=month_order, y=min_vals, mode='lines+markers', name='Minimum', line=dict(color='red', dash='dot')))
    fig.add_trace(go.Scatter(x=month_order, y=max_vals, mode='lines+markers', name='Maximum', line=dict(color='green', dash='dot')))
    fig.add_trace(go.Scatter(x=month_order, y=mean_vals, mode='lines+markers', name='Average', line=dict(color='blue', dash='dash')))
    fig.add_trace(go.Scatter(x=month_order, y=p10_vals, mode='lines+markers', name='10th Percentile', line=dict(color='orange', dash='dash')))

    fig.update_layout(
        title=f'Monthly Reservoir Storage in {basin} Basin: {year} vs Historical Range',
        xaxis_title='Month',
        yaxis_title='Storage (AF)',
        template='plotly_white',
        height=400,
        margin=dict(l=30, r=30, t=50, b=40),
        xaxis_tickangle=-45,
        font=dict(family="Arial", size=12)
    )

    return fig

####################################################################################################################################################################################################################################################################################################
##################################################################################################################################################################################################################################################################################################

# --- Loader Dispatcher: get_loader() ---
def get_loader(district: str) -> dict:
    """Returns a dictionary of plotting functions for the selected basin."""
    basin_map = {
        'gunnison': 'Gunnison',
        'white': 'White',
        'yampa': 'Yampa',
        'uppercolorado': 'Upper Colorado',
        'sanjuan': 'San Juan & Dolores'
    }

    basin = basin_map.get(district)
    if basin is None:
        raise ValueError(f"Unknown district: {district}")

    return {
        'plot_streamflow': lambda year: streamflow_fig(selected_year=year, basin=basin),
        'plot_storage': lambda year: plot_storage(year, basin=basin),
        'make_sankey_fig': lambda year: make_sankey_fig(year, basin=basin),
        'sector_water_use_shortage_plot': lambda year: sector_water_use_shortage_plot(year, basin=basin),
        'plot_drought_map': lambda year: plot_drought_map(year, basin=basin)
    }