# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 11:56:29 2025

@author: miklos.szakal
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.io as pio
import webbrowser

# Beolvasás
df = pd.read_csv('C:/Users/miklos.szakal/Documents/Corvinus/szakdoga/forrasok/estat_gov_10a_exp$defaultview_filtered_en.csv.gz',
                 compression='gzip')


# Adattisztítás - nem szükséges évek, oszlopok és sorok törlése
years_to_remove = [2013, 2014, 2015, 2016, 2017]
df = df[~df['TIME_PERIOD'].isin(years_to_remove)]

df = df[df['sector'] == 'General government']

columns_to_drop = ['DATAFLOW', 'LAST UPDATE', 'freq', 'unit', 'na_item', 'CONF_STATUS', 'OBS_FLAG']
df = df.drop(columns=columns_to_drop)

df = df.dropna(subset=['OBS_VALUE'])

df = df[~(df['geo'].str.startswith('Euro') | (df['geo'].str.startswith('European')))]


# Leíró statisztika
grouped_stats = df.groupby(['cofog99', 'TIME_PERIOD'])['OBS_VALUE'].describe()
grouped_stats = grouped_stats.reset_index()

#grouped_stats.to_excel("adatok_abrakhoz.xlsx", index=False)

# Kiadási kategóriák oszloppá konvertálása
df_pivot = df.pivot_table(index=['sector', 'geo', 'TIME_PERIOD'],
                            columns='cofog99',  
                            values='OBS_VALUE')

df_pivot = df_pivot.reset_index()


# Optimális k kiválasztása 
def elbow_gorbe_general_government(df_pivot, year):
    df_year = df_pivot[
        (df_pivot['TIME_PERIOD'] == year) & (df_pivot['sector'] == 'General government')
    ].drop(columns=['TIME_PERIOD', 'sector']).set_index('geo')

    scaler = StandardScaler()
    standardized = pd.DataFrame(
        scaler.fit_transform(df_year),
        columns=df_year.columns,
        index=df_year.index
    )

    inertias = []
    for k in range(1, 11):
        kmeans = KMeans(n_clusters=k, random_state=2002, n_init=20)
        kmeans.fit(standardized)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(6, 4))
    plt.plot(range(1, 11), inertias, marker='o')
    plt.title(f"Elbow-módszer – {year}")
    plt.xlabel("Klaszterszám (k)")
    plt.ylabel("Inertia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
elbow_gorbe_general_government(df_pivot, 2018)
elbow_gorbe_general_government(df_pivot, 2019)
elbow_gorbe_general_government(df_pivot, 2020)
elbow_gorbe_general_government(df_pivot, 2021)
elbow_gorbe_general_government(df_pivot, 2022)


# Az összes évre klaszterezés

years_to_analyze = [2018, 2019, 2020, 2021, 2022]

# Évhez kötött táblák létrehozása
tables_2018 = tables_2019 = tables_2020 = tables_2021 = tables_2022 = None

# Klaszterezés minden évre
for year in years_to_analyze:
    df_year = df_pivot[(df_pivot['TIME_PERIOD'] == year) & (df_pivot['sector'] == 'General government')]\
              .drop(columns=['TIME_PERIOD', 'sector']).set_index('geo')


    # Standardizálás
    scaler = StandardScaler()
    standardized = pd.DataFrame(scaler.fit_transform(df_year),
                                columns=df_year.columns,
                                index=df_year.index)

    # KMeans klaszterezés k=6
    kmeans = KMeans(n_clusters=6, random_state=2002, n_init=20).fit(standardized)
    standardized['Cluster'] = kmeans.labels_

    if year == 2018:
        tables_2018 = standardized
    elif year == 2019:
        tables_2019 = standardized
    elif year == 2020:
        tables_2020 = standardized
    elif year == 2021:
        tables_2021 = standardized
    elif year == 2022:
        tables_2022 = standardized


# Hőtérkép a klaszterátlagok pontos értékeiről
def plot_heatmap(df, year):
    plt.figure(figsize=(12, 6))
    cluster_means = df.groupby('Cluster').mean()
    sns.heatmap(cluster_means, cmap='coolwarm', annot=True, fmt=".2f", linewidths=0.5, linecolor='gray')
    plt.title(f"Kiadási kategóriák klaszterenkénti átlaga – {year}")
    plt.xlabel("Kiadási kategória")
    plt.ylabel("Klaszter")
    plt.tight_layout()
    plt.show()

plot_heatmap(tables_2018, 2018)
plot_heatmap(tables_2019, 2019)
plot_heatmap(tables_2020, 2020)
plot_heatmap(tables_2021, 2021)
plot_heatmap(tables_2022, 2022)



# Évek összehasonlítása kereszttábla elemzéssel
clusters_2018 = tables_2018['Cluster']
clusters_2019 = tables_2019['Cluster']
clusters_2020 = tables_2020['Cluster']
clusters_2021 = tables_2021['Cluster']
clusters_2022 = tables_2022['Cluster']

cross_2018_2019 = pd.crosstab(clusters_2018, clusters_2019)
cross_2019_2020 = pd.crosstab(clusters_2019, clusters_2020)
cross_2020_2021 = pd.crosstab(clusters_2020, clusters_2021)
cross_2021_2022 = pd.crosstab(clusters_2021, clusters_2022)


# Országok évenkénti klaszter besorolása
def extract_cluster_table(table, year):
    df = table[['Cluster']].copy()
    df = df.reset_index().rename(columns={'geo': 'Ország', 'Cluster': year})
    return df

cluster_table_2018 = extract_cluster_table(tables_2018, 2018)
cluster_table_2019 = extract_cluster_table(tables_2019, 2019)
cluster_table_2020 = extract_cluster_table(tables_2020, 2020)
cluster_table_2021 = extract_cluster_table(tables_2021, 2021)
cluster_table_2022 = extract_cluster_table(tables_2022, 2022)


merged_cluster_table = cluster_table_2018.merge(cluster_table_2019, on='Ország')\
                                         .merge(cluster_table_2020, on='Ország')\
                                         .merge(cluster_table_2021, on='Ország')\
                                         .merge(cluster_table_2022, on='Ország')




#  A klaszterek szemléltetése térképen

def klaszter_terkep(table, year):
    custom_palette = ['#0d0887', '#6a00a8', '#cb4679', '#f89441', '#f0f921', '#ffd700']

    fig = px.choropleth(
        table,
        locations='Ország',
        locationmode='country names',
        color=year,
        title=f'Európai országok klaszterei – {year}',
        color_continuous_scale=None,
        color_discrete_sequence=custom_palette,
        projection='natural earth'
    )
    
    fig.update_geos(
        visible=True,
        resolution=50,
        showcountries=True,
        countrycolor="Black",
        lataxis_range=[35, 70],   
        lonaxis_range=[-25, 40]
    )
    
    fig.update_layout(
        title={
            'text': f'Európai országok klaszterei – {year}',
            'font': {'size': 26},
        },
        coloraxis_colorbar=dict(
           title='Klaszter',
           titlefont=dict(size=20),
           tickfont=dict(size=16)
       )

    )
    filename = f"cluster_map_{year}.html"
    pio.write_html(fig, file=filename, auto_open=False)
    webbrowser.open(filename)


klaszter_terkep(cluster_table_2018, 2018)
klaszter_terkep(cluster_table_2019, 2019)
klaszter_terkep(cluster_table_2020, 2020)
klaszter_terkep(cluster_table_2021, 2021)
klaszter_terkep(cluster_table_2022, 2022)











