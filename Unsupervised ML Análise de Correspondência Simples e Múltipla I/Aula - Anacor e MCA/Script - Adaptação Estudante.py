# -*- coding: utf-8 -*-

# Análise de Correspondência Simples e Múltipla
# MBA em Data Science e Analytics USP ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Instalando os pacotes

## Executar na linha de comando do console (sem o #)

# pip install pandas
# pip install numpy
# pip install scipy
# pip install plotly
# pip install seaborn
# pip install matplotlib
# pip install statsmodels
# pip install prince

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from scipy.stats import chi2_contingency
import prince
import plotly.express as px

#%% Análise de Correspondência Múltipla (MCA)

# Importando o banco de dados

dados_mca = pd.read_csv("estudantes_adapta.csv")
# Fonte: adaptado de https://www.kaggle.com/datasets/mdmahmudulhasansuzan/students-adaptability-level-in-online-education
# Suzan et al.(2021) Students' Adaptability Level Prediction in Online Education using Machine Learning Approaches - DOI: 10.1109/ICCCNT51525.2021.9579741

#%% Tabelas de frequências das variáveis

print(dados_mca['Education'].value_counts())
print(dados_mca['Institution'].value_counts())
print(dados_mca['Financial'].value_counts())
print(dados_mca['Internet'].value_counts())
print(dados_mca['Adaptivity'].value_counts())

#%% Analisando as tabelas de contingência

# Vamos gerar as tabelas de contingência em relação à "Adaptivity"

tabela_mca_1 = pd.crosstab(dados_mca["Adaptivity"], dados_mca["Education"])
tabela_mca_2 = pd.crosstab(dados_mca["Adaptivity"], dados_mca["Institution"])
tabela_mca_3 = pd.crosstab(dados_mca["Adaptivity"], dados_mca["Financial"])
tabela_mca_4 = pd.crosstab(dados_mca["Adaptivity"], dados_mca["Internet"])

print(tabela_mca_1)
print(tabela_mca_2)
print(tabela_mca_3)
print(tabela_mca_4)

#%% Analisando a significância estatística das associações (teste qui²)

tab_1 = chi2_contingency(tabela_mca_1)

print("Adaptivity x Education")
print(f"estatística qui²: {round(tab_1[0], 2)}")
print(f"p-valor da estatística: {round(tab_1[1], 4)}")
print(f"graus de liberdade: {tab_1[2]}")

tab_2 = chi2_contingency(tabela_mca_2)

print("Adaptivity x Institution")
print(f"estatística qui²: {round(tab_2[0], 2)}")
print(f"p-valor da estatística: {round(tab_2[1], 4)}")
print(f"graus de liberdade: {tab_2[2]}")

tab_3 = chi2_contingency(tabela_mca_3)

print("Adaptivity x Financial")
print(f"estatística qui²: {round(tab_3[0], 2)}")
print(f"p-valor da estatística: {round(tab_3[1], 4)}")
print(f"graus de liberdade: {tab_3[2]}")

tab_4 = chi2_contingency(tabela_mca_4)

print("Adaptivity x Internet")
print(f"estatística qui²: {round(tab_4[0], 2)}")
print(f"p-valor da estatística: {round(tab_4[1], 4)}")
print(f"graus de liberdade: {tab_4[2]}")

#%% Elaborando a MCA

mca = prince.MCA(n_components=3).fit(dados_mca)

# Vamos parametrizar a MCA para três dimensões
# O objetivo é criar um mapa perceptual 3D

#%% Quantidade total de dimensões

# Quantidade de dimensões = qtde total de categorias - qtde de variáveis

# Quantidade total de categorias
mca.J_

# Quantidade de variáveis na análise
mca.K_

# Quantidade de dimensões
quant_dim = mca.J_ - mca.K_

# Resumo das informações
print(f"quantidade total de categorias: {mca.J_}")
print(f"quantidade de variáveis: {mca.K_}")
print(f"quantidade de dimensões: {quant_dim}")

#%% Obtendo os eigenvalues

tabela_autovalores = mca.eigenvalues_summary

print(tabela_autovalores)

#%% Inércia principal total

# Soma de todos os autovalores (todas as dimensões existentes)

print(mca.total_inertia_)

#%% Média da inércia principal total por dimensão

# É interessante plotar apenas dimensões com autovalores maiores do que a média

print(mca.total_inertia_/quant_dim)

# Neste caso, as 3 dimensões extraídas têm autovalores > 0.199

#%% Obtendo as coordenadas principais das categorias das variáveis

coord_burt = mca.column_coordinates(dados_mca)

print(coord_burt)

#%% Obtendo as coordenadas-padrão das categorias das variáveis

coord_padrao = mca.column_coordinates(dados_mca)/np.sqrt(mca.eigenvalues_)

print(coord_padrao)

#%% Obtendo as coordenadas das observações do banco de dados

# Na função, as coordenadas das observações vêm das coordenadas-padrão

coord_obs = mca.row_coordinates(dados_mca)

print(coord_obs)

#%% Plotando o mapa perceptual (coordenadas-padrão)

# Primeiro passo: gerar um DataFrame detalhado

chart = coord_padrao.reset_index()

var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])

nome_categ=[]
for col in dados_mca:
    nome_categ.append(dados_mca[col].sort_values(ascending=True).unique())
    categorias = pd.DataFrame(nome_categ).stack().reset_index()

chart_df_mca = pd.DataFrame({'categoria': chart['index'],
                             'obs_x': chart[0],
                             'obs_y': chart[1],
                             'obs_z': chart[2],
                             'variavel': var_chart,
                             'categoria_id': categorias[0]})

# Segundo passo: gerar o gráfico de pontos

fig = px.scatter_3d(chart_df_mca, 
                    x='obs_x', 
                    y='obs_y', 
                    z='obs_z',
                    color='variavel',
                    text=chart_df_mca.categoria_id)

fig.write_html('assoc_mca_adapta.html')

#%% FIM!