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
import seaborn as sns
import matplotlib.pyplot as plt
import prince

#%% Análise de Correspondência Múltipla (MCA)

# Importando o banco de dados

perfil_mca = pd.read_excel("perfil_aplicacao_civil.xlsx")
# Fonte: Fávero e Belfiore (2024, Capítulo 11)

#%% Selecionando apenas as variáveis que farão parte da análise

dados_mca = perfil_mca.drop(columns=['estudante'])

#%% Informações descritivas sobre as variáveis

print(dados_mca['perfil'].value_counts())
print(dados_mca['aplicacao'].value_counts())
print(dados_mca['estado.civil'].value_counts())

#%% Analisando as tabelas de contingência

tabela_mca_1 = pd.crosstab(dados_mca["perfil"], dados_mca["aplicacao"])
tabela_mca_2 = pd.crosstab(dados_mca["perfil"], dados_mca["estado.civil"])
tabela_mca_3 = pd.crosstab(dados_mca["aplicacao"], dados_mca["estado.civil"])

print(tabela_mca_1)
print(tabela_mca_2)
print(tabela_mca_3)

#%% Analisando a significância estatística das associações (teste qui²)

tab_1 = chi2_contingency(tabela_mca_1)

print("Associação Perfil x Aplicação")
print(f"estatística qui²: {round(tab_1[0], 2)}")
print(f"p-valor da estatística: {round(tab_1[1], 4)}")
print(f"graus de liberdade: {tab_1[2]}")

tab_2 = chi2_contingency(tabela_mca_2)

print("Associação Perfil x Estado Civil")
print(f"estatística qui²: {round(tab_2[0], 2)}")
print(f"p-valor da estatística: {round(tab_2[1], 4)}")
print(f"graus de liberdade: {tab_2[2]}")

tab_3 = chi2_contingency(tabela_mca_3)

print("Associação Aplicação x Estado Civil")
print(f"estatística qui²: {round(tab_3[0], 2)}")
print(f"p-valor da estatística: {round(tab_3[1], 4)}")
print(f"graus de liberdade: {tab_3[2]}")

#%% Elaborando a MCA

mca = prince.MCA(n_components=2).fit(dados_mca)

# Vamos parametrizar a MCA para duas dimensões (eixos X e Y no mapa perceptual)
# Note que o input é o próprio banco de dados com as variáveis categóricas

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

#%% Visualizando as matrizes: binária e Burt

# Nota: esta célula não é requerida para a função, tem fins didáticos!

binaria = pd.get_dummies(dados_mca, columns=dados_mca.columns, dtype=float)

burt = np.matmul(np.transpose(binaria), binaria)

#%% Obtendo os eigenvalues

tabela_autovalores = mca.eigenvalues_summary

print(tabela_autovalores)

#%% Inércia principal total

# Soma de todos os autovalores (todas as dimensões existentes)

print(mca.total_inertia_)

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

chart = coord_padrao.reset_index()

var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])
# Nota: para a função acima ser executada adequadamente, não deixar underline no nome original da variável no dataset!

chart_df_mca = pd.DataFrame({'categoria': chart['index'],
                             'obs_x': chart[0],
                             'obs_y': chart[1],
                             'variavel': var_chart})

plt.figure(dpi=600)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.03, point['y'] - 0.02, point['val'], fontsize=5)

label_point(x = chart_df_mca['obs_x'],
            y = chart_df_mca['obs_y'],
            val = chart_df_mca['categoria'],
            ax = plt.gca())

sns.scatterplot(data=chart_df_mca, x='obs_x', y='obs_y', hue='variavel', s=20)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.axhline(y=0, color='lightgrey', ls='--', linewidth=0.8)
plt.axvline(x=0, color='lightgrey', ls='--', linewidth=0.8)
plt.tick_params(size=2, labelsize=6)
plt.legend(bbox_to_anchor=(0,0), fancybox=True, shadow=True, fontsize = '6')
plt.title("Mapa Perceptual - MCA", fontsize=12)
plt.xlabel(f"Dim. 1: {tabela_autovalores.iloc[0,1]} da inércia", fontsize=8)
plt.ylabel(f"Dim. 2: {tabela_autovalores.iloc[1,1]} da inércia", fontsize=8)
plt.show()

#%% FIM!