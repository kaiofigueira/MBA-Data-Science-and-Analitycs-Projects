# -*- coding: utf-8 -*-

# Análise Fatorial PCA
# MBA em Data Science e Analytics USP ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Instalando os pacotes

## Executar na linha de comando do console (sem o #)

# pip install pandas
# pip install numpy
# pip install factor_analyzer
# pip install sympy
# pip install scipy
# pip install matplotlib
# pip install seaborn
# pip install plotly
# pip install pingouin
# pip install pyshp

#%% Importando os pacotes necessários

import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

#%% Importando o banco de dados

emprestimo = pd.read_excel("emprestimo_banco.xlsx")
# Fonte: adaptado de https://www.kaggle.com/datasets/itsmesunil/bank-loan-modelling

#%% Separando somente as variáveis para a análise fatorial PCA

emprestimo_pca = emprestimo.drop(columns=['ID'])

#%% Estatísticas descritivas das variáveis

tab_desc = emprestimo_pca.describe()

#%% Matriz de correlações de Pearson

# Matriz de correlações em um objeto "simples"

corr = emprestimo_pca.corr()

# Gráfico interativo

fig = go.Figure()

fig.add_trace(
    go.Heatmap(
        x = corr.columns,
        y = corr.index,
        z = np.array(corr),
        text = corr.values,
        texttemplate = '%{text:.2f}',
        colorscale = 'Viridis'))

fig.update_layout(
    height = 600,
    width = 600,
    yaxis = dict(autorange='reversed'),
    title = {'text':'Matriz de Correlações', 'x':0.50})

fig.write_html('correl_emprestimo.html')

#%% Teste de Esfericidade de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(emprestimo_pca)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Definindo a PCA (procedimento inicial com todos os fatores possíveis)

fa = FactorAnalyzer(n_factors=6, method='principal', rotation=None).fit(emprestimo_pca)

#%% Obtendo os eigenvalues (autovalores): resultantes da função FactorAnalyzer

autovalores = fa.get_eigenvalues()[0]

print(autovalores)

#%% Redefinindo a PCA pelo critério da raiz latente

# Há dois fatores derivados de autovalores > 1

fa = FactorAnalyzer(n_factors=2, method='principal', rotation=None).fit(emprestimo_pca)

#%% Eigenvalues, variâncias e variâncias acumuladas

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Determinando as cargas fatoriais

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = emprestimo_pca.columns

print(tabela_cargas)

#%% Gráfico das cargas fatoriais (loading plot)

plt.figure(figsize=(12,8), dpi=600)
tabela_cargas_chart = tabela_cargas.reset_index()
plt.scatter(tabela_cargas_chart['Fator 1'], tabela_cargas_chart['Fator 2'], s=30, color='red')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.02, point['y'], point['val'], fontsize=8)

label_point(x = tabela_cargas_chart['Fator 1'],
            y = tabela_cargas_chart['Fator 2'],
            val = tabela_cargas_chart['index'],
            ax = plt.gca()) 

plt.axhline(y=0, color='grey', ls='--')
plt.axvline(x=0, color='grey', ls='--')
plt.ylim([-1.1,1.1])
plt.xlim([-1.1,1.1])
plt.title("Loading Plot", fontsize=16)
plt.xlabel(f"Fator 1: {round(tabela_eigen.iloc[0]['Variância']*100,2)}% de variância explicada", fontsize=12)
plt.ylabel(f"Fator 2: {round(tabela_eigen.iloc[1]['Variância']*100,2)}% de variância explicada", fontsize=12)
plt.show()

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = emprestimo_pca.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

fatores = pd.DataFrame(fa.transform(emprestimo_pca))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = emprestimo_pca.columns

print(tabela_scores)

#%% Em certos casos, a "rotação de fatores" pode melhorar a interpretação

# Analisando pelo loading plot, aplica-se a rotação dos eixos na origem (0,0)
# O método mais comum é a 'varimax', que é a rotação ortogonal dos fatores
# O objetivo é aumentar a carga fatorial em um fator e diminuir em outro
# Em resumo, trata-se de uma redistribuição de cargas fatoriais

#%% Adicionando a rotação: rotation='varimax'

# Aplicando a rotação aos 2 fatores extraídos

fa = FactorAnalyzer(n_factors=2, method='principal', rotation='varimax').fit(emprestimo_pca)

#%% Eigenvalues, variâncias e variâncias acumuladas

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

## Há a redistribuição da variância entre os fatores (mas o total é o mesmo!)

#%% Determinando as cargas fatoriais

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = emprestimo_pca.columns

print(tabela_cargas)

## As cargas são alteradas: cargas fatoriais rotacionadas

#%% Gráfico das cargas fatoriais (loading plot)

plt.figure(figsize=(12,8), dpi=600)
tabela_cargas_chart = tabela_cargas.reset_index()
plt.scatter(tabela_cargas_chart['Fator 1'], tabela_cargas_chart['Fator 2'], s=30, color='blue')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.02, point['y'], point['val'], fontsize=8)

label_point(x = tabela_cargas_chart['Fator 1'],
            y = tabela_cargas_chart['Fator 2'],
            val = tabela_cargas_chart['index'],
            ax = plt.gca()) 

plt.axhline(y=0, color='grey', ls='--')
plt.axvline(x=0, color='grey', ls='--')
plt.ylim([-1.1,1.1])
plt.xlim([-1.1,1.1])
plt.title("Loading Plot - Fatores Rotacionados", fontsize=16)
plt.xlabel(f"Fator 1: {round(tabela_eigen.iloc[0]['Variância']*100,2)}% de variância explicada", fontsize=12)
plt.ylabel(f"Fator 2: {round(tabela_eigen.iloc[1]['Variância']*100,2)}% de variância explicada", fontsize=12)
plt.show()

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = emprestimo_pca.columns

print(tabela_comunalidades)

## As comunalidades não mudam!

#%% Extração dos fatores para as observações do banco de dados

fatores = pd.DataFrame(fa.transform(emprestimo_pca))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados

emprestimo = pd.concat([emprestimo.reset_index(drop=True), fatores], axis=1)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = emprestimo_pca.columns

print(tabela_scores)

## Os scores são alterados, o que geram fatores rotacionados!

#%% Interpretando os scores fatoriais em cada fator extraído

tabela_scores_graph = tabela_scores.reset_index()
tabela_scores_graph = tabela_scores_graph.melt(id_vars='index')

plt.figure(dpi=600)
sns.barplot(data=tabela_scores_graph, x='variable', y='value', hue='index', palette='viridis')
plt.legend(title='Variáveis', bbox_to_anchor=(1,1), fontsize = '6')
plt.title('Scores Fatoriais', fontsize='12')
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.show()

#%% Os fatores continuam ortogonais

pg.rcorr(emprestimo[['Fator 1', 'Fator 2']], 
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Fim!