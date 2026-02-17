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
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
import pingouin as pg
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import seaborn as sns
import sympy as sy
import scipy as sp
import warnings
warnings.filterwarnings("ignore")

#%% Importando o banco de dados

notas = pd.read_excel("notas_fatorial.xlsx")
# Fonte: Fávero e Belfiore (2024, Capítulo 10)

#%% Informações sobre as variáveis

# Informações gerais sobre o DataFrame

print(notas.info())

# Estatísticas descritiva das variáveis

print(notas.describe())

#%% Separando somente as variáveis quantitativas do banco de dados

notas_pca = notas[["finanças", "custos", "marketing", "atuária"]]

#%% Matriz de correlações de Pearson entre as variáveis

pg.rcorr(notas_pca, method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Outra maneira de analisar as informações sobre correlações

# Matriz de correlações em um objeto "simples"

corr = notas_pca.corr()

# Plotando em um gráfico heatmap

plt.figure(figsize=(12,8), dpi=600)
sns.heatmap(corr, 
            cmap=plt.cm.coolwarm,
            vmax=1, 
            vmin=-1,
            center=0,
            square=True, 
            linewidths=.5,
            annot=True,
            fmt='.2f', 
            annot_kws={'size':12},
            cbar_kws={"shrink":0.50})

plt.title('Matriz de Correlações', fontsize=14)
plt.tight_layout()
plt.tick_params(labelsize=10)
plt.show()

#%% Teste de Esfericidade de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(notas_pca)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Definindo a PCA (procedimento inicial com todos os fatores possíveis)

fa = FactorAnalyzer(n_factors=4, method='principal', rotation=None).fit(notas_pca)

#%% Obtendo os eigenvalues (autovalores): resultantes da função FactorAnalyzer

autovalores = fa.get_eigenvalues()[0]

print(autovalores) # Temos 4 autovalores, pois são 4 variáveis ao todo

# Soma dos autovalores

round(autovalores.sum(), 2)

#%% Obtendo os autovalores e autovetores: ilustrando o fundamento

## Atenção: esta célula tem fins didáticos, não é requerida na FactorAnalyzer

# Parametrizando o pacote

lamda = sy.symbols('lamda')
sy.init_printing(scale=0.8)

# Especificando a matriz de correlações

matriz = sy.Matrix(corr)
polinomio = matriz.charpoly(lamda)

polinomio

# Obtendo as raízes do polinômio característico: são os autovalores

autovalores, autovetores = sp.linalg.eigh(corr)
autovalores = autovalores[::-1]

# Obtendo os autovetores para cada autovalor extraído

autovetores = autovetores[:, ::-1]

#%% Autovalores, variâncias e variâncias acumuladas

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Gráfico da variância acumulada dos componentes principais

plt.figure(figsize=(12,8), dpi=600)
ax = sns.barplot(x=tabela_eigen.index, y='Variância', hue=tabela_eigen.index, palette='rocket', data=tabela_eigen)
for container in ax.containers:
    labels = [f"{v*100:.2f}%" for v in container.datavalues]
    ax.bar_label(container, labels=labels)
ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, pos: f"{x*100:.0f}%"))
plt.title("Fatores Extraídos", fontsize=16)
plt.xlabel(f"{tabela_eigen.shape[0]} fatores que explicam {round(tabela_eigen['Variância'].sum()*100,2)}% da variância", fontsize=12)
plt.ylabel("Variância explicada", fontsize=12)
plt.show()

#%% Determinando as cargas fatoriais

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = notas_pca.columns

print(tabela_cargas)

#%% Gráfico das cargas fatoriais (loading plot)

plt.figure(figsize=(12,8), dpi=600)
tabela_cargas_chart = tabela_cargas.reset_index()
plt.scatter(tabela_cargas_chart['Fator 1'], tabela_cargas_chart['Fator 2'], s=50, color='red')

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.05, point['y'], point['val'])

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
tabela_comunalidades.index = notas_pca.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

fatores = pd.DataFrame(fa.transform(notas_pca))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados

notas = pd.concat([notas.reset_index(drop=True), fatores], axis=1)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = notas_pca.columns

print(tabela_scores)

#%% Correlação entre os fatores extraídos

# A seguir, verifica-se que a correlação entre os fatores é zero (ortogonais)

pg.rcorr(notas[['Fator 1','Fator 2', 'Fator 3', 'Fator 4']],
         method = 'pearson', upper = 'pval', 
         decimals = 4, 
         pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Critério de Kaiser (raiz latente)

# Verificar os autovalores com valores maiores que 1
# Existem dois componentes maiores do que 1

#%% Parametrizando a PCA para dois fatores (autovalores > 1)

fa = FactorAnalyzer(n_factors=2, method='principal', rotation=None).fit(notas_pca)

#%% Autovalores, variâncias e variâncias acumuladas de 2 fatores

# Note que não há alterações nos valores, apenas ocorre a seleção dos fatores

autovalores_fatores = fa.get_factor_variance()

tabela_eigen = pd.DataFrame(autovalores_fatores)
tabela_eigen.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_eigen.columns)]
tabela_eigen.index = ['Autovalor','Variância', 'Variância Acumulada']
tabela_eigen = tabela_eigen.T

print(tabela_eigen)

#%% Determinando as cargas fatoriais

# Note que não há alterações nas cargas fatoriais nos 2 fatores!

cargas_fatoriais = fa.loadings_

tabela_cargas = pd.DataFrame(cargas_fatoriais)
tabela_cargas.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_cargas.columns)]
tabela_cargas.index = notas_pca.columns

print(tabela_cargas)

#%% Determinando as novas comunalidades

# As comunalidades são alteradas, pois há fatores retirados da análise!

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = notas_pca.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

# Vamos remover os fatores obtidos anteriormente

notas = notas.drop(columns=['Fator 1', 'Fator 2', 'Fator 3', 'Fator 4'])

#  Vamos gerar novamente, agora para os 2 fatores extraídos

fatores = pd.DataFrame(fa.transform(notas_pca))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados

notas = pd.concat([notas.reset_index(drop=True), fatores], axis=1)

# Note que são os mesmos, apenas ocorre a seleção dos 2 primeiros fatores!

#%% Identificando os scores fatoriais

# Não há mudanças nos scores fatoriais!

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = notas_pca.columns

print(tabela_scores)

#%% Criando um ranking (soma ponderada e ordenamento)

# O ranking irá considerar apenas os 2 fatores com autovalores > 1
# A base de seleção é a tabela_eigen

notas['Ranking'] = 0

for index, item in enumerate(list(tabela_eigen.index)):
    variancia = tabela_eigen.loc[item]['Variância']

    notas['Ranking'] = notas['Ranking'] + notas[tabela_eigen.index[index]]*variancia
    
print(notas)

#%% Fim!