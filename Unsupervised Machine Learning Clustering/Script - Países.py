# -*- coding: utf-8 -*-

# Análise de Cluster
# MBA em Data Science e Analytics USP ESALQ

# Prof. Dr. Wilson Tarantin Junior

#%% Instalando os pacotes

## Executar na linha de comando do console (sem o #)

# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn
# pip install plotly
# pip install scipy
# pip install scikit-learn
# pip install pingouin

#%% Importando os pacotes

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.stats import zscore
from scipy.spatial.distance import pdist
from sklearn.cluster import AgglomerativeClustering
import pingouin as pg
import plotly.express as px
import plotly.io as pio
pio.renderers.default='browser'

#%% Importando o banco de dados

dados_paises = pd.read_csv('dados_paises.csv')
## Fonte: https://www.kaggle.com/datasets/rohan0301/unsupervised-learning-on-country-data

#%% Visualizando informações sobre os dados e variáveis

# Estatísticas descritivas das variáveis

tab_desc = dados_paises.describe()

# Matriz de correlações das variáveis

# Vamos remover a coluna "country", pois é apenas um id
paises = dados_paises.drop(columns=['country'])

# Gerando a matriz de correlações de Pearson
matriz_corr = pg.rcorr(paises, method = 'pearson', upper = 'pval', 
                       decimals = 4, 
                       pval_stars = {0.01: '***', 0.05: '**', 0.10: '*'})

#%% Mapa de calor indicando a correlação entre os atributos

# Matriz de correlações básica
corr = paises.corr()

f, ax = plt.subplots(figsize=(11, 9), dpi=600)

sns.heatmap(corr, 
            cmap = plt.cm.coolwarm,
            vmax=1, 
            vmin = -1,
            center=0,
            square=True, 
            linewidths=.5,
            annot = True,
            fmt='.2f', 
            annot_kws={'size': 12},
            cbar_kws={"shrink": 0.50})

plt.title('Matriz de Correlações')
plt.tight_layout()
ax.tick_params(axis = 'x', labelsize = 11)
ax.tick_params(axis = 'y', labelsize = 11)
ax.set_ylim(len(corr))

plt.show()

#%% Padronização das variáveis

# Aplicando o procedimento de ZScore em todas as variáveis
paises_pad = paises.apply(zscore, ddof=1)

# As variáveis passam a ter média = 0 e desvio padrão = 1
print(np.round(paises_pad.mean(), 5))
print(np.round(paises_pad.std(), 5))

#%% Cluster hierárquico aglomerativo: distância euclidiana + single linkage

# Visualizando as distâncias
dist_euclidiana = pdist(paises_pad, metric='euclidean')

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

# Gerando o dendrograma
## Distância euclidiana e método de encadeamento single linkage

plt.figure(figsize=(16,8), dpi=600)
dend_sing = sch.linkage(paises_pad, method = 'single', metric = 'euclidean')
dendrogram_s = sch.dendrogram(dend_sing)
plt.title('Dendrograma Single Linkage', fontsize=16)
plt.xlabel('Países', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

# Opções para o método de encadeamento ("method"):
    ## single
    ## complete
    ## average

# Opções para as distâncias ("metric"):
    ## euclidean
    ## sqeuclidean
    ## cityblock
    ## chebyshev
    ## canberra
    ## correlation

#%% Cluster hierárquico aglomerativo: distância euclidiana + average linkage

# Gerando o dendrograma
## Distância euclidiana e método de encadeamento average linkage

plt.figure(figsize=(16,8), dpi=600)
dend_avg = sch.linkage(paises_pad, method = 'average', metric = 'euclidean')
dendrogram_a = sch.dendrogram(dend_avg)
plt.title('Dendrograma Average Linkage', fontsize=16)
plt.xlabel('Países', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.show()

#%% Cluster hierárquico aglomerativo: distância euclidiana + complete linkage

# Gerando o dendrograma
## Distância euclidiana e método de encadeamento complete linkage

plt.figure(figsize=(16,8), dpi=600)
dend_compl = sch.linkage(paises_pad, method = 'complete', metric = 'euclidean')
dendrogram_c = sch.dendrogram(dend_compl, color_threshold = 8)
plt.title('Dendrograma Complete Linkage', fontsize=16)
plt.xlabel('Países', fontsize=16)
plt.ylabel('Distância Euclidiana', fontsize=16)
plt.axhline(y = 8, color = 'red', linestyle = '--')
plt.show()

# Gerando a variável com a indicação do cluster no dataset

cluster_comp = AgglomerativeClustering(n_clusters = 5, metric = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(paises_pad)
dados_paises['cluster_complete'] = indica_cluster_comp
paises_pad['cluster_complete'] = indica_cluster_comp
dados_paises['cluster_complete'] = dados_paises['cluster_complete'].astype('category')
paises_pad['cluster_complete'] = paises_pad['cluster_complete'].astype('category')

#%% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_complete MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_complete MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# child_mort
pg.anova(dv='child_mort', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# exports
pg.anova(dv='exports', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# imports
pg.anova(dv='imports', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# health
pg.anova(dv='health', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# income
pg.anova(dv='income', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# inflation
pg.anova(dv='inflation', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# life_expec
pg.anova(dv='life_expec', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# total_fer
pg.anova(dv='total_fer', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

# gdpp
pg.anova(dv='gdpp', 
         between='cluster_complete', 
         data=paises_pad,
         detailed=True).T

## A variável mais discriminante contém a maior estatística F (e significativa)
## O valor da estatística F é sensível ao tamanho da amostra

#%% Gráfico 3D dos clusters

# Perspectiva 1
fig1 = px.scatter_3d(dados_paises, 
                     x='total_fer', 
                     y='income', 
                     z='life_expec',
                     color='cluster_complete')

fig1.write_html('paises_1.html')

# Perspectiva 2
fig2 = px.scatter_3d(dados_paises, 
                     x='gdpp', 
                     y='income', 
                     z='life_expec',
                     color='cluster_complete')

fig2.write_html('paises_2.html')

#%% Identificação das características dos clusters

# Agrupando o banco de dados

analise_paises = dados_paises.drop(columns=['country']).groupby(by=['cluster_complete'], observed=True)

# Estatísticas descritivas por grupo

tab_medias_grupo = analise_paises.mean().T
tab_desc_grupo = analise_paises.describe().T

#%% FIM