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
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% Importando o banco de dados

# Objetivo: agrupar os clientes de uma operadora de cartão de crédito
# Analisar os grupos de clientes mais e menos leais à marca (por meio do uso)

dados_cartao = pd.read_csv('cartao_credito.csv')
## Fonte: https://www.kaggle.com/datasets/aryashah2k/credit-card-customer-data

#%% Visualizando informações sobre os dados e variáveis

# Estrutura do banco de dados

print(dados_cartao.info())

#%% Estatísticas descritivas das variáveis

# Primeiramente, vamos excluir as variáveis que não serão utilizadas

cartao_cluster = dados_cartao.drop(columns=['Sl_No', 'Customer Key'])

# Obtendo as estatísticas descritivas das variáveis

tab_descritivas = cartao_cluster.describe().T
# Vamos padronizar as variáveis antes da clusterização!

#%% Padronização por meio do Z-Score

# Aplicando o procedimento de ZScore
cartao_pad = cartao_cluster.apply(zscore, ddof=1)

# Visualizando o resultado do procedimento na média e desvio padrão
print(np.round(cartao_pad.mean(), 3))
print(np.round(cartao_pad.std(), 3))

#%% Gráfico 3D das observações

fig = px.scatter_3d(cartao_pad, 
                    x='Avg_Credit_Limit', 
                    y='Total_Credit_Cards', 
                    z='Total_visits_bank')

fig.write_html('cartao_inicial.html')

#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(cartao_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8), dpi=600)
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar range de acordo com K acima
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(cartao_pad)
    silhueta.append(silhouette_score(cartao_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8), dpi=600)
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o') # ajustar range
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster Não Hierárquico K-means

# Vamos considerar 3 clusters, considerando as evidências anteriores!

kmeans_final = KMeans(n_clusters = 3, init = 'random', random_state=100).fit(cartao_pad)

# Gerando a variável para identificarmos os clusters gerados

kmeans_clusters = kmeans_final.labels_
cartao_cluster['cluster_kmeans'] = kmeans_clusters
cartao_pad['cluster_kmeans'] = kmeans_clusters
cartao_cluster['cluster_kmeans'] = cartao_cluster['cluster_kmeans'].astype('category')
cartao_pad['cluster_kmeans'] = cartao_pad['cluster_kmeans'].astype('category')

#%% Análise de variância de um fator (ANOVA)

# Interpretação do output:

## cluster_kmeans MS: indica a variabilidade entre grupos
## Within MS: indica a variabilidade dentro dos grupos
## F: estatística de teste (cluster_kmeans MS / Within MS)
## p-unc: p-valor da estatística F
## se p-valor < 0.05: pelo menos um cluster apresenta média estatisticamente diferente dos demais

# Avg_Credit_Limit
pg.anova(dv='Avg_Credit_Limit', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_Credit_Cards
pg.anova(dv='Total_Credit_Cards', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_visits_bank
pg.anova(dv='Total_visits_bank', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_visits_online
pg.anova(dv='Total_visits_online', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

# Total_calls_made
pg.anova(dv='Total_calls_made', 
         between='cluster_kmeans', 
         data=cartao_pad,
         detailed=True).T

#%% Gráfico 3D dos clusters

# Perspectiva 1

fig1 = px.scatter_3d(cartao_cluster, 
                     x='Avg_Credit_Limit', 
                     y='Total_Credit_Cards', 
                     z='Total_visits_online',
                     color='cluster_kmeans')

fig1.write_html('cartao_1.html')

# Perspectiva 2

fig2 = px.scatter_3d(cartao_cluster, 
                     x='Avg_Credit_Limit', 
                     y='Total_Credit_Cards', 
                     z='Total_visits_bank',
                     color='cluster_kmeans')

fig2.write_html('cartao_2.html')

# Perspectiva 3

fig3 = px.scatter_3d(cartao_cluster, 
                     x='Avg_Credit_Limit', 
                     y='Total_Credit_Cards', 
                     z='Total_calls_made',
                     color='cluster_kmeans')

fig3.write_html('cartao_3.html')

#%% Identificação das características dos clusters

# Agrupando o banco de dados

cartao_grupo = cartao_cluster.groupby(by=['cluster_kmeans'], observed=True)

# Estatísticas descritivas por grupo

tab_media_grupo = cartao_grupo.mean().T

#%% FIM