# -*- coding: utf-8 -*-

# Exercícios de Modelos Não Supervisionados de Machine Learning
# MBA em Data Science e Analytics (USP/ESALQ)

# Prof. Dr. Wilson Tarantin Junior

#%% Instalando os pacotes

# Executar diretamente na linha de comando no console

# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install plotly
# pip install scipy
# pip install scikit-learn
# pip install pingouin
# pip install prince

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import zscore
from sklearn.cluster import KMeans
import pingouin as pg
import plotly.express as px
import prince
from scipy.stats import chi2_contingency
from sklearn.metrics import silhouette_score

#%% Importando o banco de dados

clientes = pd.read_excel('clientes_segmenta.xlsx')
## Fonte: adaptado de https://www.kaggle.com/datasets/kaushiksuresh147/customer-segmentation
 
#%% Estatísticas gerais do banco de dados

clientes.info()
## Note que há valores faltantes "nan"

#%% Removendo as observações com valores faltantes

clientes.dropna(inplace=True)

# Novas informações sobre o dataset
clientes.info()

#%% Vamos iniciar realizando a clusterização nas variáveis quantitativas

# Separando somente as variáveis quantitativas do banco de dados
df_quanti = clientes[['Age', 'FamilySize']]

# Estatísticas descritivas das variáveis
df_quanti.describe()

#%% Realizando a padronização por meio do Z-Score

# As variáveis estão em unidades de medidas distintas
df_quanti_pad = df_quanti.apply(zscore, ddof=1)

#%% Identificação da quantidade de clusters (Método Elbow)

elbow = []
K = range(1,11) # ponto de parada pode ser parametrizado manualmente
for k in K:
    kmeanElbow = KMeans(n_clusters=k, init='random', random_state=100).fit(df_quanti_pad)
    elbow.append(kmeanElbow.inertia_)
    
plt.figure(figsize=(16,8), dpi=600)
plt.figure(figsize=(16,8))
plt.plot(K, elbow, marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.xticks(range(1,11)) # ajustar de acordo com o range de K definido acima
plt.ylabel('WCSS', fontsize=16)
plt.title('Método de Elbow', fontsize=16)
plt.show()

#%% Identificação da quantidade de clusters (Método da Silhueta)

silhueta = []
I = range(2,11) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(df_quanti_pad)
    silhueta.append(silhouette_score(df_quanti_pad, kmeansSil.labels_))

plt.figure(figsize=(16,8))
plt.plot(range(2, 11), silhueta, color = 'purple', marker='o')
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

#%% Cluster K-means

# Vamos considerar 5 clusters, dadas as evidências anteriores!
kmeans_final = KMeans(n_clusters = 5, init = 'random', random_state=100).fit(df_quanti_pad)

# Gerando a variável para identificarmos os clusters gerados
kmeans_clusters = kmeans_final.labels_
clientes['Cluster'] = kmeans_clusters
df_quanti_pad['Cluster'] = kmeans_clusters
clientes['Cluster'] = clientes['Cluster'].astype('category')
df_quanti_pad['Cluster'] = df_quanti_pad['Cluster'].astype('category')

#%% ANOVA

# Analisando se as duas variáveis são significativas para a clusterização 

# Age
pg.anova(dv='Age', 
         between='Cluster', 
         data=df_quanti_pad,
         detailed=True).T

# Family Size
pg.anova(dv='FamilySize', 
         between='Cluster', 
         data=df_quanti_pad,
         detailed=True).T

#%% Quais são as características dos clusters em termos de idade e família

clientes[['Age', 'FamilySize', 'Cluster']].groupby(by=['Cluster']).mean()

#%% Vamos realizar uma ACM nas variáveis qualitativas (incluir os clusters!)

# Separando somente as variáveis categóricas do banco de dados
df_quali = clientes[['Gender', 'EverMarried', 'Graduated', 'SpendingScore', 'Cluster']]

#%% Estatísticas descritivas univariadas

# Tabelas de frequências por variável
print(df_quali.Gender.value_counts())
print(df_quali.EverMarried.value_counts())
print(df_quali.Graduated.value_counts())
print(df_quali.SpendingScore.value_counts())
print(df_quali.Cluster.value_counts())

#%% Testes qui-quadrado para pares de variáveis

# Vamos colocar como referência 'SpendingScore'

tabela1 = chi2_contingency(pd.crosstab(df_quali["SpendingScore"],
                                       df_quali["Gender"]))
print(f"p-valor da estatística: {round(tabela1[1], 4)}")

tabela2 = chi2_contingency(pd.crosstab(df_quali["SpendingScore"], 
                                       df_quali["EverMarried"]))
print(f"p-valor da estatística: {round(tabela2[1], 4)}")

tabela3 = chi2_contingency(pd.crosstab(df_quali["SpendingScore"], 
                                       df_quali["Graduated"]))
print(f"p-valor da estatística: {round(tabela3[1], 4)}")

tabela4 = chi2_contingency(pd.crosstab(df_quali["SpendingScore"], 
                                       df_quali["Cluster"]))
print(f"p-valor da estatística: {round(tabela4[1], 4)}")

# Todas apresentam associação significativa com pelo menos uma variável

#%% Elaborando a análise de correspondência múltipla

# Criando coordenadas para 3 dimensões (a seguir, verifica-se a viabilidade)
mca = prince.MCA(n_components=3).fit(df_quali)

#%% Analisando os resultados

# Análise dos autovalores
tabela_autovalores = mca.eigenvalues_summary
print(tabela_autovalores)

# Inércia total da análise
print(mca.total_inertia_)

# Plotar apenas dimensões com inércia parcial superior à inércia total média
quant_dim = mca.J_ - mca.K_
print(mca.total_inertia_/quant_dim)

#%% Obtendo as coordenadas-padrão das categorias das variáveis

coord_padrao = mca.column_coordinates(df_quali)/np.sqrt(mca.eigenvalues_)
print(coord_padrao)

#%% Plotando o mapa perceptual (coordenadas-padrão)

# Primeiro passo: gerar um DataFrame detalhado

chart = coord_padrao.reset_index()
var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])

nome_categ=[]
for col in df_quali:
    nome_categ.append(df_quali[col].sort_values(ascending=True).unique())
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

fig.write_html('segmenta.html')

#%% Fim!