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
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.cluster.hierarchy as sch
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import pingouin as pg
import plotly.express as px 
import plotly.io as pio
pio.renderers.default='browser'

#%% Importando o banco de dados

varejista = pd.read_excel('regional_varejista.xlsx')
## Fonte: Fávero & Belfiore (2024, Capítulo 9)

#%% Visualizando os dados

print(varejista.info())

# Estatísticas descritivas das variáveis

print(varejista[['atendimento','sortimento', 'organização']].describe())
## Neste caso, não faremos a padronização. As variáveis estão na mesma escala!

#%% Ajustando o banco de dados

# Retirando todos os dados que não são numéricos do dataset

varejo = varejista.drop(columns=['loja','regional'])

#%% Cluster Hierárquico Aglomerativo: single linkage + distância cityblock

# Gerando o dendrograma

plt.figure(figsize=(16,8), dpi=600)
dend_sing = sch.linkage(varejo, method = 'single', metric = 'cityblock')
dendrogram_s = sch.dendrogram(dend_sing, color_threshold = 60, labels = list(varejista.loja))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Lojas', fontsize=16)
plt.ylabel('Distância Cityblock (Manhattan)', fontsize=16)
plt.axhline(y = 60, color = 'red', linestyle = '--')
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

# Criando a variável que indica os clusters no banco de dados

cluster_sing = AgglomerativeClustering(n_clusters = 3, metric = 'cityblock', linkage = 'single')
indica_cluster_sing = cluster_sing.fit_predict(varejo)
varejista['cluster_single'] = indica_cluster_sing
varejista['cluster_single'] = varejista['cluster_single'].astype('category')

#%% Cluster Hierárquico Aglomerativo: complete linkage + distância euclidiana

# Gerando o dendrograma

plt.figure(figsize=(16,8), dpi=600)
dend_sing_euc = sch.linkage(varejo, method = 'complete', metric = 'euclidean')
dendrogram_euc = sch.dendrogram(dend_sing_euc, color_threshold = 55, labels = list(varejista.loja))
plt.title('Dendrograma', fontsize=16)
plt.xlabel('Lojas', fontsize=16)
plt.ylabel('Distância Euclideana', fontsize=16)
plt.axhline(y = 55, color = 'red', linestyle = '--')
plt.show()

# Criando a variável que indica os clusters no banco de dados

cluster_comp = AgglomerativeClustering(n_clusters = 3, metric = 'euclidean', linkage = 'complete')
indica_cluster_comp = cluster_comp.fit_predict(varejo)
varejista['cluster_complete'] = indica_cluster_comp
varejista['cluster_complete'] = varejista['cluster_complete'].astype('category')

#%% Plotando as observações e seus clusters (single + cityblock)

plt.figure(figsize=(10,10), dpi=600)
fig = sns.scatterplot(x='atendimento', y='sortimento', s=60, data=varejista, hue='cluster_single')
plt.title('Clusters', fontsize=16)
plt.xlabel('Atendimento', fontsize=16)
plt.ylabel('Sortimento', fontsize=16)
plt.show()

#%% Método K-Means

# Considerando que identificamos 3 possíveis clusters na análise hierárquica

kmeans_varejista = KMeans(n_clusters=3, init='random', random_state=100).fit(varejo)

# Criando a variável que indica os clusters no banco de dados

kmeans_clusters = kmeans_varejista.labels_
varejista['cluster_kmeans'] = kmeans_clusters
varejista['cluster_kmeans'] = varejista['cluster_kmeans'].astype('category')
## O padrão dos clusters é o mesmo dos métodos hierárquicos anteriores

#%% Método da silhueta para identificação do nº de clusters

silhueta = []
I = range(2,9) # ponto de parada pode ser parametrizado manualmente
for i in I: 
    kmeansSil = KMeans(n_clusters=i, init='random', random_state=100).fit(varejo)
    silhueta.append(silhouette_score(varejo, kmeansSil.labels_))

plt.figure(figsize=(16,8), dpi=600)
plt.plot(range(2, 9), silhueta, color = 'purple', marker='o') # ajustar range
plt.xlabel('Nº Clusters', fontsize=16)
plt.ylabel('Silhueta Média', fontsize=16)
plt.title('Método da Silhueta', fontsize=16)
plt.axvline(x = silhueta.index(max(silhueta))+2, linestyle = 'dotted', color = 'red') 
plt.show()

# A silhueta média é praticamente igual em 2 ou 3 clusters!
# Para aprimorar a interpretação, vamos manter 3 clusters!

#%% Coordenadas dos centroides dos clusters finais

cent_finais = pd.DataFrame(kmeans_varejista.cluster_centers_)
cent_finais.columns = varejo.columns
cent_finais.index.name = 'cluster'
cent_finais

#%% Plotando as observações e seus centroides dos clusters

plt.figure(figsize=(10,10), dpi=600)
sns.scatterplot(x='atendimento', y='sortimento', data=varejista, hue='cluster_kmeans', palette='viridis', s=100)
sns.scatterplot(x='atendimento', y='sortimento', data=cent_finais, s=40, c='red', label='Centroides', marker="X")
plt.title('Clusters e centroides', fontsize=16)
plt.xlabel('Atendimento', fontsize=16)
plt.ylabel('Sortimento', fontsize=16)
plt.legend()
plt.show()

#%% Estatística F das variáveis

# Atendimento
pg.anova(dv='atendimento', 
         between='cluster_kmeans', 
         data=varejista,
         detailed=True).T

# Sortimento
pg.anova(dv='sortimento', 
         between='cluster_kmeans', 
         data=varejista,
         detailed=True).T

# Organização
pg.anova(dv='organização', 
         between='cluster_kmeans', 
         data=varejista,
         detailed=True).T

#%% Gráfico 3D dos clusters

fig = px.scatter_3d(varejista, 
                    x='atendimento', 
                    y='sortimento', 
                    z='organização',
                    color='cluster_kmeans')

fig.write_html('varejo_final.html')

#%% Identificação das características dos clusters

# Agrupando o banco de dados

analise_varejista = varejista.drop(columns=['loja']).groupby(by=['cluster_kmeans'], observed=True)

# Estatísticas descritivas por grupo

analise_varejista.describe().T

#%% FIM