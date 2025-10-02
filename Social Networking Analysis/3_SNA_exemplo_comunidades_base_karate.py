# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# SOCIAL NETWORK ANALYSIS
# Prof.ª Adriana Silva

#!/usr/bin/env python
# coding: utf-8

# Explorando como calcular medidas por comunidades no Python.

# In[0]: Instalar os pacotes necessários

# pip install igraph
# pip install pycairo
# pip install pandas
# pip install numpy

# In[1]: Importar pacotes
from igraph import Graph, plot
import igraph
import pandas as pd
import numpy as np


# In[2]: Vamos trabalhar com a base de dados Karate:
karate = pd.read_csv("dados/karate.csv",sep=";")
karate


# In[3]: Transformando a base em Grafo
rede_karate = Graph.DataFrame(karate, directed=False)


# In[4]: plotando o grafo 
plot(rede_karate, bbox = (300, 300), vertex_label=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33"])


# In[5]: Avaliando a modularidade de cada método e a quantidade de comunidades geradas:
# Lista de métodos de detecção de comunidade
metodos = {
    "Edge Betweenness": rede_karate.community_edge_betweenness().as_clustering(),
    "Fast Greedy": rede_karate.community_fastgreedy().as_clustering(),
    "Walktrap": rede_karate.community_walktrap().as_clustering(),
    "Louvain": rede_karate.community_multilevel(),
    "Label Propagation": rede_karate.community_label_propagation(),
    "Spin Glass": rede_karate.community_spinglass()
}

# Inicializando a tabela com resultados
resultados = []

# Loop através dos métodos e calcular modularidade, número de comunidades e estatísticas dos tamanhos dos grupos
for metodo_nome, metodo in metodos.items():
    modularidade = metodo.modularity
    num_comunidades = len(metodo)
    
    # Obter o tamanho das comunidades (número de nós em cada comunidade)
    tamanhos_comunidades = [len(comunidade) for comunidade in metodo]
    
    # Calcular as estatísticas: mínimo, máximo, média e desvio padrão do tamanho das comunidades
    min_tam = np.min(tamanhos_comunidades)
    max_tam = np.max(tamanhos_comunidades)
    media_tam = np.mean(tamanhos_comunidades)
    dp_tam = np.std(tamanhos_comunidades)
    
    # Adicionar resultados na lista
    resultados.append([
        metodo_nome, 
        modularidade, 
        num_comunidades, 
        min_tam, 
        max_tam, 
        media_tam, 
        dp_tam
    ])

# Criar um DataFrame para exibir os resultados
df_resultados = pd.DataFrame(resultados, columns=[
    "Método", 
    "Modularidade", 
    "Número de Comunidades", 
    "Min Tamanho", 
    "Max Tamanho", 
    "Média Tamanho", 
    "Desvio Padrão Tamanho"
])

# Mostrar a tabela de resultados
df_resultados


# In[6]: Função para criar dataframe com as medidas da rede 
def stats_rede( rede, nodes:list ):
    df = pd.DataFrame( {"nodes": nodes} )
    df["degree"] = rede.degree()
    df["cc"] = rede.transitivity_local_undirected()
    df["closeness"] = rede.closeness(weights=None, normalized=True)
    df["betweenness"] = rede.betweenness()
    df["bridge"] = df["nodes"].isin(rede.articulation_points()).astype(int)
    return df


# In[7]: Analisando as medidas para a rede completa através da função criada
df_stat_geral = stats_rede( rede_karate, list(range(rede_karate.vcount())) )
df_stat_geral.head()

# In[8]: Função para criar dataframe com as medidas por comunidade
def stats_comunidade( comunidade, nodes:list ):
    df = pd.DataFrame( {"nodes": nodes} )
    df["degree"] = comunidade.degree()
    df["cc"] = comunidade.transitivity_local_undirected()
    df["closeness"] = comunidade.closeness()
    df["betweenness"] = comunidade.betweenness()
    df["bridge"] = df["nodes"].isin(comunidade.articulation_points()).astype(int)
    return df


# In[9]: executando a comunidade pelo método escolhido
comunidades = rede_karate.community_multilevel()


# In[10]: criando data frame que marca a qual comunidade cada nó pertence
df_rede_pandas_karate = pd.DataFrame( {"nodes": list(range(len(comunidades.membership))), "membership": comunidades.membership} )

df_rede_pandas_karate.head()


# In[11]:  pega as comunidades distintas
comunidades_dst = df_rede_pandas_karate["membership"].unique()

# percorre todas comunidades para realizar a estrutura do dataframe
dfs_stats = []
for c in comunidades_dst:
    nodes = df_rede_pandas_karate[ df_rede_pandas_karate["membership"]==c ]["nodes"].tolist()
    dfs_stats.append( stats_comunidade( comunidades.subgraphs()[c], nodes) )
    
df_stat_comu = pd.concat(dfs_stats) # junta a parada toda

# Organiza o df para ficar bonitinho
df_stat_comu = df_stat_comu.merge( df_rede_pandas_karate, on="nodes", how="left" ).sort_values(by="nodes")

df_stat_comu.head()


# In[12]: Criando data frame com todas as medidas da rede e por comunidade:
df_stat = df_stat_comu.merge( df_stat_geral, on=["nodes"], how="inner", suffixes=["_comunidade", "_geral"] )
df_stat.head()

