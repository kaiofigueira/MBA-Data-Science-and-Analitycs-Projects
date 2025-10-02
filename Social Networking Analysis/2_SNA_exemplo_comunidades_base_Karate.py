# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# SOCIAL NETWORK ANALYSIS
# Prof.ª Adriana Silva

#!/usr/bin/env python
# coding: utf-8

# Explorando comunidades no Python.

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


# In[2]: importando base karate
karate = pd.read_csv("dados/karate.csv",sep=";")
karate


# In[3]: Transformando em Grafo
rede_karate = Graph.DataFrame(karate, directed=False)


# In[4]: plotando o grafo 
plot(rede_karate, bbox = (300, 300), vertex_label=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33"])


# In[5]: Explorando Comunidades: método cluster edge betweenness
    
#executa e define as comunidades
comunidade_ceb = rede_karate.community_edge_betweenness()
plot(comunidade_ceb)

#grafo marcando comunidades
comunidade=comunidade_ceb.as_clustering()
visual_style = dict()
visual_style["bbox"] = (300, 300)
igraph.plot(comunidade, mark_groups = True, **visual_style, vertex_label=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33"])

#observando a modularidade pelo método Cluster edge betweenness
comunidade.modularity


# In[8]: Explorando Comunidades: método fast greedy
comunidade_fg = rede_karate.community_fastgreedy()
comunidade1=comunidade_fg.as_clustering()
visual_style = dict()
visual_style["bbox"] = (300, 300)
igraph.plot(comunidade1, mark_groups = True, **visual_style, vertex_label=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33"])

#observando a modularidade Comunidade fast greedy
comunidade1.modularity


# In[10]: Explorando Comunidades: método walktrap
comunidade_wtrap = rede_karate.community_walktrap(weights=None, steps = 4)
comunidade2=comunidade_wtrap.as_clustering()
visual_style = dict()
visual_style["bbox"] = (300, 300)
igraph.plot(comunidade2, mark_groups = True, **visual_style, vertex_label=["0","1","2","3","4","5","6","7","8","9","10","11","12","13","14","15","16","17","18","19","20","21","22","23","24","25","26","27","28","29","30","31","32","33"])

#observando a modularidade Comunidade walktrap
comunidade2.modularity


# In[12]: Tabela sumarizando resultados dos diferentes métodos
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

# In[13]: Mostrar a tabela de resultados
print(df_resultados)

