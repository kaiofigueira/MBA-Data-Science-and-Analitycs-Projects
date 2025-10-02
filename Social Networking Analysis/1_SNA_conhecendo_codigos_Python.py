# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# SOCIAL NETWORK ANALYSIS
# Prof.ª Adriana Silva

#!/usr/bin/env python
# coding: utf-8

# Conhecendo os códigos para SNA no Python.

# In[0]: Instalar os pacotes necessários

# pip install igraph
# pip install pycairo
# pip install pandas
# pip install numpy


# In[1]: Importar pacotes
from igraph import Graph, plot
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]: Praticando com nosso exemplo feito à mão:

g = Graph() 
g.add_vertices(5)
g.vs["name"] = ["A", "B", "C", "D", "E"]
g.add_edges([("A","B"), ("B","C"), ("B","D"), ("D","C"), ("C","E")])
print(g)


# In[3]: Inserindo pesos no meu grafo (neste caso, todas 1)
g.es['weight']=[1,1,1,1,1]


# In[4]: Plotando o grafo:
plot(g)


# In[5]: Sofisticando o grafo utilizando funções para melhorar a visualização:

layout = g.layout("kk")
# existem varios layouts https://igraph.org/python/doc/tutorial/visualisation.html#graph-layouts
visual_style = {}
visual_style["vertex_color"] = "white"
visual_style["vertex_size"] = 20
visual_style["vertex_label"] = ["A", "B", "C", "D", "E"]
visual_style["edge_width"] = g.es['weight']
visual_style["bbox"] = (200, 200)
plot(g, **visual_style)


# In[6]: Extraindo metricas da rede (geral):

#densidade do grafo
g.density()


# In[7]: diametro da rede
g.diameter()


# In[8]: raio da rede
g.radius()


# In[9]: comprimeiro medio do caminho
g.average_path_length()

# In[10]: Extraindo metricas dos nós (vertices):

# degree
g.degree()


# In[11]: clustering coeficient
g.transitivity_local_undirected()


# In[12]: closeness
g.closeness(weights=None, normalized=True)


# In[13]: closeness
g.closeness(weights=None, normalized=False)
#quando normalized=False o valor anterior é dividido pelo número de caminhos curtos


# In[14]: betweeness
g.betweenness(weights=None)


# In[15]: excentricidade
g.eccentricity()


# In[16]:  o famoso e tao desejado bridge (ponte ou ponto de articulacao)
g.articulation_points()
#retorna a lista de [0 até 6] com a posição da letra (nesse caso B, C)


# In[17]: Sofisticando o grafo utilizando as medidas para melhorar a visualização:

# para plotar aumentando o tamanho dos nos a partir do degree, por exemplo:
deg = g.degree()
deg = [i * 10 for i in deg]
deg
plot(g, vertex_size=deg,  vertex_color=['red'], vertex_label=["A", "B", "C", "D", "E"],  edge_width=[ 4], edge_color=['grey'], bbox = (200, 200))


# In[19]: Como podemor ver as tabelas com essas medidas?
# Você tem que armazenar as medidas como atributos armazenando todas as medidas 
# e transformando em data frame desta forma:

names = ["A", "B", "C", "D", "E"]
degree = g.degree()
cc = g.transitivity_local_undirected()
closeness = g.closeness(weights=None, normalized=True)
betweenneess = g.betweenness(weights=None)
excentricidade = g.eccentricity()

a = pd.DataFrame(np.column_stack([names, degree, cc, closeness, betweenneess, excentricidade]), columns=['names','degree', 'cc', 'closeness', 'betweenneess', 'excentricidade'])
a


# In[21]: outra forma >>>> Função para criar dataframe com as medidas da rede 
def stats_rede( rede, nodes:list ):
    df = pd.DataFrame( {"nodes": nodes} )
    df["degree"] = rede.degree()
    df["cc"] = rede.transitivity_local_undirected()
    df["closeness"] = rede.closeness(weights=None, normalized=False)
    df["betweenness"] = rede.betweenness()
    df["bridge"] = df["nodes"].isin(rede.articulation_points()).astype(int)
    return df


# In[22]: chamando a função criada
df_stat_geral = stats_rede( g, list(range(g.vcount())) )
df_stat_geral

