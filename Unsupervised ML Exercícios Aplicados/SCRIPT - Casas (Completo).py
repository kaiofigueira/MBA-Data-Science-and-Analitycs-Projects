# -*- coding: utf-8 -*-

# Exercícios de Modelos Não Supervisionados de Machine Learning
# MBA em Data Science e Analytics (USP/ESALQ)

# Prof. Dr. Wilson Tarantin Junior

#%% Instalando os pacotes

# Executar diretamente na linha de comando no console

# pip install pandas
# pip install numpy
# pip install matplotlib
# pip install seaborn
# pip install scipy
# pip install factor_analyzer
# pip install prince

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from scipy.stats import chi2_contingency
import prince
import warnings
warnings.filterwarnings("ignore")

#%% Importando o banco de dados

casas = pd.read_excel('preco_casas_completo.xlsx')
## Fonte: https://www.kaggle.com/datasets/elakiricoder/jiffs-house-price-prediction-dataset

#%% Vamos separar o banco de dados com variáveis qualitativas

casas_quali = casas[['large_living_room', 
                     'parking_space', 
                     'front_garden',
                     'swimming_pool',
                     'wall_fence',
                     'water_front',
                     'room_size_class']]

#%% Tabelas de frequências das variáveis

print(casas_quali['large_living_room'].value_counts())
print(casas_quali['parking_space'].value_counts())
print(casas_quali['front_garden'].value_counts())
print(casas_quali['swimming_pool'].value_counts())
print(casas_quali['wall_fence'].value_counts())
print(casas_quali['water_front'].value_counts())
print(casas_quali['room_size_class'].value_counts())

#%% Testes qui-quadrado para associação

# Para simplificar, vamos colocar 'large_living_room' como referência
# Vamos testar todas as associações com 'large_living_room'
# Se alguma não passar, avaliaremos com outra variável

tabela_mca_1 = chi2_contingency(pd.crosstab(casas_quali['large_living_room'], casas_quali['parking_space']))
tabela_mca_2 = chi2_contingency(pd.crosstab(casas_quali['large_living_room'], casas_quali['front_garden']))
tabela_mca_3 = chi2_contingency(pd.crosstab(casas_quali['large_living_room'], casas_quali['swimming_pool']))
tabela_mca_4 = chi2_contingency(pd.crosstab(casas_quali['large_living_room'], casas_quali['wall_fence']))
tabela_mca_5 = chi2_contingency(pd.crosstab(casas_quali['large_living_room'], casas_quali['water_front']))
tabela_mca_6 = chi2_contingency(pd.crosstab(casas_quali['large_living_room'], casas_quali['room_size_class']))

print(f"p-valor da estatística: {round(tabela_mca_1[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_2[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_3[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_4[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_5[1], 4)}")
print(f"p-valor da estatística: {round(tabela_mca_6[1], 4)}")

#%% Elaborando a ACM

# Antes de elaborar a ACM, vamos retirar os "_" nos nomes das variáveis
# É um procedimento operacional para facilitar a geração do mapa perceptual 
casas_quali.columns = casas_quali.columns.str.replace("_", "-") 

# Elaborando a ACM
mca = prince.MCA(n_components=2).fit(casas_quali)

#%% Obtendo os autovalores

tabela_autovalores = mca.eigenvalues_summary
print(tabela_autovalores)

#%% Obtendo as coordenadas-padrão das categorias das variáveis

coord_padrao = mca.column_coordinates(casas_quali)/np.sqrt(mca.eigenvalues_)
print(coord_padrao)

#%% Obtendo as coordenadas-padrão das observações do banco de dados

coord_obs = mca.row_coordinates(casas_quali)
coord_obs.rename(columns={0: 'dim1_acm', 1: 'dim2_acm'}, inplace=True)
print(coord_obs)

#%% Plotando o mapa perceptual (coordenadas-padrão)

chart = coord_padrao.reset_index()

var_chart = pd.Series(chart['index'].str.split('_', expand=True).iloc[:,0])
# Nota: para a função acima ser executada adequadamente, não deixar underline no nome original da variável no dataset!

nome_categ=[]
for col in casas_quali:
    nome_categ.append(casas_quali[col].sort_values(ascending=True).unique())
    categorias = pd.DataFrame(nome_categ).stack().reset_index()

chart_df_mca = pd.DataFrame({'categoria': chart['index'],
                             'obs_x': chart[0],
                             'obs_y': chart[1],
                             'variavel': var_chart,
                             'categoria_id': categorias[0]})

plt.figure(dpi=600)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        ax.text(point['x'] + 0.03, point['y'] - 0.02, point['val'], fontsize=5)

label_point(x = chart_df_mca['obs_x'],
            y = chart_df_mca['obs_y'],
            val = chart_df_mca['categoria_id'],
            ax = plt.gca())

sns.scatterplot(data=chart_df_mca, x='obs_x', y='obs_y', hue='variavel', s=15)
sns.despine(top=True, right=True, left=False, bottom=False)
plt.axhline(y=0, color='lightgrey', ls='--', linewidth=0.8)
plt.axvline(x=0, color='lightgrey', ls='--', linewidth=0.8)
plt.tick_params(size=2, labelsize=6)
plt.legend(bbox_to_anchor=(1.1,-0.15), fancybox=True, shadow=False, ncols=7, fontsize='5')
plt.title("Mapa Perceptual - MCA", fontsize=12)
plt.xlabel(f"Dim. 1: {tabela_autovalores.iloc[0,1]} da inércia", fontsize=8)
plt.ylabel(f"Dim. 2: {tabela_autovalores.iloc[1,1]} da inércia", fontsize=8)
plt.show()

#%% Vamos criar o banco de dados com variáveis métricas

# Separando as variáveis originalmente métricas
casas_quanti = casas[['land_size_sqm',
                      'house_size_sqm',
                      'no_of_rooms',
                      'no_of_bathrooms',
                      'distance_to_school',
                      'house_age',
                      'distance_to_supermarket_km',
                      'crime_rate_index']]

# Adicionando as coordenadas das observações extraídas da ACM
casas_quanti = pd.concat([casas_quanti, coord_obs], axis=1)

#%% Teste de Esfericidade de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(casas_quanti)

print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Definindo a PCA (procedimento inicial com todos os fatores possíveis)

fa = FactorAnalyzer(n_factors=10, method='principal', rotation=None).fit(casas_quanti)

#%% Obtendo todos os possíveis autovalores

autovalores = fa.get_eigenvalues()[0]
print(autovalores)

# Soma dos autovalores
round(autovalores.sum(), 2)

#%% Aplicando o critério da raiz latente

sel_fator = sum(autovalores > 1)
print(f'Quantidade de fatores selecionados: {sel_fator}')

#%% Redefinindo a PCA (critério da raiz latente)

fa = FactorAnalyzer(n_factors=sel_fator, method='principal', rotation=None).fit(casas_quanti)

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
tabela_cargas.index = casas_quanti.columns

print(tabela_cargas)

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = casas_quanti.columns

print(tabela_comunalidades)

#%% Extração dos fatores para as observações do banco de dados

fatores = pd.DataFrame(fa.transform(casas_quanti))
fatores.columns =  [f"Fator {i+1}" for i, v in enumerate(fatores.columns)]

# Adicionando os fatores ao banco de dados

casas = pd.concat([casas.reset_index(drop=True), fatores], axis=1)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = casas_quanti.columns

print(tabela_scores)

#%% Analisando os scores fatoriais em cada fator extraído

tabela_scores_graph = tabela_scores.reset_index()
tabela_scores_graph = tabela_scores_graph.melt(id_vars='index')

plt.figure(figsize=(12,8), dpi=600)
sns.barplot(data=tabela_scores_graph, x='variable', y='value', hue='index', palette='viridis')
plt.legend(title='Variáveis', bbox_to_anchor=(1,1), fontsize = '6')
plt.title('Scores Fatoriais', fontsize='12')
plt.xlabel(xlabel=None)
plt.ylabel(ylabel=None)
plt.show()

#%% Fim!