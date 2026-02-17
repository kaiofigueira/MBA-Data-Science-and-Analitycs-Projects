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
# pip install plotly
# pip install scipy
# pip install statsmodels
# pip install factor_analyzer

#%% Importando os pacotes

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from factor_analyzer import FactorAnalyzer
from factor_analyzer.factor_analyzer import calculate_bartlett_sphericity
from scipy.stats import chi2_contingency
import statsmodels.api as sm
import plotly.graph_objects as go
import warnings
warnings.filterwarnings("ignore")

#%% Importando o banco de dados

pisa = pd.read_csv('notas_pisa.csv', delimiter=',')
# Fonte: https://pisadataexplorer.oecd.org/ide/idepisa/report.aspx

# Informações gerais do dataset
pisa.info()

#%% Vamos analisar apenas as notas de 2022

pisa.drop(columns=['mathematics_2018', 'reading_2018', 'science_2018'], inplace=True)
 
#%% Ajustando as variáveis de notas para numéricas

pisa['mathematics_2022'] = pd.to_numeric(pisa['mathematics_2022'], errors='coerce')
pisa['reading_2022'] = pd.to_numeric(pisa['reading_2022'], errors='coerce')
pisa['science_2022'] = pd.to_numeric(pisa['science_2022'], errors='coerce')

#%% Note que há valores faltantes "nan", vamos remover

pisa.dropna(inplace=True)

#%% Iniciando a análise fatorial PCA

pisa_pca = pisa.drop(columns=['country', 'group'])

#%% Análise gráfica das correlações de Pearson

matriz_corr = pisa_pca.corr()

plt.figure(figsize=(12,8), dpi=600)
sns.heatmap(matriz_corr, annot=True, 
            cmap = plt.cm.Purples,
            annot_kws={'size':12})

#%% Estatísticas descritivas

pisa_pca.describe()

#%% Teste de Esfericidade de Bartlett

bartlett, p_value = calculate_bartlett_sphericity(pisa_pca)
print(f'Qui² Bartlett: {round(bartlett, 2)}')
print(f'p-valor: {round(p_value, 4)}')

#%% Definindo a PCA (procedimento inicial com todos os fatores possíveis)

fa = FactorAnalyzer(n_factors=3, method='principal', rotation=None).fit(pisa_pca)

#%% Obtendo todos os possíveis autovalores

autovalores = fa.get_eigenvalues()[0]
print(autovalores)

#%% Redefinindo a PCA (critério da raiz latente)

# Escolhendo n_factors com base em autovalores > 1

fa = FactorAnalyzer(n_factors=1, method='principal', rotation=None).fit(pisa_pca)

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
tabela_cargas.index = pisa_pca.columns

print(tabela_cargas)

#%% Determinando as comunalidades

comunalidades = fa.get_communalities()

tabela_comunalidades = pd.DataFrame(comunalidades)
tabela_comunalidades.columns = ['Comunalidades']
tabela_comunalidades.index = pisa_pca.columns

print(tabela_comunalidades)

#%% Extração do fator para as observações do banco de dados

fator = pd.DataFrame(fa.transform(pisa_pca))
fator.columns = ['fator_2022']

# Adicionando os fatores ao banco de dados

pisa = pd.concat([pisa.reset_index(drop=True), fator], axis=1)

# Organizando o dataset por meio do fator

pisa.sort_values('fator_2022', ascending=False, inplace=True)
pisa.reset_index(drop=True, inplace=True)

#%% Identificando os scores fatoriais

scores = fa.weights_

tabela_scores = pd.DataFrame(scores)
tabela_scores.columns = [f"Fator {i+1}" for i, v in enumerate(tabela_scores.columns)]
tabela_scores.index = pisa_pca.columns

print(tabela_scores)

#%% Vamos categorizar o fator

# Criando 4 grupos
pisa['categoria'] = pd.qcut(pisa['fator_2022'], 4, labels=list(['menores', 'médio_menor', 'médio_maior', 'maiores']))

#%% Há associação com o grupo dos países?

# Criando a tabela de contingência

tabela = pd.crosstab(pisa['categoria'], pisa['group'])
print(tabela)

#%% Analisando a significância estatística da associação (teste qui²)

teste_qui2 = chi2_contingency(tabela)

print(f"estatística qui²: {round(teste_qui2[0], 2)}")
print(f"p-valor da estatística: {round(teste_qui2[1], 4)}")
print(f"graus de liberdade: {teste_qui2[2]}")

#%% Análise dos resíduos provenientes da tabela de contingência

# Parametrizando a função
tab_cont = sm.stats.Table(tabela)

# Resíduos padronizados ajustados
print(tab_cont.standardized_resids)

#%% Plotando o gráfico

fig = go.Figure()

maxz = np.max(tab_cont.standardized_resids)+0.1
minz = np.min(tab_cont.standardized_resids)-0.1

colorscale = ['purple' if i>1.96 else '#FAF9F6' for i in np.arange(minz,maxz,0.01)]

fig.add_trace(
    go.Heatmap(
        x = tab_cont.standardized_resids.columns,
        y = tab_cont.standardized_resids.index,
        z = np.array(tab_cont.standardized_resids),
        text=tab_cont.standardized_resids.values,
        texttemplate='%{text:.2f}',
        showscale=False,
        colorscale=colorscale))

fig.update_layout(
    title='<b>Resíduos Padronizados Ajustados</b>',
    title_x = 0.5,
    height = 600,
    width = 600)

fig.write_html('resid_pisa.html')

#%% Conclusão

# Na Anacor são gerados 'm' autovalores: m = mín(I-1,J-1)
# Como a variável 'group' tem 2 categorias, não há mapa bidimensional
# Assim, neste caso, encerramos a análise nos resíduos padronizados ajustados
# Existindo outras variáveis, poderia ser realizada uma ACM

#%% Fim!