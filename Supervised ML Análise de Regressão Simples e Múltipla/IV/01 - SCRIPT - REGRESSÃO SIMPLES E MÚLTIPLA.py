# UNIVERSIDADE DE SÃO PAULO
# MBA DATA SCIENCE & ANALYTICS USP/ESALQ
# SUPERVISED MACHINE LEARNING: ANÁLISE DE REGRESSÃO SIMPLES E MÚLTIPLA
# Prof. Dr. Luiz Paulo Fávero

#!/usr/bin/env python
# coding: utf-8

# In[0.1]: Instalação dos pacotes

# Executar cada linha de comando no Console (sem o #)

# pip install pandas
# pip install numpy
# pip install seaborn
# pip install matplotlib
# pip install plotly
# pip install scipy
# pip install statsmodels
# pip install scikit-learn
# pip install pingouin
# pip install --upgrade statstests
# pip install pygame

# In[0.2]: Importação dos pacotes

import pandas as pd # manipulação de dados em formato de dataframe
import numpy as np # operações matemáticas
import seaborn as sns # visualização gráfica
import matplotlib.pyplot as plt # visualização gráfica
import plotly.graph_objects as go # gráficos 3D
from scipy.stats import pearsonr # correlações de Pearson
import statsmodels.api as sm # estimação de modelos
from statsmodels.iolib.summary2 import summary_col # comparação entre modelos
from sklearn.preprocessing import LabelEncoder # transformação de dados
import pingouin as pg # outro modo para obtenção de matrizes de correlações
from statstests.process import stepwise # procedimento Stepwise
from statstests.tests import shapiro_francia # teste de Shapiro-Francia
from scipy.stats import boxcox # transformação de Box-Cox
from scipy.stats import norm # para plotagem da curva normal
from scipy import stats # utilizado na definição da função 'breusch_pagan_test'
import pygame # reprodução de sons

import warnings
warnings.filterwarnings('ignore')


# In[EXEMPLO 1]:
#############################################################################
#                          REGRESSÃO LINEAR SIMPLES                         #
#                  EXEMPLO 1 - CARREGAMENTO DA BASE DE DADOS                #
#############################################################################
    
df_tempodist = pd.read_csv('tempodist.csv', delimiter=',')
df_tempodist

# Características das variáveis do dataset
df_tempodist.info()

# Estatísticas univariadas
df_tempodist.describe()

# In[1.1]: Gráfico de dispersão com o ajuste linear (fitted values de um modelo
#de regressão) que se adequa às observações: função 'regplot' do pacote 'seaborn'

plt.figure(figsize=(15,10))
sns.regplot(data=df_tempodist, x='distancia', y='tempo', marker='o', ci=False,
            scatter_kws={"color":'navy', 'alpha':0.9, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('Valores Reais e Fitted Values (Modelo de Regressão)', fontsize=30)
plt.xlabel('Distância', fontsize=24)
plt.ylabel('Tempo', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Reais', 'Fitted Values'], fontsize=24, loc='upper left')
plt.show()

# In[1.2]: Gráfico de dispersão interativo (figura 'EXEMPLO1.html' salva na
#pasta do curso)

# Dados do gráfico
x = df_tempodist['distancia']
y = df_tempodist['tempo']

# Definição da regressão linear
slope, intercept = np.polyfit(x, y, 1)
y_trend = slope * x + intercept

fig = go.Figure()

# Inserção dos pontos (valores reais)
fig.add_trace(go.Scatter(
    x=x,
    y=y,
    mode='markers',
    marker=dict(color='navy', size=20), name='Valores Reais')
    )

# Inserção da reta (fitted values)
fig.add_trace(go.Scatter(
    x=x,
    y=y_trend,
    mode='lines',
    line=dict(color='dimgray', width=5), name='Fitted Values')
    )

# Configurações de layout
fig.update_layout(
    xaxis_title='Distância',
    yaxis_title='Tempo',
    title={
        'text': 'Gráfico de Dispersão com Fitted Values',
        'font': {'size': 20, 'color': 'black', 'family': 'Arial'},
        'x': 0.5,
        'y': 0.97,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    plot_bgcolor='snow',
    xaxis=dict(gridcolor='black'),
    yaxis=dict(gridcolor='black'),
    showlegend=True
)

fig.write_html('EXEMPLO1.html')

# Abrir o arquivo HTML no navegador
import webbrowser
webbrowser.open('EXEMPLO1.html')

# In[1.3]: Estimação do modelo de regressão linear simples

# Estimação do modelo
modelo = sm.OLS.from_formula('tempo ~ distancia', df_tempodist).fit()

# Observação dos parâmetros resultantes da estimação
modelo.summary()

# In[1.4]: Salvando fitted values (variável yhat) e residuals (variável erro)
#no dataset

df_tempodist['yhat'] = modelo.fittedvalues
df_tempodist['erro'] = modelo.resid
df_tempodist

# In[1.5]: Gráfico didático para visualizar o conceito de R²

plt.figure(figsize=(15,10))
y = df_tempodist['tempo']
yhat = df_tempodist['yhat']
x = df_tempodist['distancia']
mean = np.full(x.shape[0] , y.mean(), dtype=int)

for i in range(len(x)-1):
    plt.plot(x, yhat, color='grey', linewidth=7)
    plt.plot([x[i], x[i]], [yhat[i], mean[i]], '--', color='darkorchid',
             linewidth=5)
    plt.plot([x[i], x[i]], [yhat[i], y[i]],':', color='limegreen', linewidth=5)
    plt.scatter(x, y, color='navy', s=220, alpha=0.2)
    plt.axhline(y = y.mean(), color = 'silver', linestyle = '-', linewidth=4)
    plt.title('R²: ' + str(round(modelo.rsquared, 4)), fontsize=30)
    plt.xlabel('Distância', fontsize=24)
    plt.ylabel('Tempo', fontsize=24)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    plt.xlim(0, 35)
    plt.ylim(0, 60)
    plt.legend(['Fitted Values', 'Ychapéu - Ymédio', 'Erro = Y - Ychapéu'],
               fontsize=22, loc='upper left')
plt.show()

# In[1.6]: Cálculo manual do R²

R2 = ((df_tempodist['yhat']-
       df_tempodist['tempo'].mean())**2).sum()/(((df_tempodist['yhat']-
                                        df_tempodist['tempo'].mean())**2).sum()+
                                        (df_tempodist['erro']**2).sum())

float(round(R2,4))

# In[1.7]: Coeficiente de ajuste (R²) é a correlação ao quadrado

# Correlação de Pearson
df_tempodist[['tempo','distancia']].corr()

# R²
(df_tempodist[['tempo','distancia']].corr())**2

# R² de maneira direta
float(modelo.rsquared)

# In[1.8]: Modelo auxiliar para mostrar R² igual a 100% (para fins didáticos)

# Estimação do modelo com yhat como variável dependente resultará em um modelo
#com R² igual a 100%
modelo_auxiliar = sm.OLS.from_formula('yhat ~ distancia', df_tempodist).fit()

# Parâmetros resultantes da estimação deste modelo didático
modelo_auxiliar.summary()

# In[1.9]:Gráfico mostrando o perfect fit

plt.figure(figsize=(15,10))
sns.scatterplot(data=df_tempodist, x='distancia', y='yhat',
                color='navy', alpha=0.9, s=300)
sns.regplot(data=df_tempodist, x='distancia', y='yhat', ci=False, scatter=False,
            label='Fitted Values',
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('Perfect Fit', fontsize=30)
plt.xlabel('Distância', fontsize=24)
plt.ylabel('Tempo', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(loc='upper left', fontsize=24)
plt.show()

# In[1.10]:Gráfico mostrando o perfect fit com figura .JPG e som .MP3

import urllib.request
from PIL import Image
from io import BytesIO

# Define a URL da imagem (FONTE: Divulgação/Warner Bros. Pictures)
url = "https://cinebuzz.uol.com.br/media/uploads/harry_potter_3_WumwIEd.jpg"

# Define os cabeçalhos da solicitação
headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}

# Cria uma solicitação com os cabeçalhos
request = urllib.request.Request(url, headers=headers)

# Abre a URL e lê os dados da imagem
response = urllib.request.urlopen(request)
image_data = response.read()

# Carrega a imagem em um objeto PIL (Python Imaging Library)
imagem = Image.open(BytesIO(image_data)).convert("RGB")

# Define o tamanho do gráfico
fig, ax = plt.subplots(figsize=(15, 10))

# Define as dimensões e a posição desejada da imagem
ax.imshow(
    imagem,
    extent=[0, 35, 0, 60],  # mesmos limites do gráfico
    aspect='auto',
    alpha=0.20,
    zorder=0
)

# Cria o gráfico por meio da função 'regplot' do pacote 'seaborn'
sns.scatterplot(data=df_tempodist, x='distancia', y='yhat', 
                color='navy', alpha=0.9, s=300, ax=ax, legend=False)
sns.regplot(
    data=df_tempodist, x='distancia', y='yhat',
    ci=False, scatter=False,
    line_kws={"color": "grey", "linewidth": 5, "label": "Fitted Values"},
    ax=ax
)
ax.set_title('Perfect Fit', fontsize=30)
ax.set_xlabel('Distância', fontsize=24)
ax.set_ylabel('Tempo', fontsize=24)
ax.set_xlim(0, 35)
ax.set_ylim(0, 60)
ax.tick_params(axis='both', labelsize=18)
linha = ax.lines[-1]
ax.legend([linha], ['Fitted Values'], loc='upper left', fontsize=24)
plt.show()

# Reproduz um som padrão (arquivo na pasta do curso)
pygame.mixer.init()
pygame.mixer.music.load("sound.mp3")
pygame.mixer.music.play()

# In[1.11]: Voltando ao nosso modelo original

# Gráfico com intervalo de confiança de 90%

plt.figure(figsize=(15,10))
sns.regplot(data=df_tempodist, x='distancia', y='tempo', marker='o', ci=90,
            scatter_kws={"color":'navy', 'alpha':0.7, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('IC: 90%', fontsize=30)
plt.xlabel('Distância', fontsize=24)
plt.ylabel('Tempo', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Reais', 'Fitted Values', '90% IC'],
           fontsize=24, loc='upper left')
plt.show()

# In[1.12]: Gráfico com intervalo de confiança de 95%

plt.figure(figsize=(15,10))
sns.regplot(data=df_tempodist, x='distancia', y='tempo', marker='o', ci=95,
            scatter_kws={"color":'navy', 'alpha':0.7, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('IC: 95%', fontsize=30)
plt.xlabel('Distância', fontsize=24)
plt.ylabel('Tempo', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Reais', 'Fitted Values', '95% IC'],
           fontsize=24, loc='upper left')
plt.show()

# In[1.13]: Gráfico com intervalo de confiança de 99%

plt.figure(figsize=(15,10))
sns.regplot(data=df_tempodist, x='distancia', y='tempo', marker='o', ci=99,
            scatter_kws={"color":'navy', 'alpha':0.7, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('IC: 99%', fontsize=30)
plt.xlabel('Distância', fontsize=24)
plt.ylabel('Tempo', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Reais', 'Fitted Values', '99% IC'],
           fontsize=24, loc='upper left')
plt.show()

# In[1.14]: Gráfico com intervalo de confiança de 99,99999%

plt.figure(figsize=(15,10))
sns.regplot(data=df_tempodist, x='distancia', y='tempo', marker='o', ci=99.99999,
            scatter_kws={"color":'navy', 'alpha':0.7, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('IC: 99,99999%', fontsize=30)
plt.xlabel('Distância', fontsize=24)
plt.ylabel('Tempo', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Reais', 'Fitted Values', '99,99999% IC'],
           fontsize=24, loc='upper left')
plt.show()

# In[1.15]: Calculando os intervalos de confiança

# Nível de significância de 10% / Nível de confiança de 90%
modelo.conf_int(alpha=0.1)

# Nível de significância de 5% / Nível de confiança de 95%
modelo.conf_int(alpha=0.05)

# Nível de significância de 1% / Nível de confiança de 99%
modelo.conf_int(alpha=0.01)

# Nível de significância de 0,00001% / Nível de confiança de 99,99999%
modelo.conf_int(alpha=0.0000001)

# In[1.16]: Fazendo predições em modelos OLS
# Ex.: Qual seria o tempo gasto, em média, para percorrer a distância de 25km?

# Cálculo manual
5.8784 + 1.4189*(25)

# Cálculo utilizando os próprios parâmetros estimados do modelo
float(modelo.params[0] + modelo.params[1]*(25))

# Maneira direta utilizando a função 'DataFrame' do pacote 'pandas' dentro
#da função 'predict'
modelo.predict(pd.DataFrame({'distancia':[25]}))

# In[1.17]: Nova modelagem para o mesmo exemplo, com novo dataset que
#contém replicações

# Quantas replicações de cada linha você quer? -> função 'repeat' do 'numpy'
df_replicado = pd.DataFrame(np.repeat(df_tempodist.values, 3, axis=0))
df_replicado.columns = df_tempodist.columns
df_replicado

# In[1.18]: Estimação do modelo com valores replicados

modelo_replicado = sm.OLS.from_formula('tempo ~ distancia',
                                       df_replicado).fit()

# Parâmetros do 'modelo_replicado'
modelo_replicado.summary()

# In[1.19]: Calculando os novos intervalos de confiança

# Nível de significância de 5% / Nível de confiança de 95%
modelo_replicado.conf_int(alpha=0.05)

# In[1.20]: Plotando o novo gráfico com intervalo de confiança de 95%
# Note o estreitamento da amplitude dos intervalos de confiança!

plt.figure(figsize=(15,10))
sns.regplot(data=df_replicado, x='distancia', y='tempo', marker='o', ci=95,
            scatter_kws={"color":'navy', 'alpha':0.7, 's':220},
            line_kws={"color":'grey', 'linewidth': 5})
plt.title('IC: 95%', fontsize=30)
plt.xlabel('Distância', fontsize=24)
plt.ylabel('Tempo', fontsize=24)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Reais', 'Fitted Values', '95% IC'],
           fontsize=24, loc='upper left')
plt.show()

# In[1.21]: PROCEDIMENTO ERRADO: ELIMINAR O INTERCEPTO QUANDO ESTE NÃO SE
#MOSTRAR ESTATISTICAMENTE SIGNIFICANTE

modelo_errado = sm.OLS.from_formula('tempo ~ 0 + distancia', df_tempodist).fit()

# Parâmetros do 'modelo_errado'
modelo_errado.summary()

# In[1.22]: Comparando os parâmetros do modelo inicial (objeto 'modelo')
#com o 'modelo_errado' pela função 'summary_col' do pacote
#'statsmodels.iolib.summary2'

summary_col([modelo, modelo_errado])

# Outro modo mais completo também pela função 'summary_col'
summary_col([modelo, modelo_errado],
            model_names=["MODELO INICIAL","MODELO ERRADO"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })

# In[1.23]: Gráfico didático para visualizar o viés decorrente de se eliminar
#erroneamente o intercepto em modelos regressivos

x = df_tempodist['distancia']
y = df_tempodist['tempo']

yhat = df_tempodist['yhat']
yhat_errado = modelo_errado.fittedvalues

plt.plot(x, y, 'o', color='navy')
plt.plot(x, yhat, color='gray')
plt.plot(x, yhat_errado, color='red')
plt.xlabel("Distância")
plt.ylabel("Tempo")
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Observados','Fitted Values OLS',
            'Sem Intercepto'], fontsize=9)
plt.show()

#%% In[1.24]: DÚVIDA: Qual estimação devo escolher? (com figura proveninente
#de URL)

import urllib.request
import matplotlib.pyplot as plt
from PIL import Image
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Definição da URL da imagem
url = "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQKNf7Jk3b2LG23egCN7w7TW0275Vd2_lhYWHLlGGizplLYc74wLukF-EbOIB8YY8YB9L0&usqp=CAU"

# Carregamento da imagem da URL
imagem = Image.open(urllib.request.urlopen(url))

# Construção do gráfico
fig, ax = plt.subplots(figsize=(6, 4))

x = df_tempodist['distancia']
y = df_tempodist['tempo']

yhat = df_tempodist['yhat']
yhat_errado = modelo_errado.fittedvalues

ax.plot(x, y, 'o', color='navy')
ax.plot(x, yhat, color='gray')
ax.plot(x, yhat_errado, color='red')

ax.set_xlabel("Distância")
ax.set_ylabel("Tempo")
ax.set_xlim(0, 35)
ax.set_ylim(0, 60)

ax.legend(
    ['Valores Observados', 'Fitted Values OLS', 'Sem Intercepto'],
    fontsize=9
)

# Definição das dimensões e da posição desejada da imagem
imagebox = OffsetImage(imagem, zoom=0.40)  # tamanho da imagem

ab = AnnotationBbox(
    imagebox,
    (0.85, 0.15),
    xycoords='axes fraction',
    frameon=False,
    box_alignment=(0.4, 0.3)
)

ax.add_artist(ab)

plt.tight_layout()
plt.show()

# In[1.25]: DECISÃO: DEVO ESCOLHER O MODELO COM INTERCEPTO!

# Definição das URLs das imagems
url1 = "https://cdn-icons-png.flaticon.com/512/5290/5290081.png"
url2 = "https://i.pinimg.com/originals/d3/82/6a/d3826a943b0d3a9d54ec3d3cba01d0ef.png"

# Carregamento das imagens das URLs
imagem1 = Image.open(urllib.request.urlopen(url1))
imagem2 = Image.open(urllib.request.urlopen(url2))

# Definição das dimensões e das posições desejadas das imagens
nova_largura1 = 600  # Largura da imagem 1 redimensionada
nova_altura1 = 800  # Altura da imagem 1 redimensionada
posicao_x1 = 1130  # Posição horizontal da imagem 1
posicao_y1 = 1080  # Posição vertical da imagem 1

nova_largura2 = 500  # Largura da imagem 2 redimensionada
nova_altura2 = 500  # Altura da imagem 2 redimensionada
posicao_x2 = 1929  # Posição horizontal da imagem 2
posicao_y2 = 900  # Posição vertical da imagem 2

# Redimensionamento das imagens
imagem_redimensionada1 = imagem1.resize((nova_largura1, nova_altura1))
imagem_redimensionada2 = imagem2.resize((nova_largura2, nova_altura2))

# Construção do gráfico
x = df_tempodist['distancia']
y = df_tempodist['tempo']

yhat = df_tempodist['yhat']
yhat_errado = modelo_errado.fittedvalues

plt.plot(x, y, 'o', color='navy')
plt.plot(x, yhat, color='gray')
plt.plot(x, yhat_errado, color='red')
plt.xlabel("Distância")
plt.ylabel("Tempo")
plt.xlim(0, 35)
plt.ylim(0, 60)
plt.legend(['Valores Observados','Fitted Values OLS',
            'Sem Intercepto'], fontsize=9)

# Inserção das imagens redimensionadas em posições específicas no gráfico
plt.figimage(imagem_redimensionada1, posicao_x1, posicao_y1, zorder=1)
plt.figimage(imagem_redimensionada2, posicao_x2, posicao_y2, zorder=1)

plt.show()


# In[EXEMPLO 2]:
#############################################################################
#                         REGRESSÃO LINEAR MÚLTIPLA                         #
#                 EXEMPLO 2 - CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################

df_paises = pd.read_csv('paises.csv', delimiter=',', encoding="utf-8")
df_paises

#Características das variáveis do dataset
df_paises.info()

#Estatísticas univariadas
df_paises.describe()

# In[2.1]: Gráfico 3D com scatter gerado em HTML e aberto no browser
#(figura 'EXEMPLO2_scatter3D.html' salva na pasta do curso)

trace = go.Scatter3d(
    x=df_paises['horas'], 
    y=df_paises['idade'], 
    z=df_paises['cpi'], 
    mode='markers',
    marker={
        'size': 10,
        'color': 'darkorchid',
        'opacity': 0.7,
    },
)

layout = go.Layout(
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
    width=800,
    height=800,
    plot_bgcolor='white',
    scene=dict(
        xaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            backgroundcolor='whitesmoke'
        ),
        yaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            backgroundcolor='whitesmoke'
        ),
        zaxis=dict(
            gridcolor='rgb(200, 200, 200)',
            backgroundcolor='whitesmoke'
        )
    )
)

data = [trace]

plot_figure = go.Figure(data=data, layout=layout)
plot_figure.update_layout(scene=dict(
    xaxis_title='horas',
    yaxis_title='idade',
    zaxis_title='cpi'
))

plot_figure.write_html('EXEMPLO2_scatter3D.html')

# Abre o arquivo HTML no browser
import webbrowser
webbrowser.open('EXEMPLO2_scatter3D.html')

# In[2.2]: Matriz de correlações

correlation_matrix = df_paises.iloc[:,1:4].corr()
correlation_matrix

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 25}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# Paletas de cores ('_r' reverte a sequência de cores):
# viridis
# inferno
# magma
# cividis
# coolwarm
# Blues
# Greens
# Reds

# In[2.3]: Diagrama interessante (grafo) que mostra a inter-relação entre as
#variáveis e a magnitude das correlações entre elas

# conda install networkx
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Criação de um grafo direcionado
G = nx.DiGraph()

# Adição das variáveis como nós do grafo
for variable in correlation_matrix.columns:
    G.add_node(variable)

# Adição das arestas com espessuras proporcionais às correlações
for i, variable1 in enumerate(correlation_matrix.columns):
    for j, variable2 in enumerate(correlation_matrix.columns):
        if i != j:
            correlation = correlation_matrix.iloc[i, j]
            if abs(correlation) > 0:
                G.add_edge(variable1, variable2, weight=correlation)

# Obtenção da lista de correlações das arestas
correlations = [d["weight"] for _, _, d in G.edges(data=True)]

# Definição da dimensão dos nós
node_size = 2700

# Definição da cor dos nós
node_color = 'black'

# Definição da escala de cores das retas (correspondência com as correlações)
cmap = plt.colormaps.get_cmap('coolwarm_r')

# Criação de uma lista de espessuras das arestas proporcional às correlações
edge_widths = [abs(d["weight"]) * 25 for _, _, d in G.edges(data=True)]

# Criação do layout do grafo com maior distância entre os nós
pos = nx.spring_layout(G, k=0.75)  # k para controlar a distância entre os nós

# Desenho dos nós e das arestas com base nas correlações e espessuras
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                       edge_cmap=cmap, alpha=0.7)

# Adição dos rótulos dos nós
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=10, font_color='white')

# Ajuste dos limites dos eixos
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")

# Criação da legenda com a escala de cores definida
smp = cm.ScalarMappable(cmap=cmap)
smp.set_array([min(correlations), max(correlations)])
cbar = plt.colorbar(smp, ax=ax, label='Correlação')

# Exibição do gráfico
plt.show()

# In[2.4]: Matriz de correlações mais elaborada, com uso da função 'rcorr' do
#pacote 'pingouin'

correlation_matrix2 = pg.rcorr(df_paises, method='pearson',
                               upper='pval', decimals=6,
                               pval_stars={0.01: '***',
                                           0.05: '**',
                                           0.10: '*'})
correlation_matrix2

# In[2.5]: Gráfico com distribuições das variáveis, scatters, valores das
#correlações e respectivas significâncias estatísticas

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.3f}".format(r),
                xy=(.30, .9), xycoords=ax.transAxes, fontsize=14)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.30, .8), xycoords=ax.transAxes, fontsize=14)

# Configuração do gráfico
sns.set(style="whitegrid", palette="viridis")

plt.figure(figsize=(20,10))
graph = sns.pairplot(df_paises, diag_kind="kde",
                     plot_kws={"color": "darkorchid"},
                     height=2.5, aspect=1.7)
graph.map(corrfunc)
for ax in graph.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=14)
    ax.set_ylabel(ax.get_ylabel(), fontsize=14)
plt.show()

# In[2.6]: Estimação de um modelo de regressão múltipla com as variáveis do
#dataframe 'df_paises'

# Estimando o modelo de regressão múltipla por OLS
modelo_paises = sm.OLS.from_formula("cpi ~ idade + horas", df_paises).fit()

# Parâmetros do 'modelo_paises'
modelo_paises.summary()

# Cálculo do R² ajustado (slide 32 da apostila)
r2_ajust = 1-((len(df_paises.index)-1)/(len(df_paises.index)-\
                                          modelo_paises.params.count()))*\
    (1-modelo_paises.rsquared)

float(r2_ajust) # modo direto: modelo_paises.rsquared_adj

# In[2.7]: Salvando os fitted values na base de dados

df_paises['cpifit'] = modelo_paises.fittedvalues
df_paises

# In[2.8]: Gráfico 3D com scatter e fitted values (superfície espacial)
#resultantes do 'modelo_paises', gerado em HTML e aberto no browser
#(figura 'EXEMPLO2_scatter3D_fitted.html' salva na pasta do curso)

# Dados observados ('scatter')
scatter = go.Scatter3d(
    x=df_paises['horas'],
    y=df_paises['idade'],
    z=df_paises['cpi'],
    mode='markers',
    marker=dict(
        size=10,
        color='darkorchid',
        opacity=0.7
    ),
    name='Observado'
)

# Ajuste do modelo
X = np.column_stack([
    np.ones(len(df_paises)),
    df_paises['horas'],
    df_paises['idade']
])

y = df_paises['cpi'].values

b0, b1, b2 = np.linalg.lstsq(X, y, rcond=None)[0]

# Definição da grade ('meshgrid')
horas_grid = np.linspace(df_paises['horas'].min(),
                          df_paises['horas'].max(), 50)

idade_grid = np.linspace(df_paises['idade'].min(),
                          df_paises['idade'].max(), 50)

H, I = np.meshgrid(horas_grid, idade_grid)

# Definição da superfície ('go.Surface')
Z = b0 + b1 * H + b2 * I

surface = go.Surface(
    x=horas_grid,
    y=idade_grid,
    z=Z,
    colorscale='Oranges',
    opacity=0.6,
    showscale=False,
    name='Superfície ajustada'
)

# Figura final
fig = go.Figure(data=[scatter, surface])

fig.update_layout(
    width=800,
    height=800,
    margin=dict(l=0, r=0, b=0, t=0),
    plot_bgcolor='white',
    scene=dict(
        xaxis=dict(
            title='horas',
            gridcolor='rgb(200,200,200)',
            backgroundcolor='whitesmoke'
        ),
        yaxis=dict(
            title='idade',
            gridcolor='rgb(200,200,200)',
            backgroundcolor='whitesmoke'
        ),
        zaxis=dict(
            title='cpi',
            gridcolor='rgb(200,200,200)',
            backgroundcolor='whitesmoke'
        )
    )
)

# Abre o arquivo HTML no browser
import webbrowser
fig.write_html('EXEMPLO2_scatter3D_fitted.html')
webbrowser.open('EXEMPLO2_scatter3D_fitted.html')


# In[EXEMPLO 3]:
#############################################################################
#         REGRESSÃO COM UMA VARIÁVEL EXPLICATIVA (X) QUALITATIVA            #
#               EXEMPLO 3 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################

df_corrupcao = pd.read_csv('corrupcao.csv',delimiter=',',encoding='utf-8')
df_corrupcao

# Características das variáveis do dataset
df_corrupcao.info()

# Estatísticas univariadas
df_corrupcao.describe()

# Estatísticas univariadas por região
df_corrupcao.groupby('regiao').describe()

# In[3.1]: Tabela de frequências da variável 'regiao'

# Função 'value_counts' do pacote 'pandas' sem e com o argumento 'normalize'
#para gerar, respectivamente, as contagens e os percentuais
contagem = df_corrupcao['regiao'].value_counts(dropna=False)
percent = df_corrupcao['regiao'].value_counts(dropna=False, normalize=True)
pd.concat([contagem, percent], axis=1, keys=['contagem', '%'], sort=False)

# In[3.2]: Conversão dos dados de 'regiao' para dados numéricos, a fim de
#se mostrar a estimação de modelo com o problema da ponderação arbitrária

df_corrupcao['regiao_numerico'] = LabelEncoder().fit_transform(df_corrupcao['regiao'])
df_corrupcao['regiao_numerico'] = df_corrupcao['regiao_numerico'] + 1
df_corrupcao.head(10)

# A nova variável 'regiao_numerico' é quantitativa (ERRO!), fato que
#caracteriza a ponderação arbitrária!
df_corrupcao['regiao_numerico'].info()
df_corrupcao.describe()

# In[3.3]: Modelando com a variável preditora numérica, resultando na
#estimação ERRADA dos parâmetros
# PONDERAÇÃO ARBITRÁRIA!
modelo_corrupcao_errado = sm.OLS.from_formula("cpi ~ regiao_numerico",
                                              df_corrupcao).fit()

# Parâmetros do 'modelo_corrupcao_errado'
modelo_corrupcao_errado.summary()

# In[3.4]: Plotando os fitted values do 'modelo_corrupcao_errado' considerando,
#PROPOSITALMENTE, a ponderação arbitrária, ou seja, assumindo que as regiões
#representam valores numéricos (América do Sul = 1; Ásia = 2; EUA e Canadá = 3;
#Europa = 4; Oceania = 5).

plt.figure(figsize=(15,10))

ax =sns.regplot(
    data=df_corrupcao,
    x="regiao_numerico", y="cpi",
    scatter_kws={"s": 200, "color": "darkorange", "alpha": 0.5},
    line_kws={"color": "indigo"}
)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        offset = 0
        while ax.texts:
            overlapping = False
            for text in ax.texts:
                overlapping |= text.get_position()[0] == (point['x'] + 0.05) and\
                    text.get_position()[1] == (point['y'] - 0.05 + offset)
            if overlapping:
                offset += 0.15
            else:
                break
        ax.annotate(str(point['val']) + " " + str(point['y']),
                    (point['x'] + 0.05,
                     point['y'] - 0.05 + offset),
                    fontsize=11)
                
plt.title('Resultado da Ponderação Arbitrária', fontsize=20)
plt.xlabel('Região', fontsize=17)
plt.ylabel('Corruption Perception Index', fontsize=17)
plt.xticks(range(1, 6, 1), fontsize=14)
plt.yticks(range(0, 11, 1), fontsize=14)
label_point(x = df_corrupcao['regiao_numerico'],
            y = df_corrupcao['cpi'],
            val = df_corrupcao['pais'],
            ax = plt.gca())
plt.show()

# In[3.5]: Dummizando a variável 'regiao'. O código abaixo automaticamente fará:
# a) a definição de dummies que representarão cada uma das regiões do dataset;
# b) a remoção da variável original a partir da qual houve a dummização;
# c) o estabelecimento da primeira categoria como referência, ou seja, a
#categoria 'America_do_sul', por meio do argumento 'drop_first=True'.

df_corrupcao_dummies = pd.get_dummies(df_corrupcao, columns=['regiao'],
                                      dtype=int,
                                      drop_first=True)

df_corrupcao_dummies

# A variável 'regiao' estava inicialmente definida como 'object' no dataframe
#original 'df_corrupcao'
df_corrupcao['regiao'].info()

# Este procedimento de dummização também poderia ter sido realizado em uma
#variável do tipo 'category' ou 'string'!

# In[3.6]: Estimação do modelo de regressão múltipla com n-1 dummies

modelo_corrupcao_dummies = sm.OLS.from_formula("cpi ~ regiao_Asia + \
                                              regiao_EUA_e_Canada + \
                                              regiao_Europa + \
                                              regiao_Oceania",
                                              df_corrupcao_dummies).fit()

# Parâmetros do 'modelo_corrupcao_dummies'
modelo_corrupcao_dummies.summary()

# In[3.7]: Outro método de estimação (sugestão de uso para muitas dummies
#no dataset)

# Definição da fórmula utilizada no modelo
lista_colunas =list(df_corrupcao_dummies.drop(columns=['cpi','pais',
                                                        'regiao_numerico']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "cpi ~ " + formula_dummies_modelo
print("Fórmula utilizada: ",formula_dummies_modelo)

# Estimação
modelo_corrupcao_dummies = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_corrupcao_dummies).fit()

# Parâmetros do 'modelo_corrupcao_dummies'
modelo_corrupcao_dummies.summary()

# In[3.8]: Plotando o 'modelo_corrupcao_dummies' de forma interpolada

# Fitted values do 'modelo_corrupcao_dummies' no dataset 'df_corrupcao_dummies'
df_corrupcao_dummies['fitted'] = modelo_corrupcao_dummies.fittedvalues
df_corrupcao_dummies

# In[3.9]: Gráfico propriamente dito

from scipy import interpolate

plt.figure(figsize=(15, 10))

df2 = df_corrupcao_dummies[['regiao_numerico',
                            'fitted']].groupby(['regiao_numerico']).\
    median().reset_index()
x = df2['regiao_numerico']
y = df2['fitted']

tck = interpolate.splrep(x, y, k=2)
xnew = np.arange(1, 5, 0.1)
ynew = interpolate.splev(xnew, tck, der=0)

def label_point(x, y, val, ax):
    a = pd.concat({'x': x, 'y': y, 'val': val}, axis=1)
    for i, point in a.iterrows():
        offset = 0
        while ax.texts:
            overlapping = False
            for text in ax.texts:
                overlapping |= text.get_position()[0] == (point['x'] + 0.05) and\
                    text.get_position()[1] == (point['y'] - 0.05 + offset)
            if overlapping:
                offset += 0.15
            else:
                break
        ax.annotate(str(point['val']) + " " + str(point['y']),
                    (point['x'] + 0.05,
                     point['y'] - 0.05 + offset),
                    fontsize=11)

plt.scatter(df_corrupcao_dummies['regiao_numerico'],
            df_corrupcao_dummies['cpi'], color='darkorange', s=200, alpha=0.5)
plt.scatter(df_corrupcao_dummies['regiao_numerico'],
            df_corrupcao_dummies['fitted'], color='limegreen', s=240)
plt.plot(xnew, ynew, color='indigo', linewidth=2.5)
plt.title('Ajuste Não Linear do Modelo com Variáveis Dummy', fontsize=20)
plt.xlabel('Região', fontsize=17)
plt.ylabel('Corruption Perception Index', fontsize=17)
plt.xticks(range(1, 6, 1), fontsize=14)
plt.yticks(range(0, 11, 1), fontsize=14)
label_point(x=df_corrupcao_dummies['regiao_numerico'],
            y=df_corrupcao_dummies['cpi'],
            val=df_corrupcao_dummies['pais'],
            ax=plt.gca())
plt.show()

# In[3.10]: Gráfico gerado em HTML e aberto no browser, com interação
#(figura 'EXEMPLO3.html' salva na pasta do curso)

df2 = df_corrupcao_dummies[['regiao_numerico',
                            'fitted']].groupby(['regiao_numerico']).\
    median().reset_index()
x = df2['regiao_numerico']
y = df2['fitted']

tck = interpolate.splrep(x, y, k=2)
xnew = np.arange(1, 5, 0.1)
ynew = interpolate.splev(xnew, tck, der=0)

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=df_corrupcao_dummies['regiao_numerico'],
    y=df_corrupcao_dummies['cpi'],
    mode='markers',
    name='CPI',
    marker=dict(color='darkorange', size=14, opacity=0.5)
))

fig.add_trace(go.Scatter(
    x=df_corrupcao_dummies['regiao_numerico'],
    y=df_corrupcao_dummies['fitted'],
    mode='markers',
    name='Fitted',
    marker=dict(color='limegreen', size=17)
))

fig.add_trace(go.Scatter(
    x=xnew,
    y=ynew,
    mode='lines',
    name='Interpolated',
    line=dict(color='indigo', width=3.5)
))

fig.update_layout(title={
        'text': 'Ajuste Não Linear do Modelo com Variáveis Dummy',
        'font': {'size': 20, 'color': 'black', 'family': 'Arial'},
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis=dict(title='Região'),
    yaxis=dict(title='Corruption Perception Index'),
    xaxis_tickvals=list(range(1, 6)),
    yaxis_tickvals=list(range(0, 11)),
    xaxis_tickfont=dict(size=14),
    yaxis_tickfont=dict(size=14),
    template='plotly_white')

for i in range(len(df_corrupcao_dummies)):
    fig.add_annotation(
        x=df_corrupcao_dummies['regiao_numerico'][i],
        y=df_corrupcao_dummies['cpi'][i],
        text=str(df_corrupcao_dummies['pais'][i]) + ' ' + str(df_corrupcao_dummies['cpi'][i]),
        showarrow=False,
        font=dict(size=11, color='black'),
        xshift=50,
        yshift=0,
        textangle=0
    )

fig.update_annotations(dict(xref="x", yref="y"))
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

fig.write_html('EXEMPLO3.html')

# Abre o arquivo HTML no browser
import webbrowser
webbrowser.open('EXEMPLO3.html')


# In[EXEMPLO 4]:
#############################################################################
#            REGRESSÃO NÃO LINEAR E TRANSFORMAÇÃO DE BOX-COX                #
#               EXEMPLO 4 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################

df_bebes = pd.read_csv('bebes.csv', delimiter=',')
df_bebes

# Características das variáveis do dataset
df_bebes.info()

# Estatísticas univariadas
df_bebes.describe()

# In[4.1]: Gráfico de dispersão

plt.figure(figsize=(15,10))
sns.scatterplot(x="idade", y="comprimento", data=df_bebes, color='grey',
                s=300, label='Valores Reais', alpha=0.7)
plt.title('Dispersão dos dados', fontsize=20)
plt.xlabel('Idade em semanas', fontsize=17)
plt.ylabel('Comprimento em cm', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# In[4.2]: Gráfico de dispersão com emojis 01

plt.figure(figsize=(15, 10))
plt.scatter(x="idade", y="comprimento", data=df_bebes, color='grey',
            s=400, label='Valores Reais', alpha=0.7, marker='$\U0001F607$',
            linewidth=0.2)
plt.title('Dispersão dos dados', fontsize=20)
plt.xlabel('Idade em semanas', fontsize=17)
plt.ylabel('Comprimento em cm', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# In[4.3]: Gráfico de dispersão com emojis 02

emojis_coracao = ['♥'] * len(df_bebes)

plt.figure(figsize=(15, 10))
sns.scatterplot(x="idade", y="comprimento", data=df_bebes, color='none',
                s=0, label=None)
for i, emoji in enumerate(emojis_coracao):
    plt.text(df_bebes['idade'][i], df_bebes['comprimento'][i], emoji,
             fontsize=40, ha='center', alpha=0.6, color='darkorchid')
plt.title('Dispersão dos dados', fontsize=20)
plt.xlabel('Idade em semanas', fontsize=17)
plt.ylabel('Comprimento em cm', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# In[4.4]: Estimação de um modelo OLS linear
modelo_linear = sm.OLS.from_formula('comprimento ~ idade', df_bebes).fit()

# Parâmetros do 'modelo_linear'
modelo_linear.summary()

# In[4.5]: Gráfico de dispersão com ajustes (fits) linear e não linear
# com argumento 'lowess=True' (locally weighted scatterplot smoothing)

plt.figure(figsize=(15,10))
sns.scatterplot(x="idade", y="comprimento", data=df_bebes, color='grey',
                s=300, label='Valores Reais', alpha=0.7)
sns.regplot(x="idade", y="comprimento", data=df_bebes, lowess=True,
            color='darkviolet', ci=False, scatter=False,
            label='Ajuste Não Linear',
            line_kws={'linewidth': 2.5})
sns.regplot(x="idade", y="comprimento", data=df_bebes,
            color='darkorange', ci=False, scatter=False,
            label='OLS Linear',
            line_kws={'linewidth': 2.5})
plt.title('Dispersão dos dados e ajustes linear e não linear', fontsize=20)
plt.xlabel('Idade em semanas', fontsize=17)
plt.ylabel('Comprimento em cm', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# In[4.6]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
# from scipy.stats import shapiro
# shapiro(modelo_linear.resid)

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

# from statstests.tests import shapiro_francia

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_linear.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[4.7]: Histograma dos resíduos do modelo OLS linear

plt.figure(figsize=(15,10))
hist1 = sns.histplot(data=modelo_linear.resid, kde=True, bins=25,
                     color = 'darkorange', alpha=0.4, edgecolor='silver',
                     line_kws={'linewidth': 3})
hist1.get_lines()[0].set_color('orangered')
plt.xlabel('Resíduos', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

# In[4.8]: Transformação de Box-Cox

# Para o cálculo do lambda de Box-Cox
# from scipy.stats import boxcox

# 'yast' é uma variável que traz os valores transformados (Y*)
# 'lmbda' é o lambda de Box-Cox
yast, lmbda = boxcox(df_bebes['comprimento'])

# Inserção da variável transformada ('bc_comprimento') no dataset para a
#estimação de um novo modelo
df_bebes['bc_comprimento'] = yast

df_bebes

# Verificação do cálculo, apenas para fins didáticos
df_bebes['bc_comprimento2'] = ((df_bebes['comprimento']**lmbda)-1)/lmbda

df_bebes

del df_bebes['bc_comprimento2']

# In[4.9]: Estimando um novo modelo OLS com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula('bc_comprimento ~ idade', df_bebes).fit()

# Parâmetros do 'modelo_bc'
modelo_bc.summary()

# In[4.10]: Comparando os parâmetros do 'modelo_linear' com os do 'modelo_bc'

# CUIDADO!!! OS PARÂMETROS NÃO SÃO DIRETAMENTE COMPARÁVEIS!

summary_col([modelo_linear, modelo_bc])

# Outro modo mais completo também pela função 'summary_col'
summary_col([modelo_linear, modelo_bc],
            model_names=["MODELO LINEAR","MODELO BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })

# Repare que há um salto na qualidade do ajuste para o modelo não linear (R²)

pd.DataFrame({'R² OLS':[round(modelo_linear.rsquared,4)],
              'R² Box-Cox':[round(modelo_bc.rsquared,4)]})

# In[4.11]: Verificando a normalidade dos resíduos do 'modelo_bc'

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_bc.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[4.12]: Histograma dos resíduos do modelo_bc

plt.figure(figsize=(15,10))
hist2 = sns.histplot(data=modelo_bc.resid, kde=True, bins=25,
                     color='darkviolet', alpha=0.4, edgecolor='snow',
                     line_kws={'linewidth': 3})
hist2.get_lines()[0].set_color('indigo')
plt.xlabel('Resíduos', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.show()

# In[4.13]: Fazendo predições com os modelos OLS linear e Box-Cox
# Qual é o comprimento esperado de um bebê com 52 semanas de vida?

# Modelo OLS Linear:
modelo_linear.predict(pd.DataFrame({'idade':[52]}))

# Modelo Não Linear (Box-Cox):
modelo_bc.predict(pd.DataFrame({'idade':[52]}))

# Não podemos nos esquecer de fazer o cálculo inverso para a obtenção do fitted
#value de Y (variável 'comprimento')
float((54251.133574 * lmbda + 1) ** (1 / lmbda))

# In[4.14]: Salvando os fitted values dos dois modelos (modelo_linear
#e modelo_bc) no dataset 'bebes'

df_bebes['yhat_linear'] = modelo_linear.fittedvalues
df_bebes['yhat_modelo_bc'] = (modelo_bc.fittedvalues * lmbda + 1) ** (1 / lmbda)
df_bebes

# In[4.15]: Gráfico de dispersão com ajustes dos modelos OLS linear e Box-Cox

plt.figure(figsize=(15,10))
sns.scatterplot(x="idade", y="comprimento", data=df_bebes, color='grey',
                s=350, label='Valores Reais', alpha=0.7)
sns.regplot(x="idade", y="yhat_modelo_bc", data=df_bebes, order=lmbda,
            color='darkviolet', ci=False, scatter=False, label='Box-Cox',
            line_kws={'linewidth': 2.5})
sns.scatterplot(x="idade", y="yhat_modelo_bc", data=df_bebes, color='darkviolet',
                s=200, label='Fitted Values Box-Cox', alpha=0.5)
sns.regplot(x="idade", y="yhat_linear", data=df_bebes,
            color='darkorange', ci=False, scatter=False, label='OLS Linear',
            line_kws={'linewidth': 2.5})
sns.scatterplot(x="idade", y="yhat_linear", data=df_bebes, color='darkorange',
                s=200, label='Fitted Values OLS Linear', alpha=0.5)
plt.title('Dispersão dos dados e ajustes dos modelos OLS linear e Box-Cox',
          fontsize=20)
plt.xlabel('Idade em semanas', fontsize=17)
plt.ylabel('Comprimento em cm', fontsize=17)
plt.legend(loc='lower right', fontsize=17)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()

# In[4.16]: Gráfico de dispersão com ajustes dos modelos OLS linear e Box-Cox,
#com interação (figura 'EXEMPLO4.html' salva na pasta do curso)

# Ajuste polinomial com grau igual a lambda (lmbda = 2.659051008426254)
coefficients = np.polyfit(df_bebes["idade"], df_bebes["yhat_modelo_bc"], lmbda)
x_range = np.linspace(df_bebes["idade"].min(), df_bebes["idade"].max(), 100)
y_quadratic = np.polyval(coefficients, x_range)

fig = go.Figure()

fig.add_trace(go.Scatter(x=df_bebes["idade"], y=df_bebes["comprimento"],
                         mode='markers',
                         marker=dict(color='grey', opacity=0.7, size=20),
                         name='Valores Reais'))

fig.add_trace(go.Scatter(x=x_range, y=y_quadratic,
                         mode='lines',
                         line=dict(color='darkviolet', width=2),
                         name='Box-Cox'))

fig.add_trace(go.Scatter(x=df_bebes["idade"], y=df_bebes["yhat_modelo_bc"],
                         mode='markers',
                         marker=dict(color='darkviolet', opacity=0.5, size=15),
                         name='Fitted Values Box-Cox',
                         hovertemplate='Fitted Values Box-Cox: %{y:.2f}<extra></extra>'))

fig.add_trace(go.Scatter(x=df_bebes["idade"], y=df_bebes["yhat_linear"],
                         mode='lines',
                         marker=dict(color='darkorange'),
                         name='OLS Linear',
                         hovertemplate='Fitted Values OLS Linear: %{y:.2f}<extra></extra>'))

fig.add_trace(go.Scatter(x=df_bebes["idade"], y=df_bebes["yhat_linear"],
                         mode='markers',
                         marker=dict(color='darkorange', opacity=0.5, size=15),
                         name='Fitted Values OLS Linear',
                         hovertemplate='Fitted Values OLS Linear: %{y:.2f}<extra></extra>'))

fig.update_layout(title={
        'text': 'Dispersão dos dados e ajustes dos modelos OLS linear e Box-Cox',
        'font': {'size': 20, 'color': 'black', 'family': 'Arial'},
        'x': 0.5,
        'y': 0.95,
        'xanchor': 'center',
        'yanchor': 'top'
    },
    xaxis_title='Idade em semanas',
    yaxis_title='Comprimento em cm',
    legend=dict(x=1.02, y=1),
    template='plotly_white')

fig.update_annotations(dict(xref="x", yref="y"))
fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')

fig.write_html("EXEMPLO4.html")

# Abre o arquivo HTML no browser
import webbrowser
webbrowser.open('EXEMPLO4.html')


# In[EXEMPLO 5]:
#############################################################################
#                        REGRESSÃO NÃO LINEAR MÚLTIPLA                      #
#                  EXEMPLO 5 - CARREGAMENTO DA BASE DE DADOS                #
#############################################################################

df_empresas = pd.read_csv('empresas.csv', delimiter=',')
df_empresas

# Características das variáveis do dataset
df_empresas.info()

# Estatísticas univariadas
df_empresas.describe()

# In[5.1]: Matriz de correlações

correlation_matrix = df_empresas.iloc[:,1:6].corr()
correlation_matrix

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 25}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=15)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=15)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[5.2]: Matriz de correlações
# Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'

import pingouin as pg

correlation_matrix2 = pg.rcorr(df_empresas, method='pearson',
                              upper='pval', decimals=4,
                              pval_stars={0.01: '***',
                                          0.05: '**',
                                          0.10: '*'})
correlation_matrix2

# In[5.3]: Diagrama interessante (grafo) que mostra a inter-relação entre as
#variáveis e a magnitude das correlações entre elas

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Criação de um grafo direcionado
G = nx.DiGraph()

# Adição das variáveis como nós do grafo
for variable in correlation_matrix.columns:
    G.add_node(variable)

# Adição das arestas com espessuras proporcionais às correlações
for i, variable1 in enumerate(correlation_matrix.columns):
    for j, variable2 in enumerate(correlation_matrix.columns):
        if i != j:
            correlation = correlation_matrix.iloc[i, j]
            if abs(correlation) > 0:
                G.add_edge(variable1, variable2, weight=correlation)

# Obtenção da lista de correlações das arestas
correlations = [d["weight"] for _, _, d in G.edges(data=True)]

# Definição da dimensão dos nós
node_size = 2700

# Definição da cor dos nós
node_color = 'black'

# Definição da escala de cores das retas (correspondência com as correlações)
cmap = plt.colormaps.get_cmap('coolwarm_r')

# Criação de uma lista de espessuras das arestas proporcional às correlações
edge_widths = [abs(d["weight"]) * 10 for _, _, d in G.edges(data=True)]

# Criação do layout do grafo com maior distância entre os nós
pos = nx.spring_layout(G, k=0.75)  # k para controlar a distância entre os nós

# Ajuste das posições dos nós das variáveis
pos["retorno"] = (pos["retorno"][0] + 1.8, pos["retorno"][1] + 1.8)
pos["disclosure"] = (pos["disclosure"][0], pos["disclosure"][1] + 1.8)
pos["endividamento"] = (pos["endividamento"][0], pos["endividamento"][1] + 1.8)
pos["ativos"] = (pos["ativos"][0], pos["ativos"][1])
pos["liquidez"] = (pos["liquidez"][0], pos["liquidez"][1] + 1.8)

# Desenho dos nós e das arestas com base nas correlações e espessuras
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                       edge_cmap=cmap, alpha=0.7)

# Adição dos rótulos dos nós
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=7.5, font_color='white')

# Ajuste dos limites dos eixos
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")

# Criação da legenda com a escala de cores definida
smp = cm.ScalarMappable(cmap=cmap)
smp.set_array([min(correlations), max(correlations)])
cbar = plt.colorbar(smp, ax=ax, label='Correlação')

# Definição dos ticks da colorbar
cbar.set_ticks(np.arange(round(min(correlations),0) - 0.1,
                         max(correlations) + 0.1, 0.1))

# Exibição do gráfico
plt.show()

# In[5.4]: Distribuições das variáveis, scatters, valores das correlações e
#suas respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.3f}".format(r),
                xy=(.30, .9), xycoords=ax.transAxes, fontsize=16)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.30, .8), xycoords=ax.transAxes, fontsize=16)

# Configuração do gráfico
sns.set(style="whitegrid", palette="viridis")

plt.figure(figsize=(20,10))
graph = sns.pairplot(df_empresas, diag_kind="kde",
                     plot_kws={"color": "darkorchid"},
                     height=2.5, aspect=1.7)
graph.map(corrfunc)
for ax in graph.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=17)
    ax.set_ylabel(ax.get_ylabel(), fontsize=17)
plt.show()

# In[5.5]: Estimando o Modelo de Regressão Múltipla
modelo_empresas = sm.OLS.from_formula('retorno ~ disclosure +\
                                      endividamento + ativos +\
                                          liquidez', df_empresas).fit()

# Parâmetros do 'modelo_empresas'
modelo_empresas.summary()

# Note que o parâmetro da variável 'endividamento' não é estatisticamente
#significante ao nível de significância de 5% (nível de confiança de 95%).

# In[5.6]: Procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

# from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_empresas = stepwise(modelo_empresas, pvalue_limit=0.05)

# In[5.7]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Wilk (n < 30)
#from scipy.stats import shapiro
#shapiro(modelo_step_empresas.resid)

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

# from statstests.tests import shapiro_francia

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step_empresas.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[5.8]: Plotando os resíduos do 'modelo_step_empresas' e acrescentando
#uma curva normal teórica para comparação entre as distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

# from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_step_empresas.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_step_empresas.resid, bins=20, kde=True, stat="density",
             color='darkorange', alpha=0.4)
plt.xlim(-20, 20)
x = np.linspace(-20, 20, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Linear', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(['Distribuição Real','Distribuição Normal Teórica','Histograma'],
           fontsize=18)
plt.show()

# In[5.9]: Transformação de Box-Cox

# Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

# 'yast' é uma variável que traz os valores transformados (Y*)
# 'lmbda' é o lambda de Box-Cox
yast, lmbda = boxcox(df_empresas['retorno'])

print("Lambda: ",lmbda)

# In[5.10]: Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo modelo

df_empresas['bc_retorno'] = yast
df_empresas

# Verificação do cálculo, apenas para fins didáticos
df_empresas['bc_retorno2'] = ((df_empresas['retorno'])**(lmbda) - 1) / (lmbda)
df_empresas

del df_empresas['bc_retorno2']

# In[5.11]: Estimando um novo modelo múltiplo com variável dependente
#transformada por Box-Cox

modelo_bc = sm.OLS.from_formula('bc_retorno ~ disclosure +\
                                endividamento + ativos +\
                                    liquidez', df_empresas).fit()

# Parâmetros do 'modelo_bc'
modelo_bc.summary()

# In[5.12]: Aplicando o procedimento Stepwise no 'modelo_bc"

modelo_step_empresas_bc = stepwise(modelo_bc, pvalue_limit=0.05)

# Note que a variável 'disclosure' retorna ao modelo na forma funcional
#não linear!

# In[5.13]: Verificando a normalidade dos resíduos do 'modelo_step_empresas_bc'

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step_empresas_bc.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[5.14]: Plotando os novos resíduos do 'modelo_step_empresas_bc' e
#acrescentando uma curva normal teórica para comparação entre as distribuições

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_step_empresas_bc.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_step_empresas_bc.resid, bins=20, kde=True, stat="density",
             color='indigo', alpha=0.4)
plt.xlim(-0.5, 0.5)
x = np.linspace(-0.5, 0.5, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Box-Cox', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(['Distribuição Real','Distribuição Normal Teórica','Histograma'],
           fontsize=18)
plt.show()

# In[5.15]: Resumo dos dois modelos obtidos pelo procedimento Stepwise
#(linear e com Box-Cox)

summary_col([modelo_step_empresas, modelo_step_empresas_bc],
            model_names=["STEPWISE","STEPWISE BOX-COX"],
            stars=True,
            info_dict = {
                'N':lambda x: "{0:d}".format(int(x.nobs))
        })

# CUIDADO!!! OS PARÂMETROS NÃO SÃO DIRETAMENTE COMPARÁVEIS!

# In[5.16]: Fazendo predições com o 'modelo_step_empresas_bc'
# Qual é o valor do retorno, em média, para 'disclosure' igual a 50,
#'liquidez' igual a 14 e 'ativos' igual a 4000, ceteris paribus?

modelo_step_empresas_bc.predict(pd.DataFrame({'const':[1],
                                              'disclosure':[50],
                                              'ativos':[4000],
                                              'liquidez':[14]}))


# In[5.17]: Não podemos nos esquecer de fazer o cálculo para a obtenção do
#fitted value de Y (variável 'retorno')

float((3.702015 * lmbda + 1) ** (1 / lmbda))


# In[5.18]: Salvando os fitted values de 'modelo_step_empresas' e
#'modelo_step_empresas_bc'

df_empresas['yhat_step_empresas'] = modelo_step_empresas.fittedvalues
df_empresas['yhat_step_empresas_bc'] = (modelo_step_empresas_bc.fittedvalues
                                        * lmbda + 1) ** (1 / lmbda)

# Visualizando os dois fitted values dos modelos 'modelo_step_empresas' e
#'modelo_step_empresas_bc' no dataset
df_empresas[['empresa','retorno','yhat_step_empresas','yhat_step_empresas_bc']]

# In[5.19]: Ajustes dos modelos: valores previstos (fitted values) X valores
#reais

from scipy.optimize import curve_fit

def objective(x, a, b, c, d, e, f):
    return (a * x) + (b * x**2) + (c * x**3) + (d * x**4) + (e * x**5) + f

xdata = df_empresas['retorno']
ydata_linear = df_empresas['yhat_step_empresas']
ydata_bc = df_empresas['yhat_step_empresas_bc']

plt.figure(figsize=(17,10))

popt, _ = curve_fit(objective, xdata, ydata_linear)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='darkorange', linewidth=3)

popt, _ = curve_fit(objective, xdata, ydata_bc)
a, b, c, d, e, f = popt
x_line = np.arange(min(xdata), max(xdata), 1)
y_line = objective(x_line, a, b, c, d, e, f)
plt.plot(x_line, y_line, '--', color='indigo', linewidth=3)

plt.plot(xdata,xdata, color='gray', linestyle='-')
plt.scatter(xdata,ydata_linear, alpha=0.5, s=150, color='darkorange')
plt.scatter(xdata,ydata_bc, alpha=0.5, s=150, color='indigo')
plt.title('Dispersão e Fitted Values dos Modelos Linear e Box-Cox',
          fontsize=20)
plt.xlabel('Valores Reais de Retorno', fontsize=17)
plt.ylabel('Fitted Values', fontsize=17)
plt.legend(['Stepwise','Stepwise com Box-Cox','45º graus'], fontsize=18)
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)
plt.show()


# In[EXEMPLO 6]:
#############################################################################
#         DIAGNÓSTICO DE MULTICOLINEARIDADE EM MODELOS DE REGRESSÃO         #
#                EXEMPLO 6 - CARREGAMENTO DA BASE DE DADOS                  #
#############################################################################

df_salarios = pd.read_csv('salarios.csv', delimiter=',')
df_salarios

# Características das variáveis do dataset
df_salarios.info()

# Estatísticas univariadas
df_salarios.describe()

# In[6.1]: Matriz de correlações

correlation_matrix = df_salarios.iloc[:,1:6].corr()
correlation_matrix

# Mapa de calor com as correlações entre todas as variáveis quantitativas
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(correlation_matrix, annot=True, fmt=".3f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 20}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=14)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=14)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[6.2]: CORRELAÇÃO BAIXA (variáveis 'rh1' e 'econometria1'):

# Correlação entre 'rh1' e 'econometria1', com p-value
corr1, p_value1 = pearsonr(df_salarios['rh1'], df_salarios['econometria1'])
"{:.4f}".format(corr1), "{:.4f}".format(p_value1)

# Matriz de correlação (maneira simples) pela função 'corr'
corr1 = df_salarios[['rh1','econometria1']].corr()
corr1

# Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'
import pingouin as pg
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

corr1b = pg.rcorr(df_salarios[['rh1','econometria1']], method='pearson',
                  upper='pval', decimals=6,
                  pval_stars={0.01: '***',
                              0.05: '**',
                              0.10: '*'})
corr1b

# Mapa de calor com a correlação entre 'rh1' e 'econometria1'
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(corr1, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 30}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[6.3]: Grafo com a inter-relação entre as variáveis do dataframe 'df1'

df1 = df_salarios[['salario','rh1','econometria1']]
cormat1 = df1.corr()
cormat1

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Criação de um grafo direcionado
G = nx.DiGraph()

# Adição das variáveis como nós do grafo
for variable in cormat1.columns:
    G.add_node(variable)

# Adição das arestas com espessuras proporcionais às correlações
for i, variable1 in enumerate(cormat1.columns):
    for j, variable2 in enumerate(cormat1.columns):
        if i != j:
            correlation = cormat1.iloc[i, j]
            if abs(correlation) > 0:
                G.add_edge(variable1, variable2, weight=correlation)

# Obtenção da lista de correlações das arestas
correlations = [d["weight"] for _, _, d in G.edges(data=True)]

# Definição da dimensão dos nós
node_size = 2700

# Definição da cor dos nós
node_color = 'black'

# Definição da escala de cores das retas (correspondência com as correlações)
cmap = plt.colormaps.get_cmap('viridis_r')

# Criação de uma lista de espessuras das arestas proporcional às correlações
edge_widths = [abs(d["weight"]) * 10 for _, _, d in G.edges(data=True)]

# Criação do layout do grafo com maior distância entre os nós
pos = nx.spring_layout(G, k=0.75)  # k para controlar a distância entre os nós

# Desenho dos nós e das arestas com base nas correlações e espessuras
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                       edge_cmap=cmap, alpha=0.9)

# Adição dos rótulos dos nós
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=7.5, font_color='white')

# Ajuste dos limites dos eixos
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")

# Criação da legenda com a escala de cores definida
smp = cm.ScalarMappable(cmap=cmap)
smp.set_array([min(correlations), max(correlations)])
cbar = plt.colorbar(smp, ax=ax, label='Correlação')

# Definição dos ticks da colorbar
cbar.set_ticks(np.arange(round(min(correlations),1),
                         max(correlations), 0.1))

# Exibição do gráfico
plt.show()

# In[6.4]: Modelo 1

modelo1 = sm.OLS.from_formula('salario ~ rh1 + econometria1', df_salarios).fit()

modelo1.summary()

# In[6.5]: Diagnóstico de multicolinearidade (Variance Inflation Factor
#e Tolerance)

from statsmodels.stats.outliers_influence import variance_inflation_factor

# Calculando os valores de VIF
X1 = sm.add_constant(df_salarios[['rh1', 'econometria1']])
VIF = pd.DataFrame()
VIF["Variável"] = X1.columns[1:]
VIF["VIF"] = [variance_inflation_factor(X1.values, i+1)
              for i in range(X1.shape[1]-1)]

# Calculando as Tolerâncias
VIF["Tolerância"] = 1 / VIF["VIF"]
VIF

# In[6.6]: CORRELAÇÃO MUITO ALTA (variáveis 'rh2' e 'econometria2'):

# Correlação entre 'rh2' e 'econometria2', com p-value
corr2, p_value2 = pearsonr(df_salarios['rh2'], df_salarios['econometria2'])
"{:.4f}".format(corr2), "{:.4f}".format(p_value2)

# Matriz de correlação (maneira simples) pela função 'corr'
corr2 = df_salarios[['rh2','econometria2']].corr()
corr2

# Maneira mais elaborada pela função 'rcorr' do pacote 'pingouin'
import pingouin as pg

corr2b = pg.rcorr(df_salarios[['rh2','econometria2']], method='pearson',
                  upper='pval', decimals=6,
                  pval_stars={0.01: '***',
                              0.05: '**',
                              0.10: '*'})
corr2b

# Mapa de calor com a correlação entre 'rh2' e 'econometria2'
plt.figure(figsize=(15, 10))
heatmap = sns.heatmap(corr2, annot=True, fmt=".4f",
                      cmap=plt.cm.viridis_r,
                      annot_kws={'size': 30}, vmin=-1, vmax=1)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=17)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=17)
cbar = heatmap.collections[0].colorbar
cbar.ax.tick_params(labelsize=17)
plt.show()

# In[6.7]: Grafo com a inter-relação entre as variáveis do dataframe 'df2'

df2 = df_salarios[['salario','rh2','econometria2']]
cormat2 = df2.corr()
cormat2

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Criação de um grafo direcionado
G = nx.DiGraph()

# Adição das variáveis como nós do grafo
for variable in cormat2.columns:
    G.add_node(variable)

# Adição das arestas com espessuras proporcionais às correlações
for i, variable1 in enumerate(cormat2.columns):
    for j, variable2 in enumerate(cormat2.columns):
        if i != j:
            correlation = cormat2.iloc[i, j]
            if abs(correlation) > 0:
                G.add_edge(variable1, variable2, weight=correlation)

# Obtenção da lista de correlações das arestas
correlations = [d["weight"] for _, _, d in G.edges(data=True)]

# Definição da dimensão dos nós
node_size = 2700

# Definição da cor dos nós
node_color = 'black'

# Definição da escala de cores das retas (correspondência com as correlações)
cmap = plt.colormaps.get_cmap('viridis_r')

# Criação de uma lista de espessuras das arestas proporcional às correlações
edge_widths = [abs(d["weight"]) * 10 for _, _, d in G.edges(data=True)]

# Criação do layout do grafo com maior distância entre os nós
pos = nx.spring_layout(G, k=0.75)  # k para controlar a distância entre os nós

# Desenho dos nós e das arestas com base nas correlações e espessuras
nx.draw_networkx_nodes(G, pos, node_size=node_size, node_color=node_color)
nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=correlations,
                       edge_cmap=cmap, alpha=0.9)

# Adição dos rótulos dos nós
labels = {node: node for node in G.nodes}
nx.draw_networkx_labels(G, pos, labels, font_size=7.5, font_color='white')

# Ajuste dos limites dos eixos
ax = plt.gca()
ax.margins(0.1)
plt.axis("off")

# Criação da legenda com a escala de cores definida
smp = cm.ScalarMappable(cmap=cmap)
smp.set_array([min(correlations), max(correlations)])
cbar = plt.colorbar(smp, ax=ax, label='Correlação')

# Definição dos ticks da colorbar
cbar.set_ticks(np.arange(round(min(correlations) - 0.01,2),
                         max(correlations) + 0.01, 0.01))

# Exibição do gráfico
plt.show()

# In[6.8]: Modelo 2

modelo2 = sm.OLS.from_formula('salario ~ rh2 + econometria2', df_salarios).fit()

modelo2.summary()

# In[6.9]: Diagnóstico de multicolinearidade (Variance Inflation Factor
#e Tolerance)

# Calculando os valores de VIF
X2 = sm.add_constant(df_salarios[['rh2', 'econometria2']])
VIF = pd.DataFrame()
VIF["Variável"] = X2.columns[1:]
VIF["VIF"] = [variance_inflation_factor(X2.values, i+1)
              for i in range(X2.shape[1]-1)]

# Calculando as Tolerâncias
VIF["Tolerância"] = 1 / VIF["VIF"]
VIF


# In[EXEMPLO 7]:
#############################################################################
#        DIAGNÓSTICO DE HETEROCEDASTICIDADE EM MODELOS DE REGRESSÃO         #
#               EXEMPLO 7 - CARREGAMENTO DA BASE DE DADOS                   #
#############################################################################
    
df_saeb_rend = pd.read_csv('saeb_rend.csv', delimiter=',')
df_saeb_rend

# Características das variáveis do dataset
df_saeb_rend.info()

# Estatísticas univariadas
df_saeb_rend.describe()

# In[7.1]: Tabela de frequências absolutas das variáveis 'uf' e rede'

df_saeb_rend['uf'].value_counts().sort_index()
df_saeb_rend['rede'].value_counts().sort_index()

# In[7.2]: Plotando a variável 'saeb' em função de 'rendimento', com fit linear
# Gráfico pela função 'regplot' do 'seaborn'

plt.figure(figsize=(15,10))
sns.regplot(x='rendimento', y='saeb', data=df_saeb_rend, marker='o',
            color='royalblue', ci=False,
            scatter_kws={'color':'lightsalmon', 'alpha':0.5, 's':150},
            line_kws={'linewidth': 4})
plt.title('Gráfico de Dispersão com Ajuste Linear', fontsize=22)
plt.xlabel('rendimento', fontsize=20)
plt.ylabel('saeb', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# In[7.3]: Plotando a variável 'saeb' em função de 'rendimento', com destaque
#para a 'rede' escolar e linear fits -> Gráfico pela função 'regplot' do
#pacote 'seaborn'

# Definição de dataframes com subgrupos por 'rede'
df1 = df_saeb_rend[df_saeb_rend['rede'] == 'Municipal']
df2 = df_saeb_rend[df_saeb_rend['rede'] == 'Estadual']
df3 = df_saeb_rend[df_saeb_rend['rede'] == 'Federal']

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.regplot(x='rendimento', y='saeb', data=df1, marker='o', ci=False,
            scatter_kws={'color':'darkorange', 'alpha':0.3, 's':150},
            line_kws={'color':'darkorange', 'linewidth': 4}, label='Municipal')
sns.regplot(x='rendimento', y='saeb', data=df2, marker='o', ci=False,
            scatter_kws={'color':'darkviolet', 'alpha':0.3, 's':150},
            line_kws={'color':'darkviolet', 'linewidth': 4}, label='Estadual')
sns.regplot(x='rendimento', y='saeb', data=df3, marker='o', ci=False,
            scatter_kws={'color':'darkgreen', 'alpha':0.8, 's':150},
            line_kws={'color':'darkgreen', 'linewidth': 4}, label='Federal')
plt.title('Gráfico de Dispersão com Ajuste Linear por Rede', fontsize=22)
plt.xlabel('rendimento', fontsize=20)
plt.ylabel('saeb', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.legend(fontsize=18)
plt.show()

# In[7.4]: Estimação do modelo de regressão e diagnóstico de heterocedasticidade

# Estimando o modelo
modelo_saeb = sm.OLS.from_formula('saeb ~ rendimento', df_saeb_rend).fit()

# Parâmetros do 'modelo_saeb'
modelo_saeb.summary()

# In[7.5]: Adicionando fitted values e resíduos do 'modelo_saeb' no
# dataset 'df_saeb_rend'

df_saeb_rend['fitted'] = modelo_saeb.fittedvalues
df_saeb_rend['residuos'] = modelo_saeb.resid
df_saeb_rend

# In[7.6]: Gráfico que relaciona resíduos e fitted values do 'modelo_saeb'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted', y='residuos', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo', fontsize=20)
plt.ylabel('Resíduos do Modelo', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# In[7.7]: Histograma dos resíduos do 'modelo_saeb' com curva normal teórica
#para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_saeb.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_saeb.resid, bins=20, kde=True, stat="density",
             color='red', alpha=0.4)
plt.xlim(-4, 4)
x = np.linspace(-4, 4, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Linear', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(['Distribuição Real','Distribuição Normal Teórica','Histograma'],
           fontsize=18)
plt.show()

# In[7.8]: Função para o teste de Breusch-Pagan para a elaboração de diagnóstico
#de heterocedasticidade

# Criação da função 'breusch_pagan_test'

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return float(chisq), float(p_value)

# In[7.9]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_saeb)
# Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

# H0 do teste: ausência de heterocedasticidade.
# H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[7.10]: Procedimento n-1 dummies para as unidades federativas
    
# Dummização da variável 'uf'

df_saeb_rend_dummies = pd.get_dummies(df_saeb_rend, columns=['uf'],
                                      dtype=int,
                                      drop_first=True)

df_saeb_rend_dummies

# In[7.11]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_saeb_rend_dummies.drop(columns=['municipio',
                                                        'codigo',
                                                        'escola',
                                                        'rede',
                                                        'saeb',
                                                        'fitted',
                                                        'residuos']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "saeb ~ " + formula_dummies_modelo

# Estimação
modelo_saeb_dummies_uf = sm.OLS.from_formula(formula_dummies_modelo,
                                               df_saeb_rend_dummies).fit()

# Parâmetros do modelo 'modelo_saeb_dummies_uf'
modelo_saeb_dummies_uf.summary()

# In[7.12]: Estimação do modelo por meio do procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.process import stepwise

modelo_saeb_dummies_uf_step = stepwise(modelo_saeb_dummies_uf, pvalue_limit=0.05)

# In[7.13]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_saeb_dummies_uf_step'

breusch_pagan_test(modelo_saeb_dummies_uf_step)

# Interpretação
teste_bp = breusch_pagan_test(modelo_saeb_dummies_uf_step) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[7.14]: Adicionando fitted values e resíduos do 'modelo_saeb_dummies_uf_step'
#no dataset 'df_saeb_rend'

df_saeb_rend['fitted_step'] = modelo_saeb_dummies_uf_step.fittedvalues
df_saeb_rend['residuos_step'] = modelo_saeb_dummies_uf_step.resid
df_saeb_rend

# In[7.15]: Gráfico que relaciona resíduos e fitted values do
#'modelo_saeb_dummies_uf_step'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_saeb_rend,
            marker='o', fit_reg=False,
            scatter_kws={"color":'dodgerblue', 'alpha':0.2, 's':150})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise com Dummies', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise com Dummies', fontsize=20)
plt.xticks(fontsize=18)
plt.yticks(fontsize=18)
plt.show()

# In[7.16]: Histograma dos resíduos do 'modelo_saeb_dummies_uf_step' com curva
#normal teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_saeb_dummies_uf_step.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_saeb_dummies_uf_step.resid, bins=20, kde=True,
             stat="density", color='dodgerblue', alpha=0.4)
plt.xlim(-4, 4)
x = np.linspace(-4, 4, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise com Dummies', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(['Distribuição Real','Distribuição Normal Teórica','Histograma'],
           fontsize=18)
plt.show()

# In[7.17]: Plotando a variável 'saeb' em função de 'rendimento', com destaque
#para as unidades federativas e fits lineares - Gráfico pela função 'lmplot' do
#pacote 'seaborn', com estratificação de 'uf' pelo argumento 'hue'

uf_order = df_saeb_rend['uf'].value_counts().index.sort_values()

plt.figure(figsize=(15, 10))
sns.lmplot(x='rendimento', y='saeb', data=df_saeb_rend,
           hue='uf', ci=None, palette='viridis', legend=False,
           scatter_kws={'alpha': 0.5},
           hue_order=uf_order)
plt.title('Gráfico de Dispersão com Ajuste Linear por UF', fontsize=13)
plt.xlabel('rendimento', fontsize=12)
plt.ylabel('saeb', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.legend(fontsize=10, ncol=3, bbox_to_anchor=(1, 0.75))
plt.show()


# In[EXEMPLO 8]:
#############################################################################
#                 REGRESSÃO NÃO LINEAR MÚLTIPLA COM DUMMIES                 #
#                 EXEMPLO 8 - CARREGAMENTO DA BASE DE DADOS                 #
#############################################################################

df_planosaude = pd.read_csv('planosaude.csv', delimiter=',')
df_planosaude

# Características das variáveis do dataset
df_planosaude.info()

# Estatísticas univariadas
df_planosaude.describe()

# In[8.1]: Tabela de frequências absolutas da variável 'plano'

df_planosaude['plano'].value_counts().sort_index()

# In[8.2]: Distribuições das variáveis, scatters, valores das correlações e
#suas respectivas significâncias

def corrfunc(x, y, **kws):
    (r, p) = pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("r = {:.3f}".format(r),
                xy=(.30, .9), xycoords=ax.transAxes, fontsize=16)
    ax.annotate("p = {:.3f}".format(p),
                xy=(.30, .8), xycoords=ax.transAxes, fontsize=16)

# Configuração do gráfico
sns.set(style="whitegrid", palette="viridis")

plt.figure(figsize=(20,10))
graph = sns.pairplot(df_planosaude.loc[:,'despmed':'renda'], diag_kind="kde",
                     plot_kws={"color": "darkorange"},
                     height=2.5, aspect=1.7)
graph.map(corrfunc)
for ax in graph.axes.flat:
    ax.set_xlabel(ax.get_xlabel(), fontsize=17)
    ax.set_ylabel(ax.get_ylabel(), fontsize=17)
plt.show()

# In[8.3]: Dummizando a variável 'plano' (n-1 dummies)

df_planosaude_dummies = pd.get_dummies(df_planosaude, columns=['plano'],
                                       dtype=int,
                                       drop_first=True)

df_planosaude_dummies

# In[8.4]: Estimação do modelo de regressão múltipla com n-1 dummies

# Definição da fórmula utilizada no modelo
lista_colunas = list(df_planosaude_dummies.drop(columns=['id',
                                                         'despmed']).columns)
formula_dummies_modelo = ' + '.join(lista_colunas)
formula_dummies_modelo = "despmed ~ " + formula_dummies_modelo

# Estimação
modelo_planosaude = sm.OLS.from_formula(formula_dummies_modelo,
                                        df_planosaude_dummies).fit()

# Parâmetros do modelo
modelo_planosaude.summary()

# In[8.5]: Procedimento Stepwise

# Carregamento da função 'stepwise' do pacote 'statstests.process'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.process import stepwise

# Estimação do modelo por meio do procedimento Stepwise
modelo_step_planosaude = stepwise(modelo_planosaude, pvalue_limit=0.05)

# In[8.6]: Teste de verificação da aderência dos resíduos à normalidade

# Teste de Shapiro-Francia (n >= 30)
# Carregamento da função 'shapiro_francia' do pacote 'statstests.tests'
# Autores do pacote: Luiz Paulo Fávero e Helder Prado Santos
# https://stats-tests.github.io/statstests/

from statstests.tests import shapiro_francia

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step_planosaude.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[8.7]: Histograma dos resíduos do 'modelo_step_planosaude' com curva normal
#teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_step_planosaude.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_step_planosaude.resid, bins=15, kde=True, stat="density",
             color='red', alpha=0.4)
plt.xlim(-60, 70)
x = np.linspace(-60, 70, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise Linear', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(['Distribuição Real','Distribuição Normal Teórica','Histograma'],
           fontsize=18)
plt.show()

# In[8.8]: Função para o teste de Breusch-Pagan para a elaboração de diagnóstico
#de heterocedasticidade

# Criação da função 'breusch_pagan_test'

def breusch_pagan_test(modelo):

    df = pd.DataFrame({'yhat':modelo.fittedvalues,
                       'resid':modelo.resid})
   
    df['up'] = (np.square(df.resid))/np.sum(((np.square(df.resid))/df.shape[0]))
   
    modelo_aux = sm.OLS.from_formula('up ~ yhat', df).fit()
   
    anova_table = sm.stats.anova_lm(modelo_aux, typ=2)
   
    anova_table['sum_sq'] = anova_table['sum_sq']/2
    
    chisq = anova_table['sum_sq'].iloc[0]
   
    p_value = stats.chi2.pdf(chisq, 1)*2
    
    print(f"chisq: {chisq}")
    
    print(f"p-value: {p_value}")
    
    return float(chisq), float(p_value)

# In[8.9]: Teste de Breusch-Pagan propriamente dito

breusch_pagan_test(modelo_step_planosaude)
# Presença de heterocedasticidade -> omissão de variável(is) explicativa(s)
#relevante(s)

# H0 do teste: ausência de heterocedasticidade.
# H1 do teste: heterocedasticidade, ou seja, correlação entre resíduos e
#uma ou mais variáveis explicativas, o que indica omissão de variável relevante!

# Interpretação
teste_bp = breusch_pagan_test(modelo_step_planosaude) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[8.10]: Adicionando fitted values e resíduos do 'modelo_step_planosaude'
#no dataframe 'df_planosaude_dummies'

df_planosaude_dummies['fitted_step'] = modelo_step_planosaude.fittedvalues
df_planosaude_dummies['residuos_step'] = modelo_step_planosaude.resid
df_planosaude_dummies

# In[8.11]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_planosaude'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.5, 's':200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-50, 71, 20), fontsize=17)
x_min = df_planosaude_dummies['fitted_step'].min()-1
x_max = df_planosaude_dummies['fitted_step'].max()+1
plt.xlim(x_min, x_max)
plt.show()

# In[8.12]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_planosaude', com boundaries

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step', y='residuos_step', data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color":'red', 'alpha':0.5, 's':200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-50, 71, 20), fontsize=17)
x_min = df_planosaude_dummies['fitted_step'].min()-1
x_max = df_planosaude_dummies['fitted_step'].max()+1
plt.xlim(x_min, x_max)

sns.kdeplot(data=df_planosaude_dummies, x='fitted_step', y='residuos_step',
            levels=2, color='red', linewidths=3)

plt.show()

# In[8.13]: Transformação de Box-Cox

# Para o cálculo do lambda de Box-Cox
from scipy.stats import boxcox

# 'yast' é uma variável que traz os valores transformados (Y*)
# 'lmbda' é o lambda de Box-Cox
yast, lmbda = boxcox(df_planosaude_dummies['despmed'])

print("Lambda: ",lmbda)

# In[8.14]: Inserindo o lambda de Box-Cox no dataset para a estimação de um
#novo modelo

df_planosaude_dummies['bc_despmed'] = yast
df_planosaude_dummies

# Verificação do cálculo, apenas para fins didáticos
df_planosaude_dummies['bc_despmed2'] = ((df_planosaude_dummies['despmed'])**\
                                        (lmbda) - 1) / (lmbda)
df_planosaude_dummies

del df_planosaude_dummies['bc_despmed2']

# In[8.15]: Estimando um novo modelo com todas as variáveis e a variável
#dependente transformada
modelo_bc_planosaude = sm.OLS.from_formula('bc_despmed ~ idade + dcron +\
                                           renda + plano_esmeralda +\
                                               plano_ouro',
                                               df_planosaude_dummies).fit()

# Parâmetros do modelo
modelo_bc_planosaude.summary()

# In[8.16]: Procedimento Stepwise no 'modelo_bc_planosaude'

modelo_step_bc_planosaude = stepwise(modelo_bc_planosaude, pvalue_limit=0.05)

# In[8.17]: Teste de verificação da aderência à normalidade dos resíduos do novo
#'modelo_step_bc_planosaude'

# Teste de Shapiro-Francia: interpretação
teste_sf = shapiro_francia(modelo_step_bc_planosaude.resid) #criação do objeto 'teste_sf'
teste_sf = teste_sf.items() #retorna o grupo de pares de valores-chave no dicionário
method, statistics_W, statistics_z, p = teste_sf #definição dos elementos da lista (tupla)
print('Statistics W=%.5f, p-value=%.6f' % (statistics_W[1], p[1]))
alpha = 0.05 #nível de significância
if p[1] > alpha:
	print('Não se rejeita H0 - Distribuição aderente à normalidade')
else:
	print('Rejeita-se H0 - Distribuição não aderente à normalidade')

# In[8.18]: Histograma dos resíduos do 'modelo_step_bc_planosaude' com curva
#normal teórica para comparação das distribuições
# Kernel density estimation (KDE) - forma não-paramétrica para estimação da
#função densidade de probabilidade de determinada variável

from scipy.stats import norm

# Calcula os valores de ajuste da distribuição normal
(mu, sigma) = norm.fit(modelo_step_bc_planosaude.resid)

# Gráfico propriamente dito
plt.figure(figsize=(15,10))
sns.histplot(modelo_step_bc_planosaude.resid, bins=15, kde=True, stat="density",
             color='limegreen', alpha=0.4)
plt.xlim(-0.15, 0.15)
x = np.linspace(-0.15, 0.15, 100)
p = norm.pdf(x, mu, sigma)
plt.plot(x, p, 'k', linewidth=2)
plt.xlabel('Resíduos do Modelo Stepwise com Box-Cox', fontsize=20)
plt.ylabel('Frequência', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(fontsize=17)
plt.legend(['Distribuição Real','Distribuição Normal Teórica','Histograma'],
           fontsize=18)
plt.show()

# In[8.19]: Teste de Breusch-Pagan para diagnóstico de heterocedasticidade
#no 'modelo_step_bc_planosaude'

breusch_pagan_test(modelo_step_bc_planosaude)

# Interpretação
teste_bp = breusch_pagan_test(modelo_step_bc_planosaude) #criação do objeto 'teste_bp'
chisq, p = teste_bp #definição dos elementos contidos no objeto 'teste_bp'
alpha = 0.05 #nível de significância
if p > alpha:
    print('Não se rejeita H0 - Ausência de Heterocedasticidade')
else:
	print('Rejeita-se H0 - Existência de Heterocedasticidade')

# In[8.20]: Adicionando fitted values e resíduos do 'modelo_step_bc_planosaude'
#no dataframe 'df_planosaude_dummies'

df_planosaude_dummies['fitted_step_bc'] = modelo_step_bc_planosaude.fittedvalues
df_planosaude_dummies['residuos_step_bc'] = modelo_step_bc_planosaude.resid
df_planosaude_dummies

# In[8.21]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_bc_planosaude'

plt.figure(figsize=(15,10))
sns.regplot(x='fitted_step_bc', y='residuos_step_bc', data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color":'limegreen', 'alpha':0.5, 's':200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise com Box-Cox', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise com Box-Cox', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-.15, .16, .05), fontsize=17)
x_min = df_planosaude_dummies['fitted_step_bc'].min()-0.01
x_max = df_planosaude_dummies['fitted_step_bc'].max()+0.01
plt.xlim(x_min, x_max)
plt.show()

# In[8.22]: Gráfico que relaciona resíduos e fitted values do
#'modelo_step_bc_planosaude', com boundaries

plt.figure(figsize=(15, 10))
sns.regplot(x='fitted_step_bc', y='residuos_step_bc',
            data=df_planosaude_dummies,
            marker='o', fit_reg=False,
            scatter_kws={"color": 'limegreen', 'alpha': 0.5, 's': 200})
plt.title('Gráfico de Dispersão entre Resíduos e Fitted Values', fontsize=22)
plt.xlabel('Fitted Values do Modelo Stepwise com Box-Cox', fontsize=20)
plt.ylabel('Resíduos do Modelo Stepwise com Box-Cox', fontsize=20)
plt.xticks(fontsize=17)
plt.yticks(np.arange(-.15, .16, .05), fontsize=17)
x_min = df_planosaude_dummies['fitted_step_bc'].min()-0.01
x_max = df_planosaude_dummies['fitted_step_bc'].max()+0.01
plt.xlim(x_min, x_max)

sns.kdeplot(data=df_planosaude_dummies, x='fitted_step_bc', y='residuos_step_bc',
            levels=2, color='green', linewidths=3)

################################## FIM ######################################