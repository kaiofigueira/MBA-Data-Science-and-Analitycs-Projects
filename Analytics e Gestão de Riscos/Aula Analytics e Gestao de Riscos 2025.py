"""
# =============================================================================
# AULA ANALYTICS E GESTÃO DE RISCOS - Prof. Fabiano
# =============================================================================
# MBA Data Science e Analytics - USP - ESALQ - 2025
# =============================================================================
"""
#%%

"""
# Instalar pacotes  

# copiar e colar no Console (sem o #)
# pip install --upgrade pip setuptools wheel
# pip install yfinance pyportfolioopt cvxpy osqp scs clarabel

"""
#%%
# ============================ Importações ====================================

import warnings
warnings.filterwarnings("ignore")

import datetime as dt
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import cvxpy as cp
from pypfopt import EfficientFrontier, risk_models, expected_returns

print("Solvers disponíveis:", cp.installed_solvers())

#%%
# ============================ Parâmetros =====================================
# Datas (últimos 120 dias corridos)
data_final = dt.date.today()
data_inicial = data_final - dt.timedelta(days=120)

# Tickers de exemplo (B3 via Yahoo Finance)
tickers = ["PETR4.SA", "VALE3.SA", "EMBR3.SA"]

# Taxa livre de risco anual (ex.: CDI ~14,5%)
rf_annual = 0.145

# Semente para reprodutibilidade
np.random.seed(42)


#%%
"""
Aqui vamos ler as cotacoes a partir do arquivo COTACOES.XLSX

"""
# ============================ Leitura de dados (Excel) =======================
# Requer: pip install openpyxl

# Lê a primeira aba; assume 1ª coluna = Data e as demais colunas = tickers
df = pd.read_excel("COTACOES.xlsx", sheet_name=0)

# Detecta a coluna de data (usa a 1ª coluna da planilha)
col_data = df.columns[0]
df[col_data] = pd.to_datetime(df[col_data], dayfirst=True, errors="coerce")
df = df.dropna(subset=[col_data]).set_index(col_data).sort_index()

# Mantém apenas os tickers que existem na planilha
cols_validas = [t for t in tickers if t in df.columns]
precos = df.loc[(df.index.date >= data_inicial) & (df.index.date <= data_final), cols_validas]
precos = precos.dropna(axis=1, how="all")
precos = df.copy()


#%%
"""
ALTERNATIVA para não precisar ter o arquivo COTACOES.XLSX

"""
# ============================ Download de dados ==============================
precos = yf.download(
    tickers=tickers,
    start=data_inicial,
    end=data_final + dt.timedelta(days=1),
    auto_adjust=True,
    progress=False
)["Close"].dropna(axis=1, how="all")

if precos.empty:
    raise ValueError("Sem dados baixados. Verifique os tickers/período/conexão.")
    

#%%
# ============================ Retornos diários ===============================
retornos = precos.pct_change().dropna()

# Histograma exemplo PETR4
if "PETR4.SA" in retornos.columns:
    petr4 = retornos["PETR4.SA"]
    plt.figure(figsize=(8, 4.5))
    plt.hist(petr4, bins=30, edgecolor="black")
    plt.title("Histograma dos Retornos Diários - PETR4.SA")
    plt.xlabel("Retorno diário")
    plt.ylabel("Frequência")
    plt.grid(True, alpha=0.3)
    plt.show()
    print("Desvio-padrão (risco) diário PETR4:", petr4.std())

print("\nRetornos médios diários:\n", retornos.mean())
print("\nMatriz de covariância amostral:\n", retornos.cov())
print("\nMatriz de correlação (Pearson):\n", retornos.corr())

#%%
# ============================ Fronteira de Markowitz =========================
# Anualização (252 pregões/ano)
freq = 252

# Estatísticas diárias
mu_daily = retornos.mean()   # média diária por ativo
S_daily  = retornos.cov()    # covariância diária entre ativos

# Anualização manual
mu = mu_daily * freq             # retorno esperado anual
S  = S_daily * freq              # covariância anual
vol_annual_assets = np.sqrt(np.diag(S))  # risco (volatilidade) anual por ativo

print("\nRetornos esperados anualizados (manual):\n", mu.round(6))
print("\nMatriz de covariância anualizada (manual):\n", S.round(8))
print("\nRisco (volatilidade) anualizado por ativo:\n",
      dict(zip(mu.index, vol_annual_assets.round(6))))


#%%
# ============================ Carteira de Mínimo Risco ========================
ef_min = EfficientFrontier(mu, S, weight_bounds=(0, 1))
_ = ef_min.min_volatility()
w_min_clean = ef_min.clean_weights()
ret_min, vol_min, sharpe_min = ef_min.portfolio_performance(risk_free_rate=rf_annual)

# Organizar pesos em formato de coluna
pesos_min = pd.DataFrame.from_dict(w_min_clean, orient="index", columns=["Peso"])
pesos_min = pesos_min[pesos_min["Peso"] > 0]  # opcional: mostra só ativos com peso > 0

# Organizar desempenho em coluna
desempenho_min = pd.DataFrame({
    "Métrica": ["Retorno Anual", "Volatilidade Anual", "Índice de Sharpe"],
    "Valor": [ret_min, vol_min, sharpe_min]
})

print("\nCarteira de Mínimo Risco (pesos):")
print(pesos_min.round(4))

print("\nDesempenho da Carteira de Mínimo Risco:")
print(desempenho_min.round(4))

#%%
# ============================ Carteira Tangente (máx. Sharpe) =================
ef_tan = EfficientFrontier(mu, S, weight_bounds=(0, 1))
_ = ef_tan.max_sharpe(risk_free_rate=rf_annual)
w_tan_clean = ef_tan.clean_weights()
ret_tan, vol_tan, sharpe_tan = ef_tan.portfolio_performance(risk_free_rate=rf_annual)

# Organizar pesos em formato de coluna
pesos_tan = pd.DataFrame.from_dict(w_tan_clean, orient="index", columns=["Peso"])
pesos_tan = pesos_tan[pesos_tan["Peso"] > 0]  # opcional: mostra só ativos com peso > 0

# Organizar desempenho em coluna
desempenho_tan = pd.DataFrame({
    "Métrica": ["Retorno Anual", "Volatilidade Anual", "Índice de Sharpe"],
    "Valor": [ret_tan, vol_tan, sharpe_tan]
})

print("\nCarteira Tangente (pesos):")
print(pesos_tan.round(4))

print("\nDesempenho da Carteira Tangente:")
print(desempenho_tan.round(4))


#%%

# ============================ Fronteira Eficiente (curva) ====================
target_rets = np.linspace(mu.min(), mu.max(), 60)
frontier_rets, frontier_vols = [], []

for tr in target_rets:
    ef = EfficientFrontier(mu, S, weight_bounds=(0, 1))
    try:
        ef.efficient_return(target_return=tr)
        r, v, _ = ef.portfolio_performance(risk_free_rate=rf_annual)
        frontier_rets.append(r)
        frontier_vols.append(v)
    except Exception:
        pass

frontier_rets = np.array(frontier_rets)
frontier_vols = np.array(frontier_vols)

#%%

# ============================ Monte Carlo de Portfólios ======================
def random_weights(n_assets, n_port=2000):
    # Amostra pesos ~ Dirichlet para somar 1
    return np.random.dirichlet(np.ones(n_assets), size=n_port)

W = random_weights(len(mu), n_port=3000)
rets_mc = W @ mu.values
vols_mc = np.sqrt(np.einsum('ij,jk,ik->i', W, S.values, W))  # raiz(diag(W S W^T))


# ============================ Pontos Especiais ===============================
# Pesos iguais
weights_equal = np.repeat(1/len(mu), len(mu))
ret_eq = float(weights_equal @ mu.values)
vol_eq = float(np.sqrt(weights_equal @ S.values @ weights_equal))

# Ativos individuais
ret_assets = mu.values
vol_assets = np.sqrt(np.diag(S.values))
tickers_plot = list(precos.columns)


#%%
# ============================ Gráfico Final ==================================
plt.figure(figsize=(10, 6.5))

# Nuvem Monte Carlo
plt.scatter(vols_mc, rets_mc, alpha=0.25, s=8, label="Portfólios (Monte Carlo)")

# Fronteira eficiente
plt.plot(frontier_vols, frontier_rets, lw=2.2, label="Fronteira Eficiente")

# Carteira de mínimo risco
plt.scatter(vol_min, ret_min, marker="*", s=220, label="Mínimo Risco")

# Carteira tangente (máx. Sharpe)
plt.scatter(vol_tan, ret_tan, marker="X", s=130, label="Tangente (Máx. Sharpe)")

# Pesos iguais
plt.scatter(vol_eq, ret_eq, marker="o", s=90, label="Pesos Iguais")

# Ativos
plt.scatter(vol_assets, ret_assets, marker="D", s=70, label="Ativos")
for i, tkr in enumerate(tickers_plot):
    plt.annotate(tkr, (vol_assets[i], ret_assets[i]),
                 xytext=(6, 6), textcoords="offset points", fontsize=9)

plt.title("Fronteira Eficiente de Markowitz com Monte Carlo - Exemplo B3")
plt.xlabel("Volatilidade Anual (desvio-padrão)")
plt.ylabel("Retorno Anual Esperado")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()

#%%
# ============================ Resumo =========================================
print("\n--- Resumo ---")

# --- Carteira de Mínimo Risco ---
pesos_min = pd.DataFrame.from_dict(w_min_clean, orient="index", columns=["Peso"])
pesos_min = pesos_min[pesos_min["Peso"] > 0]
ret_min_carteira = float(pesos_min["Peso"].values @ mu.loc[pesos_min.index].values)
vol_min_carteira = float(np.sqrt(pesos_min["Peso"].values @ S.loc[pesos_min.index, pesos_min.index].values @ pesos_min["Peso"].values))

print("\nCarteira de Mínimo Risco (pesos):")
print(pesos_min.round(4))
print(f"Retorno Anual Esperado: {ret_min_carteira:.4f}")
print(f"Risco (Volatilidade Anual): {vol_min_carteira:.4f}")


# --- Carteira Tangente (Máx. Sharpe) ---
pesos_tan = pd.DataFrame.from_dict(w_tan_clean, orient="index", columns=["Peso"])
pesos_tan = pesos_tan[pesos_tan["Peso"] > 0]
ret_tan_carteira = float(pesos_tan["Peso"].values @ mu.loc[pesos_tan.index].values)
vol_tan_carteira = float(np.sqrt(pesos_tan["Peso"].values @ S.loc[pesos_tan.index, pesos_tan.index].values @ pesos_tan["Peso"].values))

print("\nCarteira Tangente (pesos):")
print(pesos_tan.round(4))
print(f"Retorno Anual Esperado: {ret_tan_carteira:.4f}")
print(f"Risco (Volatilidade Anual): {vol_tan_carteira:.4f}")


# --- Carteira de Pesos Iguais ---
pesos_eq = pd.DataFrame({
    "Ativo": tickers_plot,
    "Peso": np.round(weights_equal, 4)
})
pesos_eq = pesos_eq.set_index("Ativo")
ret_eq_carteira = float(weights_equal @ mu.values)
vol_eq_carteira = float(np.sqrt(weights_equal @ S.values @ weights_equal))

print("\nCarteira de Pesos Iguais:")
print(pesos_eq)
print(f"Retorno Anual Esperado: {ret_eq_carteira:.4f}")
print(f"Risco (Volatilidade Anual): {vol_eq_carteira:.4f}")


# --- Período analisado ---
print("\nPeríodo analisado:")
print(f"{precos.index.min().date()}  ->  {precos.index.max().date()}")



#%%
"""
# =============================================================================
# SIMULAÇÃO DE CENÁRIOS E VPL (NPV)  
#  
# =============================================================================
"""
# Se precisar instalar (Colab geralmente já tem):
# !pip install numpy pandas matplotlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================ PREMISSAS ======================================
# Triangular para cada variável (min = pessimista, mode = mais provável, max = otimista)
premissas = {
    "preco_venda":   {"min": 49.0,    "mode": 51.0,   "max": 55.0},     # R$/unid
    "quantidade":    {"min": 990.0,   "mode": 1100.0, "max": 1200.0},   # unid/ano
    "preco_custo":   {"min": 28.0,    "mode": 29.5,   "max": 32.0},     # R$/unid
    "desp_op":       {"min": 5000.0,  "mode": 6000.0, "max": 6500.0},   # R$/ano
    "investimento":  {"min": 19000.0, "mode": 20000.0,"max": 20800.0},  # R$ (Ano 0)
    "taxa_desconto": {"min": 0.17,    "mode": 0.18,   "max": 0.19},     # a.a.
}

anos_vida   = 4       # 4 anos de projeto (depreciação linear Invest/4)
aliquota_ir = 0.40    # 40% de IR sobre EBIT>0
n_sims      = 10000   # nº de simulações
np.random.seed(42)    # reprodutibilidade

# ======================== FUNÇÕES AUXILIARES =================================
def amostrar_triangular(p, n):
    """Amostra n valores na triangular (min, mode, max)."""
    a, c, b = p["min"], p["mode"], p["max"]
    if not (a <= c <= b):
        raise ValueError(f"Parâmetros inválidos: {p}")
    return np.random.triangular(a, c, b, size=n)

def fluxo_base_por_moda(prem):
    """Esboço determinístico do fluxo de caixa usando os valores de moda (mais prováveis)."""
    preco = prem["preco_venda"]["mode"]
    qtd   = prem["quantidade"]["mode"]
    custo = prem["preco_custo"]["mode"]
    desp  = prem["desp_op"]["mode"]
    inv   = prem["investimento"]["mode"]
    r     = prem["taxa_desconto"]["mode"]

    dep_anual = inv / anos_vida

    receita = preco * qtd
    custos  = custo * qtd
    ebitda  = receita - custos - desp
    ebit    = ebitda - dep_anual
    ir      = max(0.0, aliquota_ir * ebit)
    nopat   = ebit - ir
    fco     = nopat + dep_anual

    # Ano 0: investimento (saída)
    anos = [0] + list(range(1, anos_vida + 1))
    df = pd.DataFrame({
        "Ano": anos,
        "Receita": [0.0] + [receita]*anos_vida,
        "Custos":  [0.0] + [custos]*anos_vida,
        "Desp_Op": [0.0] + [desp]*anos_vida,
        "Dep":     [0.0] + [dep_anual]*anos_vida,
        "EBITDA":  [0.0] + [ebitda]*anos_vida,
        "EBIT":    [0.0] + [ebit]*anos_vida,
        "IR":      [0.0] + [ir]*anos_vida,
        "NOPAT":   [0.0] + [nopat]*anos_vida,
        "FCO":     [-inv] + [fco]*anos_vida,
    })

    # Fator de desconto e VP do FCO
    df["Fator_Desconto"] = [1.0] + [1.0 / ((1.0 + r) ** t) for t in range(1, anos_vida + 1)]
    df["VP_FCO"] = df["FCO"] * df["Fator_Desconto"]

    vpl_base = df["VP_FCO"].sum()
    return df, vpl_base, r

def simular_vpl(prem):
    """Calcula vetor de VPLs (n_sims) dado o dicionário de premissas (Monte Carlo)."""
    preco = amostrar_triangular(prem["preco_venda"], n_sims)
    qtd   = amostrar_triangular(prem["quantidade"], n_sims)
    custo = amostrar_triangular(prem["preco_custo"], n_sims)
    desp  = amostrar_triangular(prem["desp_op"], n_sims)
    inv   = amostrar_triangular(prem["investimento"], n_sims)
    r     = amostrar_triangular(prem["taxa_desconto"], n_sims)

    dep = inv / anos_vida

    receita = preco * qtd
    custos  = custo * qtd
    ebitda  = receita - custos - desp
    ebit    = ebitda - dep
    ir      = np.where(ebit > 0, aliquota_ir * ebit, 0.0)
    nopat   = ebit - ir
    fco     = nopat + dep

    # VPL = -INV + Σ FCO_t/(1+r)^t, t=1..anos_vida
    descontos = np.vstack([(1.0 / (1.0 + r) ** t) for t in range(1, anos_vida + 1)])
    vpl = -inv + (fco * descontos).sum(axis=0)
    return pd.Series(vpl, name="VPL")

def brl(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

#%%


# =============================================================================
# ETAPA 1 — ESBOÇO DO FLUXO DE CAIXA (DETERMINÍSTICO, USANDO MODA)
# =============================================================================
df_base, vpl_base, r_base = fluxo_base_por_moda(premissas)

print("=============== ESBOÇO DO FLUXO DE CAIXA (base: moda) ===============")
print(df_base.round(2))
print(f"\nTaxa de desconto (base): {r_base:.2%}")
print(f"VPL (base - moda): {brl(vpl_base)}")


#%%
# =============================================================================
# ETAPA 2 — SIMULAÇÃO MONTE CARLO + GRÁFICO E RESUMO
# =============================================================================
vpls = simular_vpl(premissas)

# Estatísticas
mean_vpl = vpls.mean()
std_vpl  = vpls.std(ddof=1)
q05, q50, q95 = vpls.quantile([0.05, 0.50, 0.95])
p_pos = (vpls > 0).mean()

print("\n=============== RESUMO DAS SIMULAÇÕES (VPL) ===============")
print("Média.............:", brl(mean_vpl))
print("Mediana (50%).....:", brl(q50))
print("Desvio-padrão.....:", brl(std_vpl))
print("P5................:", brl(q05))
print("P95...............:", brl(q95))
print(f"P(VPL>0)..........: {100*p_pos:.2f}%")
print("Amostras..........:", len(vpls))

#%%
# ------------------------------ GRÁFICO --------------------------------------
fig, ax = plt.subplots(figsize=(9.5, 5.6))

# Histograma
ax.hist(vpls, bins=40, edgecolor="black", alpha=0.85)

# Linhas verticais
ax.axvline(q05, linestyle="--", linewidth=1.25, label="P5 / P95")
ax.axvline(q95, linestyle="--", linewidth=1.25)
ax.axvline(0,  linestyle=":",  linewidth=1.0,  label="VPL = 0")

ax.set_title("VPL (NPV) - Distribuição Monte Carlo")
ax.set_xlabel("VPL (R$)")
ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.25)

# Barra superior (5% | 90% | 5%)
y_top = ax.get_ylim()[1]
ax.hlines(y=y_top*0.97, xmin=min(vpls), xmax=max(vpls), linestyles="-", linewidth=0.8)
ax.hlines(y=y_top*0.97, xmin=min(vpls), xmax=q05, linestyles="-", linewidth=4)
ax.hlines(y=y_top*0.97, xmin=q05, xmax=q95, linestyles="-", linewidth=8)
ax.hlines(y=y_top*0.97, xmin=q95, xmax=max(vpls), linestyles="-", linewidth=4)
ax.text(q05, (y_top*0.99), "5,0%", ha="left",  va="bottom", fontsize=9)
ax.text((q05+q95)/2, (y_top*0.99), "90,0%", ha="center", va="bottom", fontsize=9)
ax.text(q95, (y_top*0.99), "5,0%", ha="right", va="bottom", fontsize=9)

# Anotações
ax.annotate(f"P5 = {brl(q05)}",  xy=(q05, y_top*0.90), ha="left",  fontsize=9)
ax.annotate(f"P95 = {brl(q95)}", xy=(q95, y_top*0.90), ha="right", fontsize=9)

# Caixa de estatísticas
text_stats = (
    f"Média: {brl(mean_vpl)}\n"
    f"Mediana: {brl(q50)}\n"
    f"Desv.Pad.: {brl(std_vpl)}\n"
    f"P5: {brl(q05)} | P95: {brl(q95)}\n"
    f"P(VPL>0): {100*p_pos:.2f}%"
)
ax.text(0.98, 0.97, text_stats, transform=ax.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, ec="gray"))

ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

#%%

# =============================================================================
# TORNADO (centrado no baseline, ΔVPL>0 à direita, ΔVPL<0 à esquerda)
# Ordenado do MENOR para o MAIOR impacto
# =============================================================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# ---------------------------- Premissas (triangular) -------------------------
premissas = {
    "preco_venda":   {"min": 49.0,    "mode": 51.0,   "max": 55.0},     # R$/unid
    "quantidade":    {"min": 990.0,   "mode": 1100.0, "max": 1200.0},   # unid/ano
    "preco_custo":   {"min": 28.0,    "mode": 29.5,   "max": 32.0},     # R$/unid
    "desp_op":       {"min": 5000.0,  "mode": 6000.0, "max": 6500.0},   # R$/ano
    "investimento":  {"min": 19000.0, "mode": 20000.0,"max": 20800.0},  # R$ (Ano 0)
    "taxa_desconto": {"min": 0.17,    "mode": 0.18,   "max": 0.19},     # a.a.
}

anos_vida   = 4
aliquota_ir = 0.40

def brl(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

def vpl_deterministico(preco, qtd, custo, desp, inv, r):
    dep = inv / anos_vida
    receita = preco * qtd
    custos  = custo * qtd
    ebitda  = receita - custos - desp
    ebit    = ebitda - dep
    ir      = aliquota_ir * ebit if ebit > 0 else 0.0
    nopat   = ebit - ir
    fco     = nopat + dep
    pv = sum(fco / (1 + r) ** t for t in range(1, anos_vida + 1))
    return -inv + pv

# Base (valores "mode")
base = {k: v["mode"] for k, v in premissas.items()}
baseline = vpl_deterministico(
    preco=base["preco_venda"], qtd=base["quantidade"], custo=base["preco_custo"],
    desp=base["desp_op"], inv=base["investimento"], r=base["taxa_desconto"]
)

labels_map = {
    "preco_venda":   "Receitas / Ano 1",
    "quantidade":    "Quantidade / Ano 1",
    "preco_custo":   "Preço de Custo / Ano 1",
    "desp_op":       "Despesas / Ano 1",
    "investimento":  "Investimento / Ano 0",
    "taxa_desconto": "Custo de Oportunidade (a.a.)",
}

# Calcula VPL com input no "min" (Low) e no "max" (High), mantendo demais no mode
rows = []
for var in ["preco_venda","quantidade","preco_custo","desp_op","investimento","taxa_desconto"]:
    low_vals  = base.copy();  low_vals[var]  = premissas[var]["min"]
    high_vals = base.copy();  high_vals[var] = premissas[var]["max"]

    vpl_low  = vpl_deterministico(low_vals["preco_venda"], low_vals["quantidade"],
                                  low_vals["preco_custo"], low_vals["desp_op"],
                                  low_vals["investimento"], low_vals["taxa_desconto"])
    vpl_high = vpl_deterministico(high_vals["preco_venda"], high_vals["quantidade"],
                                  high_vals["preco_custo"], high_vals["desp_op"],
                                  high_vals["investimento"], high_vals["taxa_desconto"])

    d_low  = vpl_low  - baseline   # ΔVPL (low - baseline)
    d_high = vpl_high - baseline   # ΔVPL (high - baseline)
    impacto = abs(vpl_high - vpl_low)  # largura total do efeito

    rows.append([labels_map[var], d_low, d_high, impacto, vpl_low, vpl_high])

df = pd.DataFrame(rows, columns=["Variável","ΔLow","ΔHigh","Impacto","VPL Low","VPL High"])

# Ordenação do MENOR → MAIOR impacto (topo para base)
df = df.sort_values("Impacto", ascending=True).reset_index(drop=True)

# ---------------------------- Plot lado a lado (no mesmo nível) -------------
y = np.arange(len(df))
h = 0.38

fig, ax = plt.subplots(figsize=(11, 6))
# Linha vertical do baseline (apenas referência visual)
ax.axvline(baseline, linestyle=":", linewidth=1.2, color="k", label=f"Baseline = {brl(baseline)}")

# Barras: cada variável tem duas barras no MESMO y:
#   - azul (High) indo para a direita/esquerda conforme o sinal de ΔHigh
#   - vermelha (Low) idem para ΔLow
# Ambas partem do baseline, ou seja, "lado a lado" na mesma linha.
# (largura negativa em barh vai para a esquerda)
ax.barh(y, df["ΔHigh"], left=baseline, height=h, label="Input High", color="#1f4acc")
ax.barh(y, df["ΔLow"],  left=baseline, height=h, label="Input Low",  color="#b22222", alpha=0.85)

# Rótulos eixo Y
ax.set_yticks(y)
ax.set_yticklabels(df["Variável"])

ax.set_title("Gráfico Tornado — ΔVPL em relação ao baseline\n(MENOR → MAIOR impacto, barras lado a lado no mesmo nível)")
ax.grid(True, axis="x", alpha=0.25)

# Formatar eixo X em R$
ax.xaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: brl(x)))

# Anotar VPL absoluto nas pontas das barras (como no exemplo @RISK)
for i, (dl, dh, vl, vh) in enumerate(zip(df["ΔLow"], df["ΔHigh"], df["VPL Low"], df["VPL High"])):
    # ponta Low
    xlow  = baseline + dl
    ha_l  = "right" if dl < 0 else "left"
    ax.text(xlow, y[i], brl(vl), va="center", ha=ha_l, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, ec="none"))
    # ponta High
    xhigh = baseline + dh
    ha_h  = "left" if dh > 0 else "right"
    ax.text(xhigh, y[i], brl(vh), va="center", ha=ha_h, fontsize=9,
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.6, ec="none"))

# Legenda sem duplicar entradas
handles, labels = ax.get_legend_handles_labels()
bylabel = dict(zip(labels, handles))
ax.legend(bylabel.values(), bylabel.keys(), loc="lower right")

plt.tight_layout()
plt.show()

#%%
# =============================================================================
# SIMULAÇÃO DE CENÁRIOS E VPL (NPV) — Script Único
# Material didático - NÃO é recomendação de investimentos
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================ PREMISSAS (mais “pessimistas”) ==================
# Ajustadas para gerar probabilidade positiva de VPL negativo (P(VPL<0) > 0)
premissas = {
    "preco_venda":   {"min": 46.0,  "mode": 50.0,  "max": 56.0},     # R$/unid
    "quantidade":    {"min": 900.0, "mode": 1050.0,"max": 1200.0},   # unid/ano
    "preco_custo":   {"min": 30.0,  "mode": 33.0,  "max": 36.0},     # R$/unid
    "desp_op":       {"min": 6000.0,"mode": 7000.0,"max": 9000.0},   # R$/ano
    "investimento":  {"min": 22000.0,"mode": 25000.0,"max": 30000.0},# R$ (Ano 0)
    "taxa_desconto": {"min": 0.18,   "mode": 0.22,  "max": 0.28},    # a.a.
}

anos_vida   = 4       # 4 anos de projeto (depreciação linear Invest/4)
aliquota_ir = 0.40    # 40% de IR sobre EBIT>0
n_sims      = 10000   # nº de simulações
np.random.seed(42)    # reprodutibilidade

# ======================== FUNÇÕES AUXILIARES =================================
def amostrar_triangular(p, n):
    """Amostra n valores na triangular (min, mode, max)."""
    a, c, b = p["min"], p["mode"], p["max"]
    if not (a <= c <= b):
        raise ValueError(f"Parâmetros inválidos: {p}")
    return np.random.triangular(a, c, b, size=n)

def fluxo_base_por_moda(prem):
    """Esboço determinístico do fluxo de caixa usando os valores de moda (mais prováveis)."""
    preco = prem["preco_venda"]["mode"]
    qtd   = prem["quantidade"]["mode"]
    custo = prem["preco_custo"]["mode"]
    desp  = prem["desp_op"]["mode"]
    inv   = prem["investimento"]["mode"]
    r     = prem["taxa_desconto"]["mode"]

    dep_anual = inv / anos_vida

    receita = preco * qtd
    custos  = custo * qtd
    ebitda  = receita - custos - desp
    ebit    = ebitda - dep_anual
    ir      = max(0.0, aliquota_ir * ebit)
    nopat   = ebit - ir
    fco     = nopat + dep_anual

    # Ano 0: investimento (saída)
    anos = [0] + list(range(1, anos_vida + 1))
    df = pd.DataFrame({
        "Ano": anos,
        "Receita": [0.0] + [receita]*anos_vida,
        "Custos":  [0.0] + [custos]*anos_vida,
        "Desp_Op": [0.0] + [desp]*anos_vida,
        "Dep":     [0.0] + [dep_anual]*anos_vida,
        "EBITDA":  [0.0] + [ebitda]*anos_vida,
        "EBIT":    [0.0] + [ebit]*anos_vida,
        "IR":      [0.0] + [ir]*anos_vida,
        "NOPAT":   [0.0] + [nopat]*anos_vida,
        "FCO":     [-inv] + [fco]*anos_vida,
    })

    # Fator de desconto e VP do FCO
    df["Fator_Desconto"] = [1.0] + [1.0 / ((1.0 + r) ** t) for t in range(1, anos_vida + 1)]
    df["VP_FCO"] = df["FCO"] * df["Fator_Desconto"]

    vpl_base = df["VP_FCO"].sum()
    return df, vpl_base, r

def simular_vpl(prem):
    """Retorna Series com VPLs (n_sims) dado o dicionário de premissas (Monte Carlo)."""
    preco = amostrar_triangular(prem["preco_venda"], n_sims)
    qtd   = amostrar_triangular(prem["quantidade"], n_sims)
    custo = amostrar_triangular(prem["preco_custo"], n_sims)
    desp  = amostrar_triangular(prem["desp_op"], n_sims)
    inv   = amostrar_triangular(prem["investimento"], n_sims)
    r     = amostrar_triangular(prem["taxa_desconto"], n_sims)

    dep = inv / anos_vida

    receita = preco * qtd
    custos  = custo * qtd
    ebitda  = receita - custos - desp
    ebit    = ebitda - dep
    ir      = np.where(ebit > 0, aliquota_ir * ebit, 0.0)
    nopat   = ebit - ir
    fco     = nopat + dep

    # VPL = -INV + Σ FCO_t/(1+r)^t, t=1..anos_vida
    descontos = np.vstack([(1.0 / (1.0 + r) ** t) for t in range(1, anos_vida + 1)])
    vpl = -inv + (fco * descontos).sum(axis=0)
    return pd.Series(vpl, name="VPL")

def brl(x):
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")

# =============================================================================
# ETAPA 1 — ESBOÇO DO FLUXO DE CAIXA (DETERMINÍSTICO, USANDO MODA)
# =============================================================================
df_base, vpl_base, r_base = fluxo_base_por_moda(premissas)

print("=============== ESBOÇO DO FLUXO DE CAIXA (base: moda) ===============")
print(df_base.round(2))
print(f"\nTaxa de desconto (base): {r_base:.2%}")
print(f"VPL (base - moda): {brl(vpl_base)}")

# =============================================================================
# ETAPA 2 — SIMULAÇÃO MONTE CARLO + HISTOGRAMA (ênfase VPL<0) + CDF
# =============================================================================
vpls = simular_vpl(premissas)

# Estatísticas principais
mean_vpl = vpls.mean()
std_vpl  = vpls.std(ddof=1)
q05, q50, q95 = vpls.quantile([0.05, 0.50, 0.95])
p_neg = (vpls < 0).mean()

print("\n=============== RESUMO DAS SIMULAÇÕES (VPL) ===============")
print("Média.............:", brl(mean_vpl))
print("Mediana (50%).....:", brl(q50))
print("Desvio-padrão.....:", brl(std_vpl))
print("P5................:", brl(q05))
print("P95...............:", brl(q95))
print(f"P(VPL<0)..........: {100*p_neg:.2f}%")
print("Amostras..........:", len(vpls))

# ----------------------- HISTOGRAMA (área negativa em vermelho) --------------
fig, ax = plt.subplots(figsize=(9.8, 5.8))
counts, bins, patches = ax.hist(vpls, bins=40, edgecolor="black", alpha=0.85)

# Colorir as barras com centro à esquerda de 0 em vermelho
for patch, left in zip(patches, bins[:-1]):
    if left < 0:
        patch.set_facecolor("red")

ax.axvline(0, linestyle=":", linewidth=1.3, color="black", label="VPL = 0")
ax.axvline(q05, linestyle="--", linewidth=1.1, color="blue", label="P5 / P95")
ax.axvline(q95, linestyle="--", linewidth=1.1, color="blue")

ax.set_title("VPL — Distribuição (área negativa em vermelho)")
ax.set_xlabel("VPL (R$)")
ax.set_ylabel("Frequência")
ax.grid(True, alpha=0.25)

txt = (
    f"P(VPL < 0) = {100*p_neg:.2f}%\n"
    f"P(VPL ≥ 0) = {100*(1-p_neg):.2f}%\n"
    f"P5: {brl(q05)} | P50: {brl(q50)} | P95: {brl(q95)}"
)
ax.text(0.98, 0.97, txt, transform=ax.transAxes, fontsize=9,
        va="top", ha="right",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.9, ec="gray"))

ax.legend(loc="upper left", fontsize=9)
plt.tight_layout()
plt.show()

# -------------------------- CDF COM MARCAÇÃO P(VPL<0) ------------------------
v_sorted = np.sort(vpls.values)
prob = np.linspace(0, 1, len(v_sorted), endpoint=False)

fig, ax = plt.subplots(figsize=(9.0, 5.2))
ax.plot(v_sorted, prob, linewidth=1.8)

# Marcações úteis
ax.axvline(0, linestyle=":", linewidth=1.2, label="VPL = 0")
ax.axhline(p_neg, linestyle="--", linewidth=1.2, label=f"P(VPL < 0) ≈ {100*p_neg:.2f}%")

# sombrear região VPL < 0
idx_neg = np.searchsorted(v_sorted, 0, side="left")
if idx_neg > 1:
    ax.fill_between(v_sorted[:idx_neg], prob[:idx_neg], 0, alpha=0.15, step="pre")

ax.set_title("CDF do VPL — Probabilidade acumulada")
ax.set_xlabel("VPL (R$)")
ax.set_ylabel("Probabilidade acumulada")
ax.grid(True, alpha=0.3)
ax.legend(loc="lower right", fontsize=9)

# Anotação
ax.annotate(f"P(VPL < 0) ≈ {100*p_neg:.2f}%",
            xy=(0, p_neg), xytext=(10, -10),
            textcoords="offset points", fontsize=9)

plt.tight_layout()
plt.show()

#%%