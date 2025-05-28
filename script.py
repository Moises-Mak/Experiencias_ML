# %% [markdown]
# # Análise séries temporais de dados Climáticos/Uige

# %% [markdown]
# #### A análise de séries temporais de dados climáticos para identificação de ciclos e padrões de longo prazo no Uíge

# %% [markdown]
# ### Coleta dos Dados Climáticos
# EC_Terra3P_HR	Europa	Consórcio EC-Earth, Rossby Center, Instituto Meteorológico e Hidrológico Sueco/SMHI, Norrkoping, Suécia
# 
# MPI_ESM1_2_XR	Alemanha, Instituto Max Planck de Meteorologia, Hamburgo 20146, Alemanha
# Variáveis: Temperatura máxima,Temperatura minima, umidade max, umidade min.
# Período: 14 anos

# %%
#### Bibliotecas 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
from sklearn.metrics import mean_squared_error
import seaborn as sns 


from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# %% [markdown]
# ### importação do Dataset

# %%
# Load the dataset and ensure proper parsing of columns
clima_df = pd.read_csv('Uige_Clima.csv', sep=';', header=0)

# Convert 'periodo_tempo' to datetime and set it as the index
clima_df['periodo_tempo'] = pd.to_datetime(clima_df['periodo_tempo'], format='%d/%m/%y', errors='coerce')
clima_df.set_index('periodo_tempo', inplace=True)

# Convert 'temperatura_max' and 'temperatura_min' to numeric, handling commas and coercing errors
clima_df['temperatura_max'] = pd.to_numeric(clima_df['temperatura_max'].str.replace(',', '.'), errors='coerce')
clima_df['temperatura_min'] = pd.to_numeric(clima_df['temperatura_min'].str.replace(',', '.'), errors='coerce')

# Display the first few rows of the dataframe
clima_df.head()


# %%

# Exibir as primeiras linhas do dataframe atualizado
clima_df.head()

# %%
# Ver os tipos de dados
clima_df.dtypes


# %%

#clima_df.index = pd.to_datatime(clima_df.index)
clima_df.index

# %% [markdown]
# #### verificar se há dados nulos

# %%
def list_and_visualize_missing_data(dataset):
    # Listing total null items and its percent with respect to all nulls
    total = dataset.isnull().sum().sort_values(ascending=False)
    percent = ((dataset.isnull().sum())/(dataset.isnull().count())).sort_values(ascending=False)
    percent = percent*100
    
    print('Total de dados nulos : \n',total)
    print('% dados nulos : \n',percent)

list_and_visualize_missing_data(clima_df)

# %%
import nbformat
%pip install nbformat>=4.2.0

# %%


import plotly.express as px
# Gráfico dinâmico para todas as variáveis do dataframe clima_df
fig = px.line(clima_df, x=clima_df.index, y=clima_df.columns, title="Tendência e Sazonalidade dos Dados Meteorológicos")
fig.update_layout(xaxis_title="Data", yaxis_title="Valores", legend_title="Variáveis")
fig.show()

# Gráfico dinâmico para o período de 2023 a 2024
clima_df_filtered = clima_df['2023':'2024']
fig_filtered = px.line(clima_df_filtered, x=clima_df_filtered.index, y=clima_df_filtered.columns, title="Visão Detalhada (2023-2024)")
fig_filtered.update_layout(xaxis_title="Data", yaxis_title="Valores", legend_title="Variáveis")
fig_filtered.show()

# %%
#Dividir o conjunto de dados em dados de treinamento e teste
train_df = clima_df['2010':'2021'].resample('ME').mean()
train_df = train_df.fillna(train_df.mean())
test_df = clima_df['2022':'2024'].resample('ME').mean()
test_df = test_df.fillna(test_df.mean())


# %%
train_df.head()

# %%
# visualizar as últimas linhas do arquivo
train_df.tail()

# %%
#Visualizar os dados de treinamento
test_df.head()

# %%
#Visualizar últimos dados de treinamento
test_df.tail()

# %%
#Visualizar os dados de teste
train_df.describe()

# %%
#Visualizar os dados de teste
test_df.describe()

# %%
# verificar a média de rolamento e o desvio padrão de rolamento
def plot_rolling_mean_std(ts):
    rolling_mean = ts.rolling(12).mean()
    rolling_std = ts.rolling(12).std()
    plt.figure(figsize=(22,10))

    plt.plot(ts, label='Média Atual')
    plt.plot(rolling_mean, label='Média Móvel')
    plt.plot(rolling_std, label = 'Std contínuo')
    plt.xlabel("Data")
    plt.ylabel("Temperatura")
    plt.title('Média Móvel & Desvio Padrão Contínuo')
    plt.legend()
    plt.show()


# %%
plt.figure(figsize=(22, 10))

# Temperatura mínima
plt.plot(train_df.index, train_df['temperatura_min'], label='Temperatura Mínima', color='blue', alpha=0.7)
plt.plot(train_df.index, train_df['temperatura_min'].rolling(12).mean(), label='Média Móvel Mínima', color='blue', linestyle='--')
plt.plot(train_df.index, train_df['temperatura_min'].rolling(12).std(), label='Desvio Padrão Mínima', color='blue', linestyle=':')

# Temperatura máxima
plt.plot(train_df.index, train_df['temperatura_max'], label='Temperatura Máxima', color='red', alpha=0.7)
plt.plot(train_df.index, train_df['temperatura_max'].rolling(12).mean(), label='Média Móvel Máxima', color='red', linestyle='--')
plt.plot(train_df.index, train_df['temperatura_max'].rolling(12).std(), label='Desvio Padrão Máxima', color='red', linestyle=':')

plt.xlabel("Data")
plt.ylabel("Temperatura")
plt.title('Média Móvel & Desvio Padrão - Temperatura Máxima e Mínima')
plt.legend()
plt.show()


# %%
from statsmodels.tsa.seasonal import STL

def plot_stl_result(stl_result, title):
    fig, axes = plt.subplots(4, 1, figsize=(18, 10), sharex=True)
    stl_result.observed.plot(ax=axes[0], color='black')
    axes[0].set_ylabel('Observado')
    axes[0].set_title(f'{title} - Observado')
    stl_result.trend.plot(ax=axes[1], color='blue')
    axes[1].set_ylabel('Tendência')
    axes[1].set_title('Tendência')
    stl_result.seasonal.plot(ax=axes[2], color='green')
    axes[2].set_ylabel('Sazonalidade')
    axes[2].set_title('Sazonalidade')
    stl_result.resid.plot(ax=axes[3], color='red')
    axes[3].set_ylabel('Resíduo')
    axes[3].set_title('Resíduo')
    plt.xlabel('Data')
    plt.tight_layout()
    plt.show()

# STL para temperatura_max
stl_max = STL(train_df['temperatura_max'], seasonal=13)
stl_result_max = stl_max.fit()
plot_stl_result(stl_result_max, 'STL - Temperatura Máxima')

# STL para temperatura_min
stl_min = STL(train_df['temperatura_min'], seasonal=13)
stl_result_min = stl_min.fit()
plot_stl_result(stl_result_min, 'STL - Temperatura Mínima')

# %%
# Gerar os gráficos de ACF e PACF para a temperatura média, máxima e mínima mensal
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# ACF e PACF para temperatura máxima
plot_acf(train_df["temperatura_max"], lags=48, ax=axes[0, 0])
axes[0, 0].set_title("ACF - Temperatura Máxima Mensal")
plot_pacf(train_df["temperatura_max"], lags=48, ax=axes[0, 1], method="ywm")
axes[0, 1].set_title("PACF - Temperatura Máxima Mensal")

# ACF e PACF para temperatura mínima
plot_acf(train_df["temperatura_min"], lags=48, ax=axes[1, 0])
axes[1, 0].set_title("ACF - Temperatura Mínima Mensal")
plot_pacf(train_df["temperatura_min"], lags=48, ax=axes[1, 1], method="ywm")
axes[1, 1].set_title("PACF - Temperatura Mínima Mensal")

# Ajustar layout
plt.tight_layout()
plt.show()


# %%
# Plotar temperaturas máxima e mínima do conjunto de treinamento e teste no mesmo gráfico
plt.figure(figsize=(22, 10))

# Dados de treinamento
plt.plot(train_df.index, train_df.temperatura_max, label='Temperatura Máxima (Treinamento)', color='blue')
plt.plot(train_df.index, train_df.temperatura_min, label='Temperatura Mínima (Treinamento)', color='green')

# Dados de teste
plt.plot(test_df.index, test_df.temperatura_max, label='Temperatura Máxima (Teste)', color='orange', linestyle='--')
plt.plot(test_df.index, test_df.temperatura_min, label='Temperatura Mínima (Teste)', color='red', linestyle='--')

# Configurações do gráfico
plt.xlabel("Data")
plt.ylabel("Temperatura")
plt.title("Temperatura Máxima e Mínima - Dados de Treinamento e Teste")
plt.legend()
plt.show()

# %%
import plotly.express as px

# Gerar dados trimestrais para treino e teste
train_quarterly = train_df.resample('Q').mean()
train_quarterly['quarter'] = train_quarterly.index.to_period('Q')
test_quarterly = test_df.resample('Q').mean()
test_quarterly['quarter'] = test_quarterly.index.to_period('Q')

# Concatenar os dados trimestrais de treino e teste, adicionando uma coluna de origem
train_quarterly_plot = train_quarterly.copy()
train_quarterly_plot['Tipo'] = 'Treinamento'
test_quarterly_plot = test_quarterly.copy()
test_quarterly_plot['Tipo'] = 'Teste'

quarterly_all = pd.concat([train_quarterly_plot, test_quarterly_plot])
quarterly_all = quarterly_all.reset_index(drop=True)

# Converter a coluna 'quarter' para string para evitar erro de serialização
quarterly_all['quarter'] = quarterly_all['quarter'].astype(str)

# Gráfico dinâmico
fig = px.line(
    quarterly_all,
    x='quarter',
    y=['temperatura_max', 'temperatura_min'],
    color='Tipo',
    labels={'value': 'Temperatura', 'quarter': 'Trimestre', 'variable': 'Variável'},
    title='Temperatura Máxima e Mínima - Dados Trimestrais'
)
fig.update_layout(xaxis_tickangle=45)
fig.show()


# %% [markdown]
# # ===== Previsão de Temperatura Máxima =====

# %%
def fit_prophet(train_df):
    # Rename columns to match Prophet's requirements
    train_df_renamed = train_df.reset_index().rename(columns={'periodo_tempo': 'ds', 'temperatura_max': 'y'})
    
    # Initialize the Prophet model
    m = Prophet()
    
    # Fit the model
    m.fit(train_df_renamed)
    
    return m

# Fit the Prophet model using the training data
m = fit_prophet(train_df)

# %%
# Preparar os dados para o Prophet
df_prophet = train_df.reset_index()
df_prophet.rename(columns={'periodo_tempo': 'ds', 'temperatura_max': 'y'}, inplace=True)

# Criar e ajustar o modelo Prophet
modelo_prophet_max = Prophet()
modelo_prophet_max.fit(df_prophet)

# Criar um dataframe para as datas futuras
future_dates = modelo_prophet_max.make_future_dataframe(periods=108, freq='ME')

# Fazer a previsão
forecast_prophet = modelo_prophet_max.predict(future_dates)

# Visualizar a previsão
fig = modelo_prophet_max.plot(forecast_prophet)
plt.title('Previsão com Prophet')
plt.xlabel('Data')
plt.ylabel('Temperatura Máxima')
plt.show()


# %% [markdown]
# ## Previsão futura

# %%
import plotly.graph_objects as go

# Gerar datas futuras para 2025-2030
future_max = modelo_prophet_max.make_future_dataframe(periods=120, freq='ME')  # 66 meses de 2025-01 a 2030-06

# Fazer a previsão
previsao = modelo_prophet_max.predict(future_max)
previsao = previsao[(previsao['ds'] >= '2024-12-31') & (previsao['ds'] <= '2030-12-31')]

# Concatenar dados históricos de temperatura mínima
historico = pd.concat([
    train_df.reset_index()[['periodo_tempo', 'temperatura_max']].rename(columns={'periodo_tempo': 'ds', 'temperatura_max': 'yhat'}),
    test_df.reset_index()[['periodo_tempo', 'temperatura_max']].rename(columns={'periodo_tempo': 'ds', 'temperatura_max': 'yhat'})
])

# Ajustar previsão para continuidade visual
ultimo_valor_historico = historico['yhat'].iloc[-1]
primeiro_valor_previsao = previsao['yhat'].iloc[0]
ajuste = ultimo_valor_historico - primeiro_valor_previsao
previsao_ajustada = previsao.copy()
previsao_ajustada['yhat'] = previsao_ajustada['yhat'] + ajuste

# Plotar
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=historico['ds'], y=historico['yhat'],
    mode='lines', name='Histórico', line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=previsao_ajustada['ds'], y=previsao_ajustada['yhat'],
    mode='lines', name='Previsão (2025-2030)', line=dict(color='orange')
))
fig.update_layout(
    title='Previsão da Temperatura Máxima (2025-2030)',
    xaxis_title='Data',
    yaxis_title='Temperatura Max',
    legend_title='Período'
)
fig.show()


# %% [markdown]
# # ==Previsão da Temperatura Mínima ==

# %%
# Preparar os dados para o Prophet
df_prophet = train_df.reset_index()
df_prophet.rename(columns={'periodo_tempo': 'ds', 'temperatura_min': 'y'}, inplace=True)

# Criar e ajustar o modelo Prophet
modelo_prophet_min = Prophet()
modelo_prophet_min.fit(df_prophet)

# Criar um dataframe para as datas futuras
future_dates = modelo_prophet_min.make_future_dataframe(periods=108, freq='ME')

# Fazer a previsão
forecast_prophet = modelo_prophet_min.predict(future_dates)

# Visualizar a previsão
fig = modelo_prophet_min.plot(forecast_prophet)
plt.title('Previsão com Prophet')
plt.xlabel('Data')
plt.ylabel('Temperatura Mínima')
plt.show()


# %%
def fit_prophet(train_df):
    # Rename columns to match Prophet's requirements
    train_df_renamed = train_df.reset_index().rename(columns={'periodo_tempo': 'ds', 'temperatura_min': 'y'})
    
    # Initialize the Prophet model
    modelo_prophet_min = Prophet()
    
    # Fit the model
    modelo_prophet_min.fit(train_df_renamed)
    
    return modelo_prophet_min

# Fit the Prophet model using the training data
modelo_prophet_min = fit_prophet(train_df)

# %%
import plotly.graph_objects as go

# Gerar datas futuras para 2025-2030
future_min = modelo_prophet_min.make_future_dataframe(periods=120, freq='ME')  # 66 meses de 2025-01 a 2030-06

# Fazer a previsão
previsao = modelo_prophet_min.predict(future_min)
previsao = previsao[(previsao['ds'] >= '2024-12-31') & (previsao['ds'] <= '2030-12-31')]

# Concatenar dados históricos de temperatura mínima
historico = pd.concat([
    train_df.reset_index()[['periodo_tempo', 'temperatura_min']].rename(columns={'periodo_tempo': 'ds', 'temperatura_min': 'yhat'}),
    test_df.reset_index()[['periodo_tempo', 'temperatura_min']].rename(columns={'periodo_tempo': 'ds', 'temperatura_min': 'yhat'})
])

# Ajustar previsão para continuidade visual
ultimo_valor_historico = historico['yhat'].iloc[-1]
primeiro_valor_previsao = previsao['yhat'].iloc[0]
ajuste = ultimo_valor_historico - primeiro_valor_previsao
previsao_ajustada = previsao.copy()
previsao_ajustada['yhat'] = previsao_ajustada['yhat'] + ajuste

# Plotar
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=historico['ds'], y=historico['yhat'],
    mode='lines', name='Histórico', line=dict(color='blue')
))
fig.add_trace(go.Scatter(
    x=previsao_ajustada['ds'], y=previsao_ajustada['yhat'],
    mode='lines', name='Previsão (2025-2030)', line=dict(color='orange')
))
fig.update_layout(
    title='Previsão da Temperatura Mínima (2025-2030)',
    xaxis_title='Data',
    yaxis_title='Temperatura Mínima',
    legend_title='Período'
)
fig.show()


# %% [markdown]
# ## Avaliação e resumo do modelo
# 

# %%
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Criar e ajustar o modelo Prophet para temperatura máxima
df_prophet_max = train_df.reset_index().rename(columns={'periodo_tempo': 'ds', 'temperatura_max': 'y'})
modelo_prophet_max = Prophet()
modelo_prophet_max.fit(df_prophet_max)

# Fazer a previsão para temperatura máxima
future_dates_max = modelo_prophet_max.make_future_dataframe(periods=len(test_df), freq='ME')
forecast_prophet_max = modelo_prophet_max.predict(future_dates_max)

# Criar e ajustar o modelo Prophet para temperatura mínima
df_prophet_min = train_df.reset_index().rename(columns={'periodo_tempo': 'ds', 'temperatura_min': 'y'})
modelo_prophet_min = Prophet()
modelo_prophet_min.fit(df_prophet_min)

# Fazer a previsão para temperatura mínima
future_dates_min = modelo_prophet_min.make_future_dataframe(periods=len(test_df), freq='ME')
forecast_prophet_min = modelo_prophet_min.predict(future_dates_min)

# Reindexar as previsões do Prophet para corresponder ao índice do conjunto de teste
forecast_test_max = forecast_prophet_max.set_index('ds').reindex(test_df.index)
forecast_test_min = forecast_prophet_min.set_index('ds').reindex(test_df.index)

# Remover linhas com valores NaN
valid_max = ~forecast_test_max['yhat'].isna() & ~test_df['temperatura_max'].isna()
valid_min = ~forecast_test_min['yhat'].isna() & ~test_df['temperatura_min'].isna()

# Calcular as métricas de erro para temperatura_max
mae_max = mean_absolute_error(test_df['temperatura_max'][valid_max], forecast_test_max['yhat'][valid_max])
mse_max = mean_squared_error(test_df['temperatura_max'][valid_max], forecast_test_max['yhat'][valid_max])
rmse_max = np.sqrt(mse_max)

# Calcular as métricas de erro para temperatura_min
mae_min = mean_absolute_error(test_df['temperatura_min'][valid_min], forecast_test_min['yhat'][valid_min])
mse_min = mean_squared_error(test_df['temperatura_min'][valid_min], forecast_test_min['yhat'][valid_min])
rmse_min = np.sqrt(mse_min)

# Exibir o resumo da qualidade do modelo
print("Resumo da Qualidade do Modelo Prophet:")
print("Temperatura Máxima:")
print(f"Mean Absolute Error (MAE): {mae_max:.2f}")
print(f"Mean Squared Error (MSE): {mse_max:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_max:.2f}")
print("\nTemperatura Mínima:")
print(f"Mean Absolute Error (MAE): {mae_min:.2f}")
print(f"Mean Squared Error (MSE): {mse_min:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse_min:.2f}")

# %% [markdown]
# O modelo utilizado é adequado para as séries de temperatura máxima e mínima, considerando:
# 
# Indica boa precisão, com erros médios em torno de 1.4 °C.
# 
# Boa proximidade entre MAE e RMSE (indicando poucos outliers)
# 
# Simplicidade do Prophet e sua capacidade de lidar com tendências e sazonalidade.

# %% [markdown]
# ## Salvar dados da previsão

# %%
# Gerar datas mensais para o período de 2025-01-01 a 2030-12-31
future_months = pd.date_range(start='2025-01-01', end='2030-12-31', freq='D')
df_future_months = pd.DataFrame({'ds': future_months})

# Fazer previsões para máxima e mínima
forecast_max = modelo_prophet_max.predict(df_future_months)
forecast_min = modelo_prophet_min.predict(df_future_months)

# Unir as previsões em um único DataFrame
previsao_unida = pd.DataFrame({
	'periodo_tempo': forecast_max['ds'],
	'temperatura_max': forecast_max['yhat'],
	'temperatura_min': forecast_min['yhat']
})

# Salvar apenas o período de 2025 a 2030
previsao_unida = previsao_unida[(previsao_unida['periodo_tempo'] >= '2025-01-01') & (previsao_unida['periodo_tempo'] <= '2030-12-31')]

previsao_unida.to_csv('previsao_temperatura_max_min_2025_2030.csv', index=False)

# %% [markdown]
# # Previsao em Trimestre MINIMA
# 

# %%
# Agrupar a previsão ajustada por trimestre e calcular a média
previsao_ajustada['quarter'] = previsao_ajustada['ds'].dt.to_period('Q')
previsao_trimestral = previsao_ajustada.groupby('quarter')['yhat'].mean().reset_index()

# Visualizar os primeiros trimestres
previsao_trimestral.head()


# %%
import plotly.express as px

# Converter a coluna 'quarter' para string para evitar erro de serialização
previsao_trimestral_plot = previsao_trimestral.copy()
previsao_trimestral_plot['quarter'] = previsao_trimestral_plot['quarter'].astype(str)

# Gráfico dinâmico da previsão trimestral da temperatura mínima (2025-2030)
fig = px.line(
    previsao_trimestral_plot,
    x='quarter',
    y='yhat',
    title='Previsão Trimestral da Temperatura Mínima (2025-2030)',
    labels={'quarter': 'Trimestre', 'yhat': 'Temperatura Mínima (°C)'},
    markers=True
)
fig.update_layout(
    xaxis_tickangle=45,
    xaxis=dict(tickmode='linear'),
    yaxis=dict(tickformat=".1f"),
    width=900,
    height=500,
    font=dict(size=16),
    plot_bgcolor='white',
    margin=dict(l=60, r=30, t=60, b=60)
)
fig.update_traces(line=dict(width=3), marker=dict(size=8, color='orange'))
fig.show()

# %%


# %% [markdown]
# # Previsao trimestral da temperatura maxima

# %%
import plotly.graph_objects as go

# Gerar datas futuras para 2025-2030
future_max_tr = modelo_prophet_max.make_future_dataframe(periods=102, freq='Q')  # 66 meses de 2025-01 a 2030-06

# Fazer a previsão
previsao = modelo_prophet_max.predict(future_max_tr)
previsao = previsao[(previsao['ds'] >= '2025-01-01') & (previsao['ds'] <= '2030-12-31')]



# Plotar
fig = go.Figure()
fig.add_trace(go.Scatter(
         x=previsao['ds'], y=previsao['yhat'],
        mode='lines', name='Previsão (2025-2030)', line=dict(color='orange')
))
fig.update_layout(
    title='Previsão da Temperatura Máxima (2025-2030)',
    xaxis_title='Data',
    yaxis_title='Temperatura Máxima',
    legend_title='Período'
)
fig.show()


# %% [markdown]
# ## Previsão por trimestre

# %%
import plotly.express as px

# Gerar datas futuras para 5 anos (2025-2030)
future_5_years = modelo_prophet_max.make_future_dataframe(periods=108, freq='ME')  # 6 anos, mas vamos filtrar 2025-2030

# Fazer a previsão
forecast_5_years = modelo_prophet_max.predict(future_5_years)

# Filtrar apenas o período de previsão (2025-2030)
# Filtrar apenas o período de previsão (2025-2030)
forecast_5_years_filtered = forecast_5_years[(forecast_5_years['ds'] >= '2025-01-01') & (forecast_5_years['ds'] <= '2030-12-31')]

# Adicionar coluna de trimestre
forecast_5_years_filtered['quarter'] = forecast_5_years_filtered['ds'].dt.to_period('Q')

# Agrupar por trimestre e calcular a média da previsão
forecast_5_years_quarterly = forecast_5_years_filtered.groupby('quarter')['yhat'].mean().reset_index()

# Converter a coluna 'quarter' para string para evitar erro de serialização
forecast_5_years_quarterly['quarter'] = forecast_5_years_quarterly['quarter'].astype(str)

# Gráfico dinâmico da previsão
fig_5y = px.line(
    forecast_5_years_quarterly,
    x='quarter',
    y='yhat',
    title='Previsão com Prophet - Temperatura Máxima (2025-2030)',
    labels={'quarter': 'Trimestre', 'yhat': 'Temperatura Máxima'}
)
fig_5y.show()


# %% [markdown]
# # Previsão do Mês atual

# %% [markdown]
# Sabendo que a partir deo mês de maio começa o tempo seco em Angola, procuramos saber como será o comportamento do clima
# para esta época na cidade do Uíge

# %%
import numpy as np

# Gerar datas diárias para maio de 2025
maio_2025_diario = pd.date_range(start='2025-05-01', end='2025-07-30', freq='D')
df_maio_2025_diario = pd.DataFrame({'ds': maio_2025_diario})

# Prever temperaturas máximas e mínimas diárias
forecast_max_diario = modelo_prophet_max.predict(df_maio_2025_diario)
forecast_min_diario = modelo_prophet_min.predict(df_maio_2025_diario)

# Gráfico diário
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=forecast_max_diario['ds'], y=forecast_max_diario['yhat'],
    mode='lines+markers', name='Temp. Máxima (Diária)', line=dict(color='red')
))
fig.add_trace(go.Scatter(
    x=forecast_min_diario['ds'], y=forecast_min_diario['yhat'],
    mode='lines+markers', name='Temp. Mínima (Diária)', line=dict(color='blue')
))
fig.update_layout(
    title='Previsão Diária de Temperatura Máxima e Mínima - Maio - Junho 2025',
    xaxis_title='Data',
    yaxis_title='Temperatura (°C)',
    legend_title='Legenda'
)
fig.show()

# Gerar datas semanais para maio de 2025 (toda segunda-feira)
maio_2025_semanal = pd.date_range(start='2025-05-01', end='2025-07-30', freq='W-MON')
df_maio_2025_semanal = pd.DataFrame({'ds': maio_2025_semanal})

# Prever temperaturas máximas e mínimas semanais
forecast_max_semanal = modelo_prophet_max.predict(df_maio_2025_semanal)
forecast_min_semanal = modelo_prophet_min.predict(df_maio_2025_semanal)

import plotly.colors

def temp_to_color(temp, vmin, vmax, colorscale):
    norm = (temp - vmin) / (vmax - vmin)
    idx = np.clip((norm * (len(colorscale) - 1)).astype(int), 0, len(colorscale) - 1)
    return [colorscale[i] for i in idx]

# Escala de cor para máxima: 'Purples' (escala roxa)
max_colorscale = plotly.colors.sequential.Purples
max_temp = forecast_max_semanal['yhat']
max_colors = temp_to_color(max_temp.values, max_temp.min(), max_temp.max(), max_colorscale)

# Escala de cor para mínima: 'Magma' (escala roxa-avermelhada)
min_colorscale = plotly.colors.sequential.Magma
min_temp = forecast_min_semanal['yhat']
min_colors = temp_to_color(min_temp.values, min_temp.min(), min_temp.max(), min_colorscale)

fig2 = go.Figure()
fig2.add_trace(go.Bar(
    y=forecast_max_semanal['ds'], x=forecast_max_semanal['yhat'],
    name='Temp. Máxima (Semanal)', marker_color=max_colors, orientation='h'
))
fig2.add_trace(go.Bar(
    y=forecast_min_semanal['ds'], x=forecast_min_semanal['yhat'],
    name='Temp. Mínima (Semanal)', marker_color=min_colors, orientation='h'
))
fig2.update_layout(
    title='Previsão Semanal de Temperatura Máxima e Mínima - Maio - Junho 2025',
    yaxis_title='Data (Início da Semana)',
    xaxis_title='Temperatura (°C)',
    legend_title='Legenda',
    barmode='group',
    height=600
)
fig2.show()


# %%
import matplotlib.pyplot as plt

# Filtrar e ordenar apenas trimestres a partir de 2025
minima_trimestre_2025 = previsao_trimestral_plot[previsao_trimestral_plot['quarter'] >= '2025Q1'].copy()
maxima_trimestre_2025 = forecast_5_years_quarterly[forecast_5_years_quarterly['quarter'] >= '2025Q1'].copy()

minima_trimestre_2025 = minima_trimestre_2025.sort_values('quarter')
maxima_trimestre_2025 = maxima_trimestre_2025.sort_values('quarter')

# Calcular a diferença entre o início e o fim da previsão trimestral para mínima
minima_inicio = minima_trimestre_2025['yhat'].iloc[0]
minima_fim = minima_trimestre_2025['yhat'].iloc[-1]
minima_delta = minima_fim - minima_inicio

# Calcular a diferença entre o início e o fim da previsão trimestral para máxima
maxima_inicio = maxima_trimestre_2025['yhat'].iloc[0]
maxima_fim = maxima_trimestre_2025['yhat'].iloc[-1]
maxima_delta = maxima_fim - maxima_inicio

# Exibir os resultados
print(f"Tendência Temperatura Mínima (2025Q1 a 2030Q4): {minima_inicio:.2f}°C → {minima_fim:.2f}°C | Δ = {minima_delta:.2f}°C")
print(f"Tendência Temperatura Máxima (2025Q1 a 2030Q4): {maxima_inicio:.2f}°C → {maxima_fim:.2f}°C | Δ = {maxima_delta:.2f}°C")

# Gráfico comparativo das tendências a partir de 2025
plt.figure(figsize=(12, 6))
plt.plot(minima_trimestre_2025['quarter'], minima_trimestre_2025['yhat'], label='Temperatura Mínima (trimestral)', marker='o')
plt.plot(maxima_trimestre_2025['quarter'], maxima_trimestre_2025['yhat'], label='Temperatura Máxima (trimestral)', marker='o')
plt.xlabel('Trimestre')
plt.ylabel('Temperatura (°C)')
plt.title('Tendência Trimestral das Temperaturas Máxima e Mínima (2025-2030)')
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# %%
import matplotlib.pyplot as plt

# Médias anuais das temperaturas mínima e máxima
media_anual = train_df.resample('Y').mean()
media_anual_test = test_df.resample('Y').mean()
media_anual_total = pd.concat([media_anual, media_anual_test])

# Ajustar o índice para o ano
media_anual_total.index = media_anual_total.index.year

plt.figure(figsize=(12, 6))
plt.plot(media_anual_total.index, media_anual_total['temperatura_max'], marker='o', label='Temperatura Máxima Média Anual')
plt.plot(media_anual_total.index, media_anual_total['temperatura_min'], marker='o', label='Temperatura Mínima Média Anual')

# Destacar início e fim
plt.scatter(media_anual_total.index[0], media_anual_total['temperatura_max'].iloc[0], color='red', zorder=5)
plt.scatter(media_anual_total.index[-1], media_anual_total['temperatura_max'].iloc[-1], color='red', zorder=5)
plt.scatter(media_anual_total.index[0], media_anual_total['temperatura_min'].iloc[0], color='blue', zorder=5)
plt.scatter(media_anual_total.index[-1], media_anual_total['temperatura_min'].iloc[-1], color='blue', zorder=5)

plt.title('Evolução da Temperatura Máxima e Mínima Média Anual (2010-2024)')
plt.xlabel('Ano')
plt.ylabel('Temperatura (°C)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Cálculo da variação
delta_max = media_anual_total['temperatura_max'].iloc[-1] - media_anual_total['temperatura_max'].iloc[0]
delta_min = media_anual_total['temperatura_min'].iloc[-1] - media_anual_total['temperatura_min'].iloc[0]

print(f"Variação da temperatura máxima média anual de 2010 a 2024: {media_anual_total['temperatura_max'].iloc[0]:.2f}°C → {media_anual_total['temperatura_max'].iloc[-1]:.2f}°C | Δ = {delta_max:.2f}°C")
print(f"Variação da temperatura mínima média anual de 2010 a 2024: {media_anual_total['temperatura_min'].iloc[0]:.2f}°C → {media_anual_total['temperatura_min'].iloc[-1]:.2f}°C | Δ = {delta_min:.2f}°C")

if delta_max > 0:
    print("A temperatura máxima média anual aumentou no período.")
else:
    print("A temperatura máxima média anual reduziu no período.")

if delta_min > 0:
    print("A temperatura mínima média anual aumentou no período.")
else:
    print("A temperatura mínima média anual reduziu no período.")

# %% [markdown]
# # Resultados e Discussão
# 
# ## Identificação da Sazonalidade
# 
# A análise dos gráficos de ACF e PACF revela uma forte tendência anual com defasagem de 12 meses, indicando uma sazonalidade clara. Isso sugere que a temperatura máxima de um determinado mês tende a se assemelhar à do mesmo mês do ano anterior. O PACF apresenta valores significativos nas defasagens 1 e 12, confirmando a existência de dependência tanto de curto quanto de longo prazo. A ACF mostra picos nos meses 12 e 24, reforçando o padrão sazonal anual. A sazonalidade é evidente tanto nas temperaturas máximas quanto nas mínimas, o que é compatível com o clima tropical da região do Uíge.
# 
# ## Tendência de Longo Prazo
# 
# A decomposição STL da temperatura média máxima no Uíge apresenta três componentes principais: tendência, sazonalidade e resíduo. A série observada revela flutuações sazonais recorrentes ao longo do tempo. A tendência indica um aumento gradual da temperatura entre 2010 e 2015, possivelmente associado a alterações climáticas regionais ou eventos específicos. A sazonalidade é regular e se repete anualmente, com amplitude estável de aproximadamente 1,5°C. Os resíduos são pequenos, mas entre 2020 e 2022 houve desvios significativos, sugerindo eventos climáticos extremos.
# 
# Para a temperatura mínima, os dados mostram:
# 
# * Linha preta: valores observados, com clara variabilidade sazonal;
# * Linha azul (tendência): leve declínio ao longo dos anos, indicando uma possível queda nas temperaturas mínimas;
# * Linha verde (sazonalidade): padrões sazonais bem definidos;
# * Linha vermelha (resíduo): variações aleatórias, sem padrão claro, sugerindo que o modelo se ajusta bem aos dados.
# 
# Essas observações podem estar associadas a **mudanças climáticas globais, desmatamento local** e **urbanização crescente no Uíge**, reforçando a importância de estratégias de monitoramento e adaptação ambiental.
# 
# ## Ciclos Relevantes
# 
# Além da sazonalidade de 12 meses, foram identificados ciclos climáticos moderados de 3 a 5 anos, como observado na tendência entre 2012 e 2018. Esses ciclos podem estar ligados a fenômenos como El Niño e La Niña, que influenciam o clima na África Central.
# 
# As temperaturas mínimas apresentam variações entre 14°C e 18°C. A tendência mostra declínios específicos ao longo do tempo, enquanto a sazonalidade é consistente. Embora esses ciclos plurianuais sejam menos regulares, sua identificação é fundamental para prever extremos de temperatura.
# 
# ## Anomalias Climáticas
# 
# Foram identificadas anomalias de temperatura nos últimos anos:
# 
# * Em julho de 2021, houve um pico de 34°C;
# * Em julho de 2022, a temperatura foi de 33°C;
# * Em julho de 2023, registrou-se uma queda significativa (\~4°C), com as temperaturas mais baixas dos últimos quatro anos, tanto para máximas quanto para mínimas;
# * Em julho de 2024, as temperaturas voltaram a subir, alcançando novamente 33°C.
# 
# A previsão feita com o modelo Prophet indica uma temperatura máxima de 30,7°C para julho de 2025. Essas anomalias indicam possíveis falhas no regime de chuvas, eventos extremos ou mudanças no uso do solo. A análise dos resíduos da decomposição corrobora essas irregularidades.
# 
# ## Variação Climática
# 
# Observa-se uma elevação das médias anuais entre 2010 e 2024:
# 
# * Temperatura máxima média: de 28,78°C para 29,37°C (Δ = +0,59°C);
# * Temperatura mínima média: de 17,87°C para 19,84°C (Δ = +1,98°C).
# 
# Esses dados confirmam um aumento significativo das temperaturas, particularmente das mínimas, o que pode ter implicações importantes para a saúde pública e agricultura.
# 
# ## Previsão
# 
# As previsões apontam:
# 
# * A temperatura máxima apresentará aumento entre o 1º trimestre de 2025 e o 4º trimestre de 2030: de 29,11°C para 30,02°C (Δ = +0,92°C);
# * A temperatura mínima aumentará de 20,99°C para 21,34°C no mesmo período (Δ = +0,35°C).
# 
# Prevê-se que os picos de temperatura máxima ocorram no 3º trimestre dos anos entre 2025 e 2029, e no 4º trimestre de 2030. Já o ponto mínimo da temperatura mínima será registrado no mês de junho, com previsão de 12°C.
# 
# ## Aplicações Práticas e Impactos
# 
# As variações e anomalias climáticas observadas têm implicações diretas em diversos setores:
# 
# * **Agricultura**: mudanças no calendário agrícola e riscos maiores de secas ou geadas inesperadas;
# * **Saúde pública**: aumento de doenças relacionadas ao calor e à umidade;
# * **Planejamento urbano**: necessidade de adaptar construções e infraestrutura para extremos climáticos;
# * **Gestão ambiental**: reforça a importância de políticas públicas de reflorestamento, controle da urbanização e uso sustentável do solo.
# 
# Portanto, compreender essas dinâmicas climáticas é essencial para desenvolver **estratégias locais de mitigação e adaptação**, especialmente em regiões vulneráveis como o Uíge.
# 
# 

# %% [markdown]
# 


