#!/usr/bin/env python
# coding: utf-8

# # Neurostock - Team Galaxy - Samsung Innovation Campus 2024 - 2025

# ## Paso 1: Preparación de los Datos

# In[18]:


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset

data_folder = r"C:\Users\inesc\Desktop\neural_catch\Data"
companies = [
    "AAPL",  # 0
    "ADBE",  # 1
    "AMZN",  # 2
    "CSCO",  # 3
    "DELL",  # 4
    "GOOGL",  # 5
    "IBM",  # 6
    "INTC",  # 7
    "META",  # 8
    "MSFT",  # 9
    "NOK",  # 10
    "NTDOY",  # 11
    "NVDA",  # 12
    "NFLX",  # 13
    "ORCL",  # 14
    "QCOM",  # 15
    "SONY",  # 16
    "SSNLF",  # 17
    "TSLA",  # 18
]

data_paths = {}
# Generación de rutas automatizada para cada Data de cada empresa
for ticker in companies:
    data_paths[ticker] = {
        "historical_data": f"{data_folder}/Historical_Data/{ticker}_historical_data.csv".replace(
            "\\", "/"
        ),
        "balance_sheet": f"{data_folder}/Balance_Sheet_Data/{ticker}_balance_sheet.csv".replace(
            "\\", "/"
        ),
        "cash_flow": f"{data_folder}/Cash_Flow_Data/{ticker}_cash_flow.csv".replace(
            "\\", "/"
        ),
        "income_statement": f"{data_folder}/Income_Statement_Data/{ticker}_income_statement.csv".replace(
            "\\", "/"
        ),
        "financial_ratios": f"{data_folder}/Financial_Ratios_Data/{ticker}_financial_ratios.csv".replace(
            "\\", "/"
        ),
    }


# ### Funciones para manejar declaraciones de Dataframes

# In[19]:


def load_Historical_Data(ticker, data=data_paths):
    historical_csv_path = data_paths[ticker]["historical_data"]
    return pd.read_csv(historical_csv_path)


def load_Balance_Sheet(ticker, data=data_paths):
    Balance_Sheet_csv_path = data_paths[ticker]["balance_sheet"]
    return pd.read_csv(Balance_Sheet_csv_path)


def load_Cash_Flow(ticker, data=data_paths):
    Cash_Flow_csv_path = data_paths[ticker]["cash_flow"]
    return pd.read_csv(Cash_Flow_csv_path)


def load_Income_Statement(ticker, data=data_paths):
    Income_Statement_csv_path = data_paths[ticker]["income_statement"]
    return pd.read_csv(Income_Statement_csv_path)


def load_Financial_Ratios(ticker, data=data_paths):
    Financial_Ratios_csv_path = data_paths[ticker]["financial_ratios"]
    return pd.read_csv(Financial_Ratios_csv_path)


# ### Limpieza de Dataset

# In[20]:


# Dataset de prueba
ticker = companies[17]
historical_data = load_Historical_Data(ticker)

# Verificar que las columnas sean las esperadas
expected_columns = [
    "Date",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "Dividends",
    "Stock Splits",
    "Adj Close",
]
actual_columns = historical_data.columns.tolist()

if actual_columns == expected_columns:
    historical_data = historical_data.drop(columns=["Dividends", "Stock Splits"])

# Ajustando el formato de las fechas
historical_data["Date"] = pd.to_datetime(
    historical_data["Date"], errors="coerce", utc=True
)
historical_data["Date"] = historical_data["Date"].dt.tz_localize(None)
historical_data["Date"] = historical_data["Date"].dt.strftime("%Y-%m-%d")

# Limpiar Data
historical_data = historical_data.dropna()


# ##### Samsung Stocks KRW to USD

# In[21]:


if ticker == companies[17]:
    won_to_usd_00_17 = pd.read_csv(
        r"C:\Users\inesc\Desktop\neural_catch\Data\Won_Conversion_Data\KRW_TO_USD_2000-2017.csv"
    )
    won_to_usd_04_22 = pd.read_csv(
        r"C:\Users\inesc\Desktop\neural_catch\Data\Won_Conversion_Data\KRW_TO_USD_2004-2022.csv"
    )
    # Preparar Datos de conversión
    won_to_usd_04_22 = won_to_usd_04_22[
        ["Date", "KRW=X"]
    ]  # dejar solamente conversion de wones
    # Borrar filas con datos perdidos
    historical_data = historical_data[historical_data["Volume"] != 0]
    won_to_usd_00_17 = won_to_usd_00_17[won_to_usd_00_17["DEXKOUS"] != "."]
    won_to_usd_04_22 = won_to_usd_04_22[won_to_usd_04_22["KRW=X"] != "."]
    # Renombrar columnas
    won_to_usd_00_17 = won_to_usd_00_17.rename(
        columns={"DEXKOUS": "Value", "DATE": "Date"}
    )
    won_to_usd_04_22 = won_to_usd_04_22.rename(columns={"KRW=X": "Value"})
    # Ajustar formato de fechas
    won_to_usd_00_17["Date"] = pd.to_datetime(won_to_usd_00_17["Date"], errors="coerce")
    won_to_usd_00_17["Date"] = won_to_usd_00_17["Date"].dt.tz_localize(None)
    won_to_usd_00_17["Date"] = won_to_usd_00_17["Date"].dt.strftime("%Y-%m-%d")
    won_to_usd_00_17["Value"] = won_to_usd_00_17["Value"].astype(float, errors="ignore")
    won_to_usd_04_22["Date"] = pd.to_datetime(won_to_usd_04_22["Date"], errors="coerce")
    won_to_usd_04_22["Date"] = won_to_usd_04_22["Date"].dt.tz_localize(None)
    won_to_usd_04_22["Date"] = won_to_usd_04_22["Date"].dt.strftime("%Y-%m-%d")
    won_to_usd_04_22["Value"] = won_to_usd_04_22["Value"].astype(float, errors="ignore")
    # Eliminar filas vacias
    won_to_usd_00_17 = won_to_usd_00_17.dropna()
    won_to_usd_04_22 = won_to_usd_04_22.dropna()
    # Unir los datasets
    won_combined = pd.concat([won_to_usd_00_17, won_to_usd_04_22])
    combined_dollar_values = won_combined.drop_duplicates(subset="Date")
    historical_data = pd.merge(
        historical_data, combined_dollar_values, on="Date", how="left"
    )
    historical_data = historical_data.dropna()
    # Convertir a USD
    historical_data["Close"] = historical_data["Close"] / historical_data["Value"]
    historical_data["Open"] = historical_data["Open"] / historical_data["Value"]
    historical_data["High"] = historical_data["High"] / historical_data["Value"]
    historical_data["Low"] = historical_data["Low"] / historical_data["Value"]
    historical_data["Adj Close"] = (
        historical_data["Adj Close"] / historical_data["Value"]
    )
    # Eliminar columna Value
    historical_data = historical_data.drop(columns=["Value"])


# ### Funciones para Calculo de Indices Básicos

# In[22]:


# Calculo del SMA: Simple Moving Average
historical_data["SMA_5"] = historical_data["Close"].rolling(window=5).mean()
historical_data["SMA_10"] = historical_data["Close"].rolling(window=10).mean()
historical_data["SMA_20"] = historical_data["Close"].rolling(window=20).mean()
historical_data["SMA_50"] = historical_data["Close"].rolling(window=50).mean()
historical_data["SMA_100"] = historical_data["Close"].rolling(window=100).mean()
historical_data["SMA_200"] = historical_data["Close"].rolling(window=200).mean()

# Calculo del RSI: Relative Strength Index
window = 14
delta = historical_data["Close"].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
rs = gain / loss
historical_data["RSI_14"] = 100 - (100 / (1 + rs))

# Calculo del MACD: Moving Average Convergence Divergence
short_window = 12
long_window = 26
signal_window = 9
short_ema = historical_data["Close"].ewm(span=short_window, adjust=False).mean()
long_ema = historical_data["Close"].ewm(span=long_window, adjust=False).mean()
historical_data["MACD"] = short_ema - long_ema
historical_data["Signal_Line"] = (
    historical_data["MACD"].ewm(span=signal_window, adjust=False).mean()
)
historical_data = historical_data.dropna()


# ### New Features

# In[41]:


# Calcular las nuevas features usando .loc
historical_data.loc[:, "EMA_50"] = (
    historical_data["Close"].ewm(span=50, adjust=False).mean()
)
historical_data.loc[:, "Daily_Return"] = historical_data["Close"].pct_change()
historical_data.loc[:, "Log_Return"] = np.log(
    historical_data["Close"] / historical_data["Close"].shift(1)
)
historical_data.loc[:, "Middle_Band"] = (
    historical_data["Close"].rolling(window=20).mean()
)
historical_data.loc[:, "Std"] = historical_data["Close"].rolling(window=20).std()
historical_data.loc[:, "Upper_Band"] = (
    historical_data["Middle_Band"] + 2 * historical_data["Std"]
)
historical_data.loc[:, "Lower_Band"] = (
    historical_data["Middle_Band"] - 2 * historical_data["Std"]
)
# Puedes eliminar la columna temporal "Std"
historical_data = historical_data.drop(columns=["Std"])

# Calcular Delta, Gain, Loss y Rs
historical_data.loc[:, "Delta"] = historical_data["Close"].diff()
historical_data.loc[:, "Gain"] = historical_data["Delta"].apply(
    lambda x: x if x > 0 else 0
)
historical_data.loc[:, "Loss"] = historical_data["Delta"].apply(
    lambda x: -x if x < 0 else 0
)
historical_data.loc[:, "Avg_Gain"] = (
    historical_data["Gain"].rolling(window=14, min_periods=14).mean()
)
historical_data.loc[:, "Avg_Loss"] = (
    historical_data["Loss"].rolling(window=14, min_periods=14).mean()
)
historical_data.loc[:, "Rs"] = historical_data["Avg_Gain"] / historical_data["Avg_Loss"]

# Crear DataFrames independientes (opcional)
Delta_df = historical_data.loc[:, ["Date", "Delta"]].dropna().reset_index(drop=True)
Gain_df = historical_data.loc[:, ["Date", "Gain"]].dropna().reset_index(drop=True)
Loss_df = historical_data.loc[:, ["Date", "Loss"]].dropna().reset_index(drop=True)
Rs_df = historical_data.loc[:, ["Date", "Rs"]].dropna().reset_index(drop=True)

# Revisa el resultado
historical_data = historical_data.dropna()

historical_data


# ## Paso 2: Entrenamiento de Redes Neuronales

# ### Preparación de Datos Históricos de Acciones

# In[24]:


historical_features = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "SMA_5",
    "SMA_10",
    "SMA_20",
    "SMA_50",
    "SMA_100",
    "SMA_200",
    "RSI_14",
    "MACD",
    "Signal_Line",
    "EMA_50",
    "Daily_Return",
    "Log_Return",
    "Middle_Band",
    "Upper_Band",
    "Lower_Band",
]
data_hist = historical_data[historical_features].values

# Normalizar los datos históricos
scaler_hist = MinMaxScaler(feature_range=(0, 1))
data_hist = scaler_hist.fit_transform(data_hist)

# Crear secuencias de tiempo para datos históricos
time_steps = 60
X_hist, y_hist = [], []
for i in range(len(data_hist) - time_steps):
    X_hist.append(data_hist[i : i + time_steps])
    y_hist.append(
        data_hist[i + time_steps, 3]
    )  # Usamos 'Close' como etiqueta (índice 3)
X_hist = np.array(X_hist)
y_hist = np.array(y_hist)


# Dividir los datos en conjuntos de entrenamiento y prueba
train_size_hist = int(len(X_hist) * 0.8)
X_train_hist, X_test_hist = X_hist[:train_size_hist], X_hist[train_size_hist:]
y_train_hist, y_test_hist = y_hist[:train_size_hist], y_hist[train_size_hist:]

# Convertir a tensores de PyTorch
X_train_hist = torch.tensor(X_train_hist, dtype=torch.float32)
y_train_hist = torch.tensor(y_train_hist, dtype=torch.float32)
X_test_hist = torch.tensor(X_test_hist, dtype=torch.float32)
y_test_hist = torch.tensor(y_test_hist, dtype=torch.float32)


# ### Definición de Modelo LSTM

# In[25]:


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(self.dropout(out[:, -1, :]))
        return out


# ### Entrenamiento de Histórico de Acciones

# In[26]:


# Parámetros del modelo
input_size_hist = len(historical_features)
hidden_size = 50
num_layers = 2
output_size = 1
num_epochs = 50
batch_size = 64
learning_rate = 0.001

# Definir y entrenar el modelo LSTM para datos históricos
model_hist = LSTMModel(input_size_hist, hidden_size, num_layers, output_size)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model_hist.parameters(), lr=learning_rate)

train_loader_hist = DataLoader(
    TensorDataset(X_train_hist, y_train_hist), batch_size=batch_size, shuffle=True
)
test_loader_hist = DataLoader(
    TensorDataset(X_test_hist, y_test_hist), batch_size=batch_size, shuffle=False
)


# Entrenamiento del Modelo
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(X_batch)
            y_batch = y_batch.view(-1, 1)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")


train_model(model_hist, train_loader_hist, criterion, optimizer, num_epochs)


# ### Evaluación de Modelo Histórico

# In[27]:


# Evaluación del modelo obtenido:
model_hist.eval()
with torch.no_grad():
    preds = model_hist(X_test_hist).detach().cpu().numpy().flatten()
    targets = y_test_hist.view(-1, 1).detach().cpu().numpy().flatten()
    print("Ejemplo de Predicciones vs. Targets:")
    for p, t in zip(preds[:10], targets[:10]):
        print(f"Predicción: {p:.4f}, Target: {t:.4f}")


# ## Desnormalizar Valores

# ### Función para desnormalizar

# In[28]:


def denormalize_value(norm_val, scaler, feature_index, n_features):
    norm_val = np.array(norm_val).reshape(-1, 1)
    dummy = np.zeros((norm_val.shape[0], n_features))
    dummy[:, feature_index] = norm_val[:, 0]
    inv = scaler.inverse_transform(dummy)
    return inv[:, feature_index]


# In[29]:


n_features_hist = len(historical_features)
baseline_preds_norm = X_test_hist[:, -1, 3].detach().cpu().numpy()
model_hist.eval()
with torch.no_grad():
    model_preds_norm = model_hist(X_test_hist).detach().cpu().numpy().flatten()
    y_test_norm = y_test_hist.detach().cpu().numpy().flatten()
model_preds = denormalize_value(
    model_preds_norm, scaler_hist, feature_index=3, n_features=n_features_hist
)
y_test_actual = denormalize_value(
    y_test_norm, scaler_hist, feature_index=3, n_features=n_features_hist
)
baseline_preds = denormalize_value(
    baseline_preds_norm, scaler_hist, feature_index=3, n_features=n_features_hist
)
mse_model = np.mean((model_preds - y_test_actual) ** 2)
mse_baseline = np.mean((baseline_preds - y_test_actual) ** 2)
print(f"MSE Modelo LSTM (desnormalizado): {mse_model:.4f}")
print(f"MSE Línea Base (desnormalizado): {mse_baseline:.4f}")


# In[30]:


model_hist.eval()
with torch.no_grad():
    y_final_pred_tensor = model_hist(X_test_hist)
    y_final_pred_norm = y_final_pred_tensor.detach().cpu().numpy().flatten()

y_final_pred = denormalize_value(
    y_final_pred_norm, scaler_hist, feature_index=3, n_features=n_features_hist
)


# ### Graficar y Visualizar

# In[31]:


import plotly.graph_objects as go

# 1. Obtener todas las fechas y datos reales del DataFrame completo
all_dates = historical_data["Date"].tolist()
all_close_prices = historical_data["Close"].tolist()

# 2. Extraer las fechas correspondientes al conjunto de prueba.
# Dado que al generar secuencias se pierden los primeros "time_steps" registros,
# el conjunto de prueba empieza en la posición: train_size_hist + time_steps.
test_dates = historical_data["Date"].iloc[train_size_hist + time_steps :].tolist()

# 3. Crear la gráfica en Plotly
fig = go.Figure()

# Trazo 1: Toda la serie real de precios (histórica)
fig.add_trace(
    go.Scatter(
        x=all_dates,
        y=all_close_prices,
        mode="lines",
        name="Datos Reales Históricos",
        line=dict(color="blue"),
    )
)

# Trazo 2: Datos reales del conjunto de prueba (sobre el período de predicción)
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=y_test_actual,
        mode="lines",
        name="Datos Reales Test",
        line=dict(color="green"),
    )
)

# Trazo 3: Predicciones del modelo para el conjunto de prueba
fig.add_trace(
    go.Scatter(
        x=test_dates,
        y=y_final_pred,
        mode="lines",
        name="Predicciones",
        line=dict(color="red"),
    )
)

# Actualizar layout de la gráfica
fig.update_layout(
    title="Datos Reales Históricos y Predicciones a Partir del Período de Test",
    xaxis_title="Fecha",
    yaxis_title="Precio de Cierre (USD)",
    template="plotly_white",
)

fig.show()


# # Guardamos el Modelo antes de subir a repositorio

# In[32]:


import os

directory = r"C:\Users\inesc\Desktop\neural_catch\models"
model_filename = "modelo_lstm_historico.pth"
optimizer_filename = "optimizer_lstm_historico.pth"

model_save_path = os.path.join(directory, model_filename)
optimizer_save_path = os.path.join(directory, optimizer_filename)

torch.save(model_hist.state_dict(), model_save_path)
torch.save(optimizer.state_dict(), optimizer_save_path)

print(f"Modelo guardado en {model_save_path}")
print(f"Estado del optimizador guardado en {optimizer_save_path}")


# ## Simulación de Compra-Venta

# In[33]:


df_pred = pd.DataFrame(
    {
        "Date": test_dates,  # lista de fechas del conjunto de prueba
        "Predicted_Close": y_final_pred,  # precios predichos (desnormalizados)
    }
)

df_pred["Real_Close"] = y_test_actual


# In[34]:


import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

covid_start = "2020-03-11"
covid_end = "2021-12-31"

df_pred["Date"] = pd.to_datetime(df_pred["Date"])

df_sim = df_pred[
    (df_pred["Date"] >= covid_start) & (df_pred["Date"] <= covid_end)
].copy()


df_sim["Date_str"] = df_sim["Date"].dt.strftime("%Y-%m-%d")

# =============================================================
# 2. Generar señales de trading basadas en la predicción
# =============================================================
# Usamos la estrategia simple: si el precio predicho sube respecto al día anterior, señal "Buy";
# si baja, señal "Sell"; y para el primer día (o sin cambio), "Hold".

n = len(df_sim)
signals = []
for i in range(n):
    if i == 0 or i == n - 1:
        signals.append("Hold")
    else:
        prev_price = df_sim.iloc[i - 1]["Predicted_Close"]
        curr_price = df_sim.iloc[i]["Predicted_Close"]
        next_price = df_sim.iloc[i + 1]["Predicted_Close"]
        if curr_price < prev_price and curr_price < next_price:
            signals.append("Buy")
        elif curr_price > prev_price and curr_price > next_price:
            signals.append("Sell")
        else:
            signals.append("Hold")
df_sim["Decision"] = signals

# =============================================================
# 3. Simulación de trading usando precios predichos para decidir y precios reales para operar
# =============================================================
initial_capital = 10000.0
capital = initial_capital
shares = 0.0
equity_history = []  # Almacena el equity diario
executed_signals = []  # Guarda la acción ejecutada cada día ("Buy", "Sell" o "Hold")

# Se recorre el DataFrame de simulación (cada fila representa un día del test dentro del período seleccionado)
for idx, row in df_sim.iterrows():
    # Para tomar la decisión, usamos la predicción, pero para la operación usamos el precio real.
    predicted_price = row["Predicted_Close"]  # Guía para la señal
    real_price = row["Real_Close"]  # Precio con el que se opera
    decision = row["Decision"]

    if decision == "Buy" and shares == 0:
        # Compra con todo el capital al precio real
        shares = capital / real_price
        capital = 0.0
        executed_signals.append("Buy")
    elif decision == "Sell" and shares > 0:
        # Vende toda la posición
        capital = shares * real_price
        shares = 0.0
        executed_signals.append("Sell")
    else:
        executed_signals.append("Hold")

    current_equity = capital + shares * real_price
    equity_history.append(current_equity)

df_sim["Equity"] = equity_history
df_sim["Executed_Signal"] = executed_signals

final_gain_pct = (equity_history[-1] / initial_capital - 1) * 100
print(f"Ganancia Hipotética: {final_gain_pct:.2f}%")

# =============================================================
# 4. Graficar la simulación
# =============================================================
# Extraer las fechas en formato string para el eje X
test_dates_str = df_sim["Date_str"].tolist()

# Extraer los precios reales (para mostrar la serie), las predicciones y el equity
real_prices = df_sim["Real_Close"].tolist()
predicted_prices = df_sim["Predicted_Close"].tolist()
equity_vals = df_sim["Equity"].tolist()

# Para marcar las señales, extraemos las fechas y precios donde se ejecutó cada acción:
buy_dates = df_sim[df_sim["Executed_Signal"] == "Buy"]["Date_str"].tolist()
buy_prices = df_sim[df_sim["Executed_Signal"] == "Buy"]["Predicted_Close"].tolist()

sell_dates = df_sim[df_sim["Executed_Signal"] == "Sell"]["Date_str"].tolist()
sell_prices = df_sim[df_sim["Executed_Signal"] == "Sell"]["Predicted_Close"].tolist()

hold_dates = df_sim[df_sim["Executed_Signal"] == "Hold"]["Date_str"].tolist()
hold_prices = df_sim[df_sim["Executed_Signal"] == "Hold"]["Predicted_Close"].tolist()

# Crear la gráfica con dos subplots: una para la serie de precios y señales, y otra para la evolución del equity.
fig = make_subplots(
    rows=2,
    cols=1,
    shared_xaxes=True,
    subplot_titles=[
        "Precio de Cierre (Guía de Predicción) y Señales",
        "Evolución del Equity (Ganancia Hipotética)",
    ],
)

# Subplot 1: Precios
fig.add_trace(
    go.Scatter(
        x=test_dates_str,
        y=real_prices,
        mode="lines",
        name="Precio Real",
        line=dict(color="blue"),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=test_dates_str,
        y=predicted_prices,
        mode="lines+markers",
        name="Predicción (Guía)",
        line=dict(color="red"),
    ),
    row=1,
    col=1,
)

# Señales: Marcar puntos de compra, venta y aguante usando los precios predichos
fig.add_trace(
    go.Scatter(
        x=buy_dates,
        y=buy_prices,
        mode="markers",
        name="Comprar",
        marker=dict(color="red", symbol="triangle-up", size=10),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=sell_dates,
        y=sell_prices,
        mode="markers",
        name="Vender",
        marker=dict(color="green", symbol="triangle-down", size=10),
    ),
    row=1,
    col=1,
)

fig.add_trace(
    go.Scatter(
        x=hold_dates,
        y=hold_prices,
        mode="markers",
        name="Aguantar",
        marker=dict(color="darkgrey", symbol="circle", size=7),
    ),
    row=1,
    col=1,
)

# Subplot 2: Evolución del equity
fig.add_trace(
    go.Scatter(
        x=test_dates_str,
        y=equity_vals,
        mode="lines",
        name="Equity (Ganancia)",
        line=dict(color="purple"),
    ),
    row=2,
    col=1,
)

fig.update_layout(
    height=800,
    title="Simulación de Trading utilizando Precio Predicho como Guía\nOperando con Precios Reales (Período COVID)",
    xaxis_title="Fecha",
    yaxis_title="Precio y Equity (USD)",
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
)

fig.update_yaxes(title_text="Precio (USD)", row=1, col=1)
fig.update_yaxes(title_text="Equity (USD)", row=2, col=1)

fig.show()

