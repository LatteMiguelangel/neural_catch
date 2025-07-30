import flet as ft
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
import traceback
from logic.graphics import financial_charts

# ==================== MODELO LSTM ====================
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
    
# ==================== APP PRINCIPAL ====================
def main(page: ft.Page):
    # === Configuración del Tema ===
    page.title = "Neural Catch"
    page.theme_mode = ft.ThemeMode.DARK
    page.bgcolor = "#0D1117"
    page.padding = 0
    page.scroll = ft.ScrollMode.AUTO
    page.theme = ft.Theme(
        color_scheme_seed=ft.Colors.CYAN_500,
        font_family="Segoe UI",
        visual_density=ft.VisualDensity.COMFORTABLE,
    )

    # === Variables Globales ===
    model_hist = None
    scaler_hist = None
    historical_features = None
    data_folder = None
    historical_data = None
    X_train_hist = None
    y_train_hist = None
    X_test_hist = None
    y_test_hist = None
    train_size_hist = None
    time_steps = None
    mse_model = None
    last_real_price = None
    last_pred_price = None

    companies = [
        "AAPL", "ADBE", "AMZN", "CSCO", "DELL", "GOOGL", 
        "IBM", "INTC", "META", "MSFT", "NOK", "NTDOY",
        "NVDA", "NFLX", "ORCL", "QCOM", "SONY", "SSNLF", "TSLA"
    ]

    # === Componentes de la UI ===
    company_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(c) for c in companies],
        label="Empresa",
        width=200,
        bgcolor="#161B22",
        color=ft.Colors.WHITE,
        border_color=ft.Colors.CYAN_400,
    )

    data_folder_textfield = ft.TextField(
        label="Ruta carpeta datos",
        value=r"C:\Users\inesc\Desktop\neural_catch\Data",
        width=380,
        bgcolor="#161B22",
        color=ft.Colors.WHITE,
        border_color=ft.Colors.CYAN_400,
    )

    status_text = ft.Text(
        "Seleccione una empresa y cargue los datos",
        size=16,
        weight=ft.FontWeight.W_600,
        color=ft.Colors.CYAN_300,
    )

    progress_bar = ft.ProgressBar(width=400, visible=False, color=ft.Colors.CYAN_500)

    load_data_button = ft.ElevatedButton(
        "Cargar Datos",
        icon=ft.Icons.DOWNLOAD,
        style=ft.ButtonStyle(bgcolor="#238636", color=ft.Colors.WHITE),
        on_click=lambda e: load_data_click(e),
    )

    train_model_button = ft.ElevatedButton(
        "Entrenar Modelo",
        icon=ft.Icons.SCIENCE,
        style=ft.ButtonStyle(bgcolor="#8256D0", color=ft.Colors.WHITE),
        disabled=True,
        on_click=lambda e: train_model_click(e),
    )

    predict_button = ft.ElevatedButton(
        "Predecir",
        icon=ft.Icons.SHOW_CHART,
        style=ft.ButtonStyle(bgcolor="#1F6FEB", color=ft.Colors.WHITE),
        disabled=True,
        on_click=lambda e: predict_click(e),
    )

    # Tarjetas métricas
    def metric_card(title):
        return ft.Container(
            content=ft.Column(
                [
                    ft.Text(title, size=14, color=ft.Colors.CYAN_200),
                    ft.Text("-", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.WHITE),
                ]
            ),
            padding=15,
            bgcolor="#161B22",
            border_radius=12,
            shadow=ft.BoxShadow(blur_radius=10, color=ft.Colors.with_opacity(0.4, ft.Colors.BLACK)),
            expand=True,
        )

    metric_card1 = metric_card("MSE Modelo")
    metric_card2 = metric_card("Último precio real (USD)")
    metric_card3 = metric_card("Última predicción (USD)")

    # Gráfica principal
    chart = ft.LineChart(
        expand=True,
        tooltip_bgcolor=ft.Colors.with_opacity(0.9, ft.Colors.BLACK),
        interactive=True,
    )

    plot_container = ft.Container(
        content=chart,
        border=ft.border.all(1, ft.Colors.with_opacity(0.3, ft.Colors.CYAN_200)),
        border_radius=12,
        padding=20,
        expand=True,
        bgcolor="#161B22",
    )

    # === Funciones principales ===
    def update_status(message):
        status_text.value = message
        page.update()

    def load_data_click(e):
        nonlocal data_folder, historical_data, X_train_hist, y_train_hist, X_test_hist, y_test_hist, train_size_hist, time_steps, historical_features, scaler_hist
        data_folder = data_folder_textfield.value
        ticker = company_dropdown.value

        if not data_folder or not ticker:
            update_status("⚠️ Error: Debe seleccionar empresa y ruta.")
            return

        try:
            update_status(f"Cargando datos para {ticker}...")
            progress_bar.visible = True
            page.update()

            historical_data = pd.read_csv(f"{data_folder}/Historical_Data/{ticker}_historical_data.csv")
            historical_data["Date"] = pd.to_datetime(historical_data["Date"], errors="coerce", utc=True)
            historical_data["Date"] = historical_data["Date"].dt.tz_localize(None)
            historical_data = historical_data.dropna()

            historical_features = ["Open", "High", "Low", "Close", "Volume"]
            data_hist = historical_data[historical_features].values

            scaler_hist = MinMaxScaler(feature_range=(0, 1))
            data_hist = scaler_hist.fit_transform(data_hist)

            time_steps = 60
            X_hist, y_hist = [], []
            for i in range(len(data_hist) - time_steps):
                X_hist.append(data_hist[i : i + time_steps])
                y_hist.append(data_hist[i + time_steps, 3])

            X_hist = np.array(X_hist)
            y_hist = np.array(y_hist)

            train_size_hist = int(len(X_hist) * 0.8)
            X_train_hist, X_test_hist = X_hist[:train_size_hist], X_hist[train_size_hist:]
            y_train_hist, y_test_hist = y_hist[:train_size_hist], y_hist[train_size_hist:]

            X_train_hist = torch.tensor(X_train_hist, dtype=torch.float32)
            y_train_hist = torch.tensor(y_train_hist, dtype=torch.float32)
            X_test_hist = torch.tensor(X_test_hist, dtype=torch.float32)
            y_test_hist = torch.tensor(y_test_hist, dtype=torch.float32)

            train_model_button.disabled = False
            progress_bar.visible = False
            update_status("✅ Datos cargados. Listo para entrenar.")

        except Exception as ex:
            progress_bar.visible = False
            update_status(f"Error al cargar datos: {str(ex)}")
            print(traceback.format_exc())

    def train_model_click(e):
        nonlocal model_hist
        try:
            update_status("⏳ Entrenando modelo...")
            progress_bar.visible = True
            page.update()

            model_hist = LSTMModel(len(historical_features), 50, 2, 1)
            criterion = nn.MSELoss()
            optimizer = torch.optim.Adam(model_hist.parameters(), lr=0.001)

            loader = DataLoader(TensorDataset(X_train_hist, y_train_hist), batch_size=64, shuffle=True)

            for epoch in range(30):
                for X_batch, y_batch in loader:
                    optimizer.zero_grad()
                    y_pred = model_hist(X_batch)
                    loss = criterion(y_pred, y_batch.view(-1, 1))
                    loss.backward()
                    optimizer.step()

            predict_button.disabled = False
            progress_bar.visible = False
            update_status("✅ Modelo entrenado.")

        except Exception as ex:
            progress_bar.visible = False
            update_status(f"Error al entrenar: {str(ex)}")

    def predict_click(e):
        nonlocal mse_model, last_real_price, last_pred_price
        try:
            update_status("🔮 Generando predicciones...")
            progress_bar.visible = True
            page.update()

            historical_data["Date"] = pd.to_datetime(historical_data["Date"], errors="coerce")

            model_hist.eval()
            with torch.no_grad():
                preds = model_hist(X_test_hist).numpy().flatten()
                targets = y_test_hist.numpy().flatten()

            def denorm(values):
                dummy = np.zeros((len(values), len(historical_features)))
                dummy[:, 3] = values
                return scaler_hist.inverse_transform(dummy)[:, 3]

            pred_vals = denorm(preds)
            real_vals = denorm(targets)

            mse_model = np.mean((pred_vals - real_vals) ** 2)
            last_real_price = real_vals[-1]
            last_pred_price = pred_vals[-1]

            # Actualizar métricas
            metric_card1.content.controls[1].value = f"{mse_model:.4f}"
            metric_card2.content.controls[1].value = f"{last_real_price:.2f}"
            metric_card3.content.controls[1].value = f"{last_pred_price:.2f}"

            chart.data_series.clear()
            all_dates = historical_data["Date"].iloc[train_size_hist + time_steps :].dt.strftime("%Y-%m-%d").tolist()
            
            chart.data_series.append(
                ft.LineChartData(
                    data_points=[
                        ft.LineChartDataPoint(
                            x=i, y=real_vals[i],
                            tooltip=f"{all_dates[i]}\nReal: {real_vals[i]:.2f}"
                        ) for i in range(len(real_vals))
                    ],
                    color=ft.Colors.CYAN_300,
                    stroke_width=2,
                    curved=True,
                )
            )
            
            chart.data_series.append(
                ft.LineChartData(
                    data_points=[
                        ft.LineChartDataPoint(
                            x=i, y=pred_vals[i],
                            tooltip=f"{all_dates[i]}\nPredicción: {pred_vals[i]:.2f}"
                        ) for i in range(len(pred_vals))
                    ],
                    color=ft.Colors.PINK_300,
                    stroke_width=3,
                    curved=True,
                )
            )

            chart.min_x = 0
            chart.max_x = len(real_vals)
            chart.min_y = min(min(real_vals), min(pred_vals)) * 0.95
            chart.max_y = max(max(real_vals), max(pred_vals)) * 1.05

            progress_bar.visible = False
            update_status("✅ Predicciones generadas.")

        except Exception as ex:
            progress_bar.visible = False
            update_status(f"Error al predecir: {str(ex)}")
            print(traceback.format_exc())

    # === Manejo de Navegación ===
    def route_change(route):
        if page.route == "/financial-charts":
            page.views.clear()
            page.update()
            financial_charts(page)
            return
        # Vista del Dashboard Principal
        page.views.clear()
        page.views.append(
            ft.View(
                "/",
                [
                    ft.Row(
                        [
                            # Sidebar
                            ft.Container(
                                bgcolor="#161B22",
                                padding=20,
                                width=230,
                                content=ft.Column(
                                    [
                                        ft.Image(
                                            src="https://images-wixmp-ed30a86b8c4ca887773594c2.wixmp.com/f/3044f73c-0547-4b01-aeec-7ebff6555e1b/dk8oj7q-b6b00fe4-92fb-4618-878b-240d312a4f4f.png",
                                            width=150,
                                            height=150,
                                        ),
                                        ft.Text("Neural Catch", size=22, weight=ft.FontWeight.BOLD, color=ft.Colors.CYAN_300),
                                        ft.Divider(color=ft.Colors.CYAN_700),
                                        ft.ListTile(
                                            leading=ft.Icon(ft.Icons.ANALYTICS, color=ft.Colors.CYAN_300),
                                            title=ft.Text("Dashboard", color=ft.Colors.WHITE),
                                            on_click=lambda _: page.go("/"),
                                        ),
                                        ft.ListTile(
                                            leading=ft.Icon(ft.Icons.BAR_CHART, color=ft.Colors.CYAN_300),
                                            title=ft.Text("Análisis Financiero", color=ft.Colors.WHITE),
                                            on_click=lambda _: page.go("/financial-charts"),
                                        ),
                                        ft.ListTile(
                                            leading=ft.Icon(ft.Icons.SETTINGS, color=ft.Colors.CYAN_300),
                                            title=ft.Text("Configuración", color=ft.Colors.WHITE),
                                        ),
                                    ],
                                    spacing=15,
                                    expand=True,
                                ),
                            ),
                            # Contenido Principal
                            ft.Column(
                                [
                                    # Barra superior
                                    ft.Container(
                                        bgcolor="#0D1117",
                                        padding=15,
                                        border=ft.border.only(bottom=ft.BorderSide(1, ft.Colors.CYAN_800)),
                                        content=ft.Row(
                                            [
                                                company_dropdown,
                                                data_folder_textfield,
                                                load_data_button,
                                                train_model_button,
                                                predict_button,
                                            ],
                                            spacing=12,
                                        ),
                                    ),
                                    # Contenido
                                    ft.Column(
                                        [
                                            ft.Container(content=status_text, padding=10),
                                            ft.Row([metric_card1, metric_card2, metric_card3], spacing=15),
                                            plot_container,
                                            ft.Container(progress_bar, padding=10),
                                        ],
                                        spacing=20,
                                        expand=True,
                                    ),
                                ],
                                expand=True,
                            ),
                        ],
                        expand=True,
                    )
                ],
            )
        )
        page.update()

    def view_pop(view):
        page.views.pop()
        top_view = page.views[-1]
        page.go(top_view.route)

    page.on_route_change = route_change
    page.on_view_pop = view_pop

    # === Inicialización ===
    page.go(page.route)

if __name__ == "__main__":
    ft.app(target=main)