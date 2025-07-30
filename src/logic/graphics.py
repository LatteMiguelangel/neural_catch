import flet as ft
import pandas as pd
from neuronb import (
    load_Balance_Sheet,
    load_Cash_Flow,
    load_Income_Statement,
    load_Financial_Ratios,
    companies,
)

def graphics(page: ft.Page):
    page.title = "Gráficas Financieras"
    page.theme_mode = ft.ThemeMode.DARK
    page.padding = 10  # Reducir padding general

    # Dropdown para seleccionar empresa
    company_dropdown = ft.Dropdown(
        options=[ft.dropdown.Option(company) for company in companies],
        value=companies[0],
        width=300,
        on_change=lambda e: update_charts(e.control.value),
    )

    # Contenedores para las gráficas (tamaño fijo más pequeño)
    chart_container_height = 500  # Altura fija para todos los contenedores
    chart_container_width = 600   # Ancho fijo para balance general (más ancho)

    balance_chart = ft.BarChart(expand=True)
    income_chart = ft.LineChart(expand=True)
    cash_flow_chart = ft.BarChart(expand=True)
    ratios_chart = ft.LineChart(expand=True)

    def update_charts(selected_company):
        # Cargar datos
        balance_sheet = load_Balance_Sheet(selected_company)
        income_statement = load_Income_Statement(selected_company)
        cash_flow = load_Cash_Flow(selected_company)
        financial_ratios = load_Financial_Ratios(selected_company)

        # Actualizar gráficas
        update_bar_chart(balance_chart, balance_sheet, "Balance General")
        update_line_chart(income_chart, income_statement, "Estado de Resultados")
        update_bar_chart(cash_flow_chart, cash_flow, "Flujo de Efectivo")
        update_line_chart(ratios_chart, financial_ratios, "Ratios Financieros")

        page.update()

    def update_bar_chart(chart, data, title):
        chart.bar_groups.clear()
        if not data.empty:
            numeric_cols = data.select_dtypes(include='number').columns[:5]
            for idx, col in enumerate(numeric_cols):
                chart.bar_groups.append(
                    ft.BarChartGroup(
                        x=idx,
                        bar_rods=[
                            ft.BarChartRod(
                                from_y=0,
                                to_y=float(data[col].iloc[-1]),
                                width=15,  # Barras más delgadas
                                color=ft.Colors.AMBER,
                                tooltip=f"{col}: {data[col].iloc[-1]:.2f}",
                            )
                        ],
                    )
                )
        chart.title = ft.Text(title, size=14, weight="bold")
        chart.bar_gap = 20  # Espacio entre grupos de barras

    def update_line_chart(chart, data, title):
        chart.data_series.clear()
        if not data.empty:
            numeric_cols = data.select_dtypes(include='number').columns[:3]
            colors = [ft.Colors.BLUE, ft.Colors.GREEN, ft.Colors.RED]
            for col, color in zip(numeric_cols, colors):
                chart.data_series.append(
                    ft.LineChartData(
                        data_points=[
                            ft.LineChartDataPoint(i, float(val))
                            for i, val in enumerate(data[col].tail(8))  # Menos puntos
                        ],
                        stroke_width=2.5,  # Líneas más delgadas
                        color=color,
                        curved=False,     # Líneas rectas para más claridad
                    )
                )
        chart.title = ft.Text(title, size=14, weight="bold")

    # Diseño de la interfaz con contenedores ajustados
    page.add(
        ft.Column(
            [
                ft.Row([company_dropdown], alignment=ft.MainAxisAlignment.CENTER),
                ft.Tabs(
                    selected_index=0,
                    tabs=[
                        ft.Tab(
                            text="Balance General",
                            content=ft.Container(
                                balance_chart,
                                height=chart_container_height,
                                width=chart_container_width,
                                padding=10,
                                border_radius=10,
                                bgcolor=ft.Colors.GREY_100,
                            ),
                        ),
                        ft.Tab(
                            text="Estado de Resultados",
                            content=ft.Container(
                                income_chart,
                                height=chart_container_height,
                                width=600,  # Más estrecho
                                padding=10,
                                border_radius=10,
                                bgcolor=ft.Colors.GREY_100,
                            ),
                        ),
                        ft.Tab(
                            text="Flujo de Efectivo",
                            content=ft.Container(
                                cash_flow_chart,
                                height=chart_container_height,
                                width=600,
                                padding=10,
                                border_radius=10,
                                bgcolor=ft.Colors.GREY_100,
                            ),
                        ),
                        ft.Tab(
                            text="Ratios Financieros",
                            content=ft.Container(
                                ratios_chart,
                                height=chart_container_height,
                                width=600,
                                padding=10,
                                border_radius=10,
                                bgcolor=ft.Colors.GREY_100,
                            ),
                        ),
                    ],
                    expand=True,
                ),
            ],
            expand=True,
        )
    )

    # Inicializar con la primera empresa
    update_charts(companies[0])

ft.app(target=graphics)