import dash
from dash import dcc, html, Input, Output
import plotly.express as px
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Carregar e preparar dados
df = pd.read_csv('df_cat.csv')
variaveis_selecionadas = ['skate_stance', 'ollie_foot', 'bowl_foot', 'hand_write', 
                         'hand_throw', 'foot_kick', 'foot_pedal', 'eye_test1', 'eye_test2']
df = df[variaveis_selecionadas]

# Configurar app Dash
app = dash.Dash(__name__)

app.layout = html.Div([
    html.H1("Análise de Lateralidade - Painel Interativo", style={'textAlign': 'center', 'color': '#2E86AB'}),
    html.Div([
        dcc.Dropdown(
            id='variavel-dropdown',
            options=[{'label': col, 'value': col} for col in variaveis_selecionadas],
            value='skate_stance',
            style={'width': '50%', 'margin': '20px auto'}
        ),
        dcc.RadioItems(
            id='plot-type',
            options=[
                {'label': 'Histograma', 'value': 'hist'},
                {'label': 'Boxplot', 'value': 'box'},
                {'label': 'Violino', 'value': 'violin'}
            ],
            value='hist',
            inline=True,
            style={'width': '50%', 'margin': '0 auto'}
        ),
        dcc.Graph(id='distribution-plot')
    ]),
    html.Div([
        dcc.Graph(
            id='correlation-heatmap',
            figure=px.imshow(
                df.corr(),
                text_auto=True,
                color_continuous_scale='RdBu_r',
                aspect="auto",
                labels=dict(color="Correlação")
			),
            config={'displayModeBar': False}
        )
    ], style={'margin-top': '30px'})
])

@app.callback(
    Output('distribution-plot', 'figure'),
    [Input('variavel-dropdown', 'value'),
     Input('plot-type', 'value')]
)
def update_plot(selected_var, plot_type):
    if plot_type == 'hist':
        fig = px.histogram(df, x=selected_var, nbins=20, 
                          color_discrete_sequence=['#2E86AB'],
                          title=f"Distribuição de {selected_var}")
    elif plot_type == 'box':
        fig = px.box(df, y=selected_var, 
                    color_discrete_sequence=['#2E86AB'],
                    title=f"Boxplot de {selected_var}")
    else:
        fig = px.violin(df, y=selected_var, 
                       box=True, points="all",
                       color_discrete_sequence=['#2E86AB'],
                       title=f"Distribuição de {selected_var} (Violino)")
    
    fig.update_layout(title_font_size=16)
    return fig

if __name__ == '__main__':
    app.run(debug=True, port=8051)
