import dash
from dash import dcc, html, Input, Output
from Dataloader_UC_WSB import get_loader

app = dash.Dash(__name__)
server = app.server

# --- Dashboard Layout ---
app.layout = html.Div([
    html.H1("Hydrologic and Drought Analysis in the Upper Colorado River Basin (1909–2013)", style={
        'textAlign': 'center',
        'fontSize': '30px',
        'fontFamily': 'Arial, sans-serif',
        'marginBottom': '8px',
        'marginTop': '10px'
    }),

    html.Div([
        html.Div([
            html.Label("District:", style={
                'fontSize': '14px',
                'fontFamily': 'Arial, sans-serif',
                'marginBottom': '3px'
            }),
            dcc.Dropdown(
                id='district-dropdown',
                options=[
                    {'label': 'Gunnison', 'value': 'gunnison'},
                    {'label': 'White', 'value': 'white'},
                    {'label': 'Yampa', 'value': 'yampa'},
                    {'label': 'Upper Colorado', 'value': 'uppercolorado'},
                    {'label': 'San Juan & Dolores', 'value': 'sanjuan'}
                ],
                value='gunnison',
                clearable=False,
                style={
                    'width': '160px',
                    'fontSize': '13px',
                    'fontFamily': 'Arial, sans-serif',
                    'marginBottom': '8px'
                }
            ),
            html.Label("Year:", style={
                'fontSize': '14px',
                'fontFamily': 'Arial, sans-serif',
                'marginBottom': '3px'
            }),
            dcc.Dropdown(
                id='year-dropdown',
                options=[{'label': str(y), 'value': y} for y in range(1909, 2014)],
                value=2002,
                clearable=False,
                style={
                    'width': '120px',
                    'fontSize': '13px',
                    'fontFamily': 'Arial, sans-serif'
                }
            )
        ], style={
            'backgroundColor': '#f9f9f9',
            'padding': '10px',
            'borderRadius': '6px',
            'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
        })
    ], style={
        'position': 'absolute',
        'top': '20px',
        'right': '20px',
        'display': 'flex',
        'flexDirection': 'column',
        'alignItems': 'flex-end',
        'zIndex': '1000'
    }),

    html.Div([
        html.Div([
            html.H3("Streamflow", style={'fontFamily': 'Arial, sans-serif', 'marginBottom': '0px'}),
            dcc.Graph(id='streamflow-plot', config={'displayModeBar': False}, style={'height': '255px'})
        ], style={'flex': '1', 'paddingRight': '15px'}),

        html.Div([
            html.H3("Reservoir Storage", style={'fontFamily': 'Arial, sans-serif', 'marginBottom': '0px'}),
            dcc.Graph(id='reservoir-plot', config={'displayModeBar': False}, style={'height': '255px'})
        ], style={'flex': '1', 'paddingLeft': '10px'}),
    ], style={'display': 'flex', 'width': '100%', 'marginBottom': '2px'}),

    html.Div([
        html.Div([
            html.H3("Water Use Sankey", style={'fontFamily': 'Arial, sans-serif', 'marginBottom': '0px'}),
            dcc.Graph(id='sankey-plot', config={'displayModeBar': False}, style={'height': '255px'})
        ], style={'flex': '1', 'paddingRight': '10px'}),

        html.Div([
            html.H3("Sectoral Shortage Fraction", style={'fontFamily': 'Arial, sans-serif', 'marginBottom': '0px'}),
            dcc.Graph(id='shortage-plot', config={'displayModeBar': False}, style={'height': '255px'})
        ], style={'flex': '1', 'paddingLeft': '10px'}),
    ], style={'display': 'flex', 'width': '100%', 'marginBottom': '2px'}),

    html.Div([
        html.H3("District-Level Drought Map", style={'fontFamily': 'Arial, sans-serif', 'marginBottom': '0px'}),
        dcc.Graph(
            id='drought-map',
            config={
                'displayModeBar': True,
                'modeBarButtonsToRemove': [
                    'zoom2d', 'select2d', 'lasso2d', 'autoScale2d', 'resetScale2d',
                    'hoverClosestCartesian', 'hoverCompareCartesian', 'toggleSpikelines',
                    'toImage'
                ]
            },
            style={'height': '255px'}
        )
    ], style={'width': '100%', 'marginTop': '0px', 'marginBottom': '0px'})
])

# --- Callbacks ---
@app.callback(
    Output('streamflow-plot', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_streamflow(district, year):
    fig = get_loader(district)['plot_streamflow'](year)
    fig.update_layout(height=255, font=dict(family="Arial", size=12), autosize=True)
    return fig

@app.callback(
    Output('sankey-plot', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_sankey(district, year):
    fig = get_loader(district)['make_sankey_fig'](year)
    fig.update_layout(height=255, font=dict(family="Arial", size=12))
    return fig

@app.callback(
    Output('shortage-plot', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_shortage(district, year):
    fig = get_loader(district)['sector_water_use_shortage_plot'](year)
    fig.update_layout(height=255, font=dict(family="Arial", size=12))
    return fig

@app.callback(
    Output('drought-map', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_drought_map(district, year):
    fig = get_loader(district)['plot_drought_map'](year)
    fig.update_layout(height=255, font=dict(family="Arial", size=12))
    return fig

@app.callback(
    Output('reservoir-plot', 'figure'),
    [Input('district-dropdown', 'value'),
     Input('year-dropdown', 'value')]
)
def update_reservoir(district, year):
    fig = get_loader(district)['plot_storage'](year)
    fig.update_layout(height=255, font=dict(family="Arial", size=12))
    return fig

if __name__ == '__main__':
    app.run(debug=True)