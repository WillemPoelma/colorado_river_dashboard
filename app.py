import dash
from dash import dcc, html
from dashboard_UC_WSB import layout  # if you defined layout in another file

app = dash.Dash(__name__)
server = app.server  # <-- critical for Vercel

# If you don’t import layout, define it here:
app.layout = html.Div("Hello Colorado River")

if __name__ == "__main__":
    app.run_server(debug=True)
