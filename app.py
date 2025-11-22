import dash
from dash import dcc, html
import DashBoard_UC_WSB   # must match filename exactly

app = dash.Dash(__name__)
server = app.server  # critical for Vercel

# Use the layout defined in DashBoard_UC_WSB.py
app.layout = DashBoard_UC_WSB.layout

if __name__ == "__main__":
    app.run_server(debug=True)
