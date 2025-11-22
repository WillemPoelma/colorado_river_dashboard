from DashBoard_UC_WSB import app

# Expose the server for Vercel
server = app.server

if __name__ == "__main__":
    app.run_server(debug=True)
