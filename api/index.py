import os
import sys
from streamlit.web.cli import main

def handler(event, context):
    # This points to your app.py in the main folder
    sys.argv = [
        "streamlit",
        "run",
        os.path.join(os.path.dirname(__file__), "..", "app.py"),
        "--server.port",
        "8080",
        "--server.address",
        "0.0.0.0",
    ]
    main()