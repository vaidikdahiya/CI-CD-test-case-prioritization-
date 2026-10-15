"""
Tiny local server for the frontend — avoids the file:// CORS/fetch issue.

Run with:
    python serve.py

Then open http://localhost:5500 in your browser (NOT by double-clicking index.html).
Make sure the backend is also running in another terminal:
    cd ../backend
    uvicorn app:app --reload --port 8000
"""
import http.server
import socketserver

PORT = 5500

Handler = http.server.SimpleHTTPRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print(f"Serving frontend at http://localhost:{PORT}")
    print("Make sure the backend is running at http://localhost:8000 (uvicorn app:app --reload --port 8000)")
    httpd.serve_forever()
