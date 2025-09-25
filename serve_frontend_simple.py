#!/usr/bin/env python3
"""
Simple HTTP server for the frontend with PDF proxy
"""
import http.server
import socketserver
import os
import webbrowser
from pathlib import Path


# Change to frontend directory
frontend_dir = Path(__file__).parent / "frontend"
os.chdir(frontend_dir)

PORT = 8080


class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Cache-Control', 'no-cache, no-store, must-revalidate')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

    def do_GET(self):
        # Handle regular frontend requests
        if self.path == '/':
            self.path = '/index.html'
        return super().do_GET()


if __name__ == "__main__":
    try:
        with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
            print(f"✅ Frontend server starting on http://localhost:{PORT}")
            print(f"📁 Serving from: {frontend_dir.absolute()}")
            print(f"🌐 Open your browser to: http://localhost:{PORT}")
            print("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        print("\n🛑 Server stopped")
    except Exception as e:
        print(f"❌ Error starting server: {e}")