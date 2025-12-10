import http.server
import socketserver
import os
import sys

PORT = 3000
DIRECTORY = "."

class QuietHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress logging of successful requests to reduce noise
        pass

    def handle(self):
        # Handle BrokenPipeError (client disconnected) gracefully
        try:
            super().handle()
        except BrokenPipeError:
            pass
        except ConnectionResetError:
            pass

if __name__ == "__main__":
    # Change to the directory where the script is located (frontend)
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    
    Handler = QuietHandler
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        print(f"Frontend serving at http://localhost:{PORT}")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nStopping frontend server...")
            httpd.shutdown()
