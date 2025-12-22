#!/usr/bin/env python3
"""
Minimal test server to verify Java sends correct prev_transition fields.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json

class Handler(BaseHTTPRequestHandler):
    def do_POST(self):
        if self.path != "/infer_and_update":
            self.send_response(404)
            self.end_headers()
            return
        
        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            req = json.loads(raw.decode("utf-8"))
        except Exception as e:
            print(f"[ERROR] Failed to parse request: {e}")
            self.send_response(400)
            self.end_headers()
            return
        
        # Log prev_transition data
        prev_tr = req.get("prev_transition", [])
        print(f"\n{'='*60}")
        print(f"[PREV_TRANSITION] Received {len(prev_tr)} transitions")
        
        for i, item in enumerate(prev_tr):
            print(f"\nTransition {i+1}:")
            print(f"  host: {item.get('host')}")
            print(f"  dest: {item.get('dest')}")
            print(f"  delivered: {item.get('delivered', 0)}")
            print(f"  relayed: {item.get('relayed', 0)}")
            print(f"  drops: {item.get('drops', 0)}")
            print(f"  ⭐ p_base: {item.get('p_base', 'MISSING!')}")
            print(f"  ⭐ action: {item.get('action', 'MISSING!')}")
            print(f"  ⭐ my_buffer_norm: {item.get('my_buffer_norm', 'MISSING!')}")
            print(f"  neighbor_p_base: {item.get('neighbor_p_base', 0.0)}")
        
        print(f"{'='*60}\n")
        
        # Send dummy response
        resp = {"policy_id": "test_v1", "actions": [], "actions_kv": {}}
        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    print("Starting test server at http://127.0.0.1:5000")
    print("Waiting for Java to send prev_transition data...\n")
    srv = HTTPServer(("127.0.0.1", 5000), Handler)
    srv.serve_forever()
