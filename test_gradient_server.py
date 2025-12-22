#!/usr/bin/env python3
"""
Test server to verify Gradient Alignment Reward implementation.
Logs all prev_transition data to verify neighbor_p_base is working.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import json
import sys

class GradientTestHandler(BaseHTTPRequestHandler):
    request_count = 0

    def log_message(self, format, *args):
        # Suppress default logging
        pass

    def do_POST(self):
        if self.path == "/episode_end":
            try:
                length = int(self.headers.get("Content-Length", "0"))
                raw = self.rfile.read(length)
                req = json.loads(raw.decode("utf-8")) if length > 0 else {}

                print("\n" + "="*70)
                print("📊 EPISODE END")
                print("="*70)
                print(f"Sim ID: {req.get('sim_id', 'N/A')}")
                print(f"Delivered: {req.get('delivered', 0)}")
                print(f"Created: {req.get('created', 0)}")
                print(f"Avg Delivery Rate: {req.get('average_delivery_rate', 0.0):.2%}")
                print("="*70 + "\n")

                body = json.dumps({"ok": True}).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            except Exception as e:
                print(f"[ERROR] Episode end failed: {e}")
                self.send_response(500)
                self.end_headers()
                return

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

        GradientTestHandler.request_count += 1

        # Extract data
        prev_tr = req.get("prev_transition", [])
        state_batch = req.get("state_batch", [])
        sim_time = req.get("time", 0)

        # Log prev_transition data (focusing on Gradient Alignment fields)
        if prev_tr:
            print(f"\n{'='*70}")
            print(f"⏰ Time: {sim_time}s | Request #{GradientTestHandler.request_count}")
            print(f"📦 Prev Transitions: {len(prev_tr)} | States: {len(state_batch)}")
            print(f"{'='*70}")

            for i, item in enumerate(prev_tr[:5]):  # Show first 5
                host = item.get("host", "?")
                dest = item.get("dest", "?")
                delivered = item.get("delivered", 0)
                relayed = item.get("relayed", 0)
                drops = item.get("drops", 0)

                # Gradient Alignment fields
                p_base = item.get("p_base", "MISSING")
                neighbor_p_base = item.get("neighbor_p_base", "MISSING")
                my_buffer_norm = item.get("my_buffer_norm", "MISSING")
                action = item.get("action", "MISSING")

                print(f"\n  [{i+1}] {host} → {dest}")
                print(f"      Events: delivered={delivered}, relayed={relayed}, drops={drops}")
                print(f"      🔥 p_base: {p_base}")
                print(f"      🔥 neighbor_p_base: {neighbor_p_base}")
                print(f"      🔥 my_buffer_norm: {my_buffer_norm}")
                print(f"      🔥 action: {action}")

                # Calculate Gradient Alignment reward (demo)
                if p_base != "MISSING" and neighbor_p_base != "MISSING" and action != "MISSING":
                    try:
                        p_b = float(p_base)
                        n_p_b = float(neighbor_p_base)
                        act = float(action)
                        buf_norm = float(my_buffer_norm) if my_buffer_norm != "MISSING" else 0.5

                        # R_grad
                        diff = p_b - n_p_b
                        r_grad = act * diff

                        # R_align
                        r_align = -(buf_norm - 0.5) * act

                        # R_arrival
                        r_arrival = 10.0 if delivered > 0 else 0.0

                        # Total (using default weights)
                        total_reward = 1.0 * r_grad + 0.5 * r_align + r_arrival

                        print(f"      💰 Gradient Reward Preview:")
                        print(f"         R_grad = {r_grad:.4f} (action × ({p_b:.3f} - {n_p_b:.3f}))")
                        print(f"         R_align = {r_align:.4f}")
                        print(f"         R_arrival = {r_arrival:.1f}")
                        print(f"         Total ≈ {total_reward:.4f}")
                    except Exception as e:
                        print(f"      ⚠️ Could not calculate reward: {e}")

            if len(prev_tr) > 5:
                print(f"\n  ... and {len(prev_tr) - 5} more transitions")

        # Build dummy response
        actions = []
        for item in state_batch[:20]:  # Limit to 20 states
            host = item.get("host", "")
            dest = item.get("dest", "")
            if host and dest:
                # Simple heuristic action
                actions.append({
                    "host": host,
                    "per_message": [{"dest": dest, "delta": 0.0}]
                })

        resp = {
            "policy_id": "gradient_test_v1",
            "actions": actions,
            "actions_kv": {}
        }

        body = json.dumps(resp).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

if __name__ == "__main__":
    HOST = "127.0.0.1"
    PORT = 5000

    print("="*70)
    print("🧪 Gradient Alignment Reward Test Server")
    print("="*70)
    print(f"Listening on http://{HOST}:{PORT}")
    print("Waiting for Java simulator to connect...")
    print("="*70 + "\n")

    try:
        srv = HTTPServer((HOST, PORT), GradientTestHandler)
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\n\n✅ Server stopped by user")
        sys.exit(0)
