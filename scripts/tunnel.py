import os
import sys
from pyngrok import ngrok, conf

# Configuration
BACKEND_TARGET = "http://127.0.0.1:8000"
FRONTEND_TARGET = "http://127.0.0.1:5173"

def start_tunnels():
    # Attempt to read from .env file manually
    auth_token = os.environ.get("NGROK_AUTHTOKEN")
    if not auth_token and os.path.exists(".env"):
        with open(".env", "r") as f:
            for line in f:
                if line.startswith("NGROK_AUTHTOKEN="):
                    auth_token = line.split("=")[1].strip()
                    break

    if auth_token:
        ngrok.set_auth_token(auth_token)
    else:
        print("!" * 50, flush=True)
        print("Warning: NGROK_AUTHTOKEN not found in environment.", flush=True)
        print("Ngrok requires an authtoken for multiple simultaneous tunnels.", flush=True)
        print("You can set it via: export NGROK_AUTHTOKEN=your_token", flush=True)
        print("!" * 50, flush=True)

    print("\n--- Starting NextGenDB Neural Tunnels ---", flush=True)
    
    try:
        # Backend Tunnel (API) - Force random domain
        backend_tunnel = ngrok.connect(BACKEND_TARGET, bind_tls=True)
        print(f"Backend API:  {backend_tunnel.public_url}  ->  {BACKEND_TARGET}", flush=True)
        
        # Frontend Tunnel (UI) - Force random domain
        frontend_tunnel = ngrok.connect(FRONTEND_TARGET, bind_tls=True)
        print(f"Frontend UI:   {frontend_tunnel.public_url}   ->  {FRONTEND_TARGET}", flush=True)
        
        print("\n[NextGenDB] Tunnels are active. Keep this process running to maintain access.", flush=True)
        print("Press Ctrl+C to terminate.\n", flush=True)
        
        # Block until process is terminated
        ngrok_process = ngrok.get_ngrok_process()
        ngrok_process.proc.wait()
        
    except Exception as e:
        print(f"\n❌ Error starting tunnels: {e}")
        if "session" in str(e).lower():
            print("Tip: This usually happens if you try to start multiple tunnels without a registered authtoken.")
        sys.exit(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down tunnels...")
        ngrok.kill()

if __name__ == "__main__":
    start_tunnels()
