"""
startup.py — NeuralGuard Single Launcher
Run: python startup.py
To stop: python startup.py --stop
"""
import subprocess, sys, time, os, webbrowser, platform, signal, urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON   = sys.executable
PID_FILE = os.path.join(BASE_DIR, 'logs', 'portal.pid')
URL      = "http://127.0.0.1:5000"

def stop_existing():
    if os.path.exists(PID_FILE):
        try:
            pid = int(open(PID_FILE).read().strip())
            if platform.system() == 'Windows':
                subprocess.call(['taskkill', '/F', '/PID', str(pid)],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            else:
                os.kill(pid, signal.SIGTERM)
            print(f"    Stopped existing portal (PID {pid})")
        except Exception:
            pass
        try:
            os.remove(PID_FILE)
        except Exception:
            pass

def is_portal_up():
    try:
        urllib.request.urlopen(URL + '/api/alert_state', timeout=2)
        return True
    except Exception:
        return False

def launch():
    os.makedirs(os.path.join(BASE_DIR, 'logs'), exist_ok=True)
    os.chdir(BASE_DIR)

    if '--stop' in sys.argv:
        stop_existing()
        print("Portal stopped.")
        return

    print("=" * 55)
    print("   NeuralGuard Smart CCTV — Starting...")
    print("=" * 55)
    print(f"   Python : {PYTHON}")
    print(f"   Folder : {BASE_DIR}")
    print()

    stop_existing()

    log_path = os.path.join(BASE_DIR, 'logs', 'portal.log')
    print("[1/2] Starting Admin Portal...")

    try:
        if platform.system() == 'Windows':
            DETACHED_PROCESS      = 0x00000008
            CREATE_NEW_PROC_GROUP = 0x00000200
            portal_proc = subprocess.Popen(
                [PYTHON, 'portal.py'],
                cwd=BASE_DIR,
                stdout=open(log_path, 'a'),
                stderr=subprocess.STDOUT,
                creationflags=DETACHED_PROCESS | CREATE_NEW_PROC_GROUP,
                close_fds=True,
            )
        else:
            portal_proc = subprocess.Popen(
                [PYTHON, 'portal.py'],
                cwd=BASE_DIR,
                stdout=open(log_path, 'a'),
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        open(PID_FILE, 'w').write(str(portal_proc.pid))
        print(f"    ✅ Portal started (PID {portal_proc.pid})")
        print(f"    📄 Logs → logs/portal.log")
    except Exception as e:
        print(f"    ❌ Failed to start portal: {e}")
        input("Press Enter to exit...")
        return

    # ── Wait for portal to actually respond (up to 15s) ───────
    print("[2/2] Waiting for portal to be ready", end='', flush=True)
    ready = False
    for _ in range(15):
        time.sleep(1)
        print('.', end='', flush=True)
        if is_portal_up():
            ready = True
            break
    print()

    if not ready:
        print("    ⚠  Portal is taking longer than expected.")
        print(f"    Check logs/portal.log for errors.")
        print(f"    Try opening {URL} manually in a moment.")
    else:
        print(f"    ✅ Portal is ready!")
        try:
            webbrowser.open(URL)
            print(f"    ✅ Browser opened: {URL}")
        except Exception:
            print(f"    ⚠  Open manually: {URL}")

    print()
    print("=" * 55)
    print("   NeuralGuard Portal is running!")
    print(f"   URL  : {URL}")
    print()
    print("   ✅ Safe to close this window — portal keeps running")
    print("   🛑 To stop:  python startup.py --stop")
    print("   📄 Logs:     logs/portal.log")
    print("=" * 55)

if __name__ == '__main__':
    launch()