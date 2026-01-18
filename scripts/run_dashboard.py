

import subprocess
import sys
import os

if __name__ == "__main__":
    # FIXED: Run via Streamlit CLI (no import error)
    dashboard_path = os.path.join('src', 'dashboard', 'dashboard.py')
    subprocess.call([
        sys.executable, '-m', 'streamlit', 'run', dashboard_path,
        '--server.port', '8501',
        '--server.address', '0.0.0.0'
    ])