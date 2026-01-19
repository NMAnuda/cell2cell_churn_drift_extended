import uvicorn

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)