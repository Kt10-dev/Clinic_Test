import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from main import app
import uvicorn

def main():
    uvicorn.run("main:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()
