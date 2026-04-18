import sys
from pathlib import Path

# Add src to python path so we can run from root
src_path = str(Path(__file__).parent / "src")
if src_path not in sys.path:
    sys.path.append(src_path)

from rag_manager import main

if __name__ == "__main__":
    main()
