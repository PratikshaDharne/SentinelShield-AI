import os, sys, tempfile
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp()}/test.db"
os.environ["OWNER_PASSWORD"] = "test-pass"
os.environ["JWT_SECRET"] = "test-secret"
