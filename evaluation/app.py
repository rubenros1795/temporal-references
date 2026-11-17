"""
Streamlit Cloud entry point. The app was deployed from evaluation/app.py before the folder moved to
curation/evaluation/; this file only runs the real app. To remove it: point the deployment at
curation/evaluation/app.py (Streamlit Cloud: delete and redeploy with that main file, same secrets), then delete
this file.
"""
import runpy
from pathlib import Path

runpy.run_path(str(Path(__file__).resolve().parents[1] / "curation" / "evaluation" / "app.py"), run_name="__main__")
