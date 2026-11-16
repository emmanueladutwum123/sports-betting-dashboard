"""Streamlit entry point. No startup API requests or module-cache surgery."""
from dotenv import load_dotenv

load_dotenv()

from src.ui.dashboard import main  # noqa: E402

main()
