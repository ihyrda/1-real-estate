"""Console logging configuration for the real-estate project.

Logs go to the console only. They appear in the VS Code terminal, the local
Streamlit server terminal, and the Streamlit Community Cloud log view, so no
logs/ folder is required.
"""

import logging


def configure_logging() -> None:
    """Configure project console logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
