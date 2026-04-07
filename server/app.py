"""
server/app.py - Server module for OpenEnv deployment.
This contains the actual FastAPI app and main() entry point.
"""
import uvicorn
from server import app


def main():
    """Run the uvicorn server."""
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=7860,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    main()