"""Root-level server entry point for openenv validate compatibility."""

from invoice_agent.server.app import app  # noqa: F401


def main() -> None:
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
