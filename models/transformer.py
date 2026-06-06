"""Softmax transformer entrypoint.

Run:
    uv run python -m models.transformer
"""

from models.attention import Attention

__all__ = ["Attention"]


if __name__ == "__main__":
    from models.attention import main

    main()
