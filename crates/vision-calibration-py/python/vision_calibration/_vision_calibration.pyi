"""The compiled extension. Its functions exchange serde payloads and are
wrapped by the typed modules of the package; this stub leaves them untyped."""

from typing import Any

def __getattr__(name: str) -> Any: ...
