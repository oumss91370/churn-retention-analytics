"""Scénario critique CRIT-07 : le dashboard ne charge jamais le `.pkl`."""

from __future__ import annotations

import ast
from pathlib import Path


def test_crit07_dashboard_does_not_import_joblib() -> None:
    """Le dashboard doit appeler l'API en HTTP, jamais lire le modèle directement."""
    source = Path("app/dashboard.py").read_text(encoding="utf-8")
    tree = ast.parse(source)

    forbidden = {"joblib", "pickle"}
    forbidden_from = {"src.models", "api.predict"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden, f"Le dashboard importe {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            assert module not in forbidden_from, f"Le dashboard importe {module}"
            assert module not in forbidden, f"Le dashboard importe {module}"


def test_crit07_dashboard_uses_requests() -> None:
    """Le dashboard doit utiliser `requests` (ou httpx) pour parler à l'API."""
    source = Path("app/dashboard.py").read_text(encoding="utf-8")
    assert "import requests" in source or "from requests" in source
