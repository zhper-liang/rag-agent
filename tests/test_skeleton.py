import importlib

import pytest

PACKAGES = [
    "ingest", "ingest.chunking", "index", "retrieval", "preprocess", "context",
    "prompt", "llm", "postprocess", "graph", "graph.nodes", "api", "monitor",
]


@pytest.mark.parametrize("name", PACKAGES)
def test_package_importable(name):
    importlib.import_module(f"ragagent.{name}")
