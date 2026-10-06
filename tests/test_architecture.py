"""
Import direction: a package may import only from its own layer or below, and the package graph has no cycles.

Layers are ranked from the lowest (foundation) up; a package may import any package of equal or lower
rank. The ranks encode today's real dependency order, so a new upward import fails here instead of
growing into a cycle.
"""
import ast
from collections import defaultdict
from pathlib import Path

SRC = Path(__file__).resolve().parent.parent / "src"

LAYERS = [
    {"config", "errors", "retry", "tokens", "protected_markdown"},
    {"db", "observability"},
    {"stores", "maintenance"},
    {"chunking", "processing", "parsing", "llm", "crawling"},
    {"embedding", "config_checks"},
    {"retrieving", "ingestion", "runtime"},
    {"generating"},
    {"evaluation"},
    {"testsets"},
    {"services"},
    {"jobs"},
    {"api"},
]
RANK = {package: rank for rank, layer in enumerate(LAYERS) for package in layer}

# The API <-> worker contract is deliberately importable by the services (see its docstring).
SERVICE_VISIBLE_JOB_MODULES = {"src.jobs.contract", "src.jobs.queue"}


def package_of(path: Path) -> str:
    parts = path.relative_to(SRC).with_suffix("").parts
    return parts[0]


def imports_of(path: Path):
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            yield node.module
        elif isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name


def package_edges():
    edges = defaultdict(set)
    violations = []
    for path in sorted(SRC.rglob("*.py")):
        source = package_of(path)
        for module in imports_of(path):
            if not module.startswith("src."):
                continue
            target = module.split(".")[1]
            if target == source:
                continue
            if source == "services" and module in SERVICE_VISIBLE_JOB_MODULES:
                continue
            edges[source].add(target)
            if RANK[source] < RANK[target]:
                violations.append(f"{path.relative_to(SRC.parent)} imports {module}")
    return edges, violations


def test_every_package_has_a_layer():
    packages = {package_of(path) for path in SRC.rglob("*.py")} - {"__init__"}
    assert packages <= set(RANK), f"packages without a layer: {sorted(packages - set(RANK))}"


def test_imports_point_down_only():
    _, violations = package_edges()
    assert not violations, "upward imports:\n" + "\n".join(violations)


def test_package_graph_has_no_cycles():
    edges, _ = package_edges()
    visiting, done = [], set()

    def visit(package):
        if package in done:
            return
        assert package not in visiting, "import cycle: " + " -> ".join(visiting[visiting.index(package):] + [package])
        visiting.append(package)
        for target in sorted(edges[package]):
            visit(target)
        visiting.pop()
        done.add(package)

    for package in sorted(edges):
        visit(package)
