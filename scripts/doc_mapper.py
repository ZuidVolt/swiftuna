#!/usr/bin/env python3
"""
scripts/doc_mapper.py

Extracts docstrings from rustuna Python type stubs (_rustuna.pyi), maps them
to equivalent Swiftuna Swift symbols, and generates idiomatic Swift DocC comments.
"""

import argparse
import ast
import re
import sys
from pathlib import Path
from typing import NamedTuple, Optional

REPO_ROOT = Path(__file__).resolve().parent.parent
STUB_PATH = REPO_ROOT / "ref/rustuna/rustuna_pyo3/rustuna/_rustuna.pyi"


class SymbolDoc(NamedTuple):
    parent: Optional[str]
    name: str
    signature: str
    docstring: str


def parse_python_stubs(stub_file: Path) -> dict[str, SymbolDoc]:
    """Parse _rustuna.pyi using ast and return a dict of qualified symbol names to SymbolDoc."""
    with open(stub_file, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(stub_file))

    docs: dict[str, SymbolDoc] = {}

    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            class_name = node.name
            class_doc = ast.get_docstring(node) or ""
            docs[class_name] = SymbolDoc(
                parent=None,
                name=class_name,
                signature=f"class {class_name}",
                docstring=class_doc,
            )
            for item in node.body:
                if isinstance(item, ast.FunctionDef):
                    func_doc = ast.get_docstring(item) or ""
                    key = f"{class_name}.{item.name}"
                    args = [arg.arg for arg in item.args.args if arg.arg != "self"]
                    docs[key] = SymbolDoc(
                        parent=class_name,
                        name=item.name,
                        signature=f"{item.name}({', '.join(args)})",
                        docstring=func_doc,
                    )
        elif isinstance(node, ast.FunctionDef):
            func_doc = ast.get_docstring(node) or ""
            args = [arg.arg for arg in node.args.args]
            docs[node.name] = SymbolDoc(
                parent=None,
                name=node.name,
                signature=f"{node.name}({', '.join(args)})",
                docstring=func_doc,
            )

    return docs


def python_to_swift_docc(py_doc: str, symbol_name: str) -> str:
    """Convert Python Google-style docstring into Swift DocC markdown format."""
    if not py_doc.strip():
        return "/// No documentation available in Python stubs."

    lines = py_doc.split("\n")
    swift_lines: list[str] = []

    in_args = False
    in_returns = False
    in_raises = False
    in_example = False
    in_code_block = False

    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # Handle code fences
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                swift_lines.append("/// ```swift")
            else:
                in_code_block = False
                swift_lines.append("/// ```")
            i += 1
            continue

        if in_code_block:
            # Basic Python-to-Swift syntactic replacements in code examples
            sw_line = line
            sw_line = re.sub(
                r"\bdef objective\(trial: [^\)]+\) -> float:",
                "let objective = { (trial: inout Trial) throws -> Double in",
                sw_line,
            )
            sw_line = re.sub(
                r"trial\.suggest_float\(([^,]+),\s*([^,]+),\s*([^)]+)\)",
                r"try trial.suggest(\1, in: \2...\3)",
                sw_line,
            )
            sw_line = re.sub(
                r"trial\.suggest_int\(([^,]+),\s*([^,]+),\s*([^)]+)\)",
                r"try trial.suggest(\1, in: \2...\3)",
                sw_line,
            )
            sw_line = re.sub(
                r"trial\.suggest_categorical\(([^,]+),\s*([^)]+)\)",
                r"try trial.suggest(\1, choices: \2)",
                sw_line,
            )
            sw_line = re.sub(
                r"rustuna\.create_study\(\)", "try Swiftuna.createStudy()", sw_line
            )
            sw_line = re.sub(
                r"study\.optimize\(([^,]+),\s*n_trials=(\d+)\)",
                r"try study.optimize(nTrials: \2, objective: \1)",
                sw_line,
            )
            sw_line = sw_line.replace("import rustuna", "import Swiftuna")
            sw_line = (
                sw_line.replace("True", "true")
                .replace("False", "false")
                .replace("None", "nil")
            )
            swift_lines.append(f"/// {sw_line}")
            i += 1
            continue

        # Section headers
        if stripped.startswith("Args:"):
            in_args = True
            in_returns = False
            in_raises = False
            in_example = False
            swift_lines.append("///")
            swift_lines.append("/// - Parameters:")
            i += 1
            continue

        if stripped.startswith("Returns:"):
            in_args = False
            in_returns = True
            in_raises = False
            in_example = False
            swift_lines.append("///")
            swift_lines.append("/// - Returns:")
            i += 1
            continue

        if stripped.startswith("Raises:"):
            in_args = False
            in_returns = False
            in_raises = True
            in_example = False
            swift_lines.append("///")
            swift_lines.append("/// - Throws:")
            i += 1
            continue

        if stripped.startswith("Example:") or stripped.startswith("Examples:"):
            in_args = False
            in_returns = False
            in_raises = False
            in_example = True
            swift_lines.append("///")
            swift_lines.append("/// ### Example")
            i += 1
            continue

        if stripped.startswith("Note:") or stripped.startswith("Notes:"):
            swift_lines.append("///")
            swift_lines.append("/// > Note:")
            i += 1
            continue

        # Transform section content
        if in_args:
            m = re.match(r"^(\s+)([a-zA-Z0-9_]+)(\s*\([^)]*\))?:\s*(.*)$", line)
            if m:
                indent, param, _, desc = m.groups()
                param = {
                    "low": "range",
                    "high": "range",
                    "n_trials": "nTrials",
                    "study_name": "name",
                    "load_if_exists": "loadIfExists",
                }.get(param, param)
                swift_lines.append(f"///   - {param}: {desc}")
            elif stripped:
                swift_lines.append(f"///     {stripped}")
            else:
                swift_lines.append("///")
            i += 1
            continue

        if in_returns:
            if stripped:
                swift_lines.append(f"///   {stripped}")
            else:
                swift_lines.append("///")
            i += 1
            continue

        if in_raises:
            if stripped:
                err_clean = re.sub(r"\bRuntimeError\b", "SwiftunaError", stripped)
                err_clean = re.sub(
                    r"\bValueError\b", "SwiftunaError.invalidArgument", err_clean
                )
                swift_lines.append(f"///   {err_clean}")
            else:
                swift_lines.append("///")
            i += 1
            continue

        line_clean = line
        line_clean = line_clean.replace("``None``", "`nil`")
        line_clean = line_clean.replace("`None`", "`nil`")
        line_clean = line_clean.replace("`True`", "`true`")
        line_clean = line_clean.replace("`False`", "`false`")
        line_clean = line_clean.replace("``float``", "`Double`")
        line_clean = line_clean.replace("`float`", "`Double`")
        line_clean = line_clean.replace("``int``", "`Int`")
        line_clean = line_clean.replace("`int`", "`Int`")
        line_clean = line_clean.replace("dict", "Dictionary")
        line_clean = line_clean.replace("tuple", "Tuple")
        line_clean = line_clean.replace("Rustuna", "Swiftuna")
        line_clean = re.sub(
            r"\[Study\.optimize\]\[[^\]]+\]", "``Study/optimize``", line_clean
        )
        line_clean = re.sub(r"\[Study\.ask\]\[[^\]]+\]", "``Study/ask()``", line_clean)
        line_clean = re.sub(
            r"\[Trial\.set_constraints\]\[[^\]]+\]",
            "``Trial/setConstraints(_:)``",
            line_clean,
        )

        swift_lines.append(f"/// {line_clean}".rstrip())
        i += 1

    return "\n".join(swift_lines)


# Pilot Golden Path mappings
GOLDEN_PATH_SYMBOLS = [
    ("Trial.suggest_float", "Trial.suggest(in: ClosedRange<Double>)"),
    ("Trial.suggest_int", "Trial.suggest(in: ClosedRange<Int>)"),
    ("Trial.suggest_categorical", "Trial.suggest(choices:)"),
    ("Trial.set_constraint", "Trial.setConstraint(_:value:)"),
    ("Trial.set_constraints", "Trial.setConstraints(_:)"),
    ("Study.ask", "Study.ask()"),
    ("Study.tell", "Study.tell(consuming:value:state:)"),
    ("Study.optimize", "Study.optimize(nTrials:timeout:objective:)"),
    ("Study.best_trial", "Study.bestTrial"),
    ("Study.best_value", "Study.bestValue"),
    ("Study.best_trials", "Study.bestTrials"),
    ("Study.enqueue_trial", "Study.enqueue(_:userAttrs:)"),
    ("Study.add_trial", "Study.addTrial(_:)"),
    ("create_study", "Swiftuna.createStudy(...)"),
    ("load_study", "Swiftuna.loadStudy(...)"),
]


def main():
    parser = argparse.ArgumentParser(
        description="Swiftuna DocC Documentation Assistant"
    )
    parser.add_argument(
        "--list", action="store_true", help="List all Python stub symbols"
    )
    parser.add_argument(
        "--audit", action="store_true", help="Audit parity for Golden Path symbols"
    )
    parser.add_argument(
        "--symbol",
        type=str,
        help="Generate Swift DocC block for symbol (e.g. Trial.suggest_float)",
    )
    parser.add_argument(
        "--golden-path",
        action="store_true",
        help="Print all translated Golden Path doc comments",
    )
    args = parser.parse_args()

    if not STUB_PATH.exists():
        print(f"Error: Stub file not found at {STUB_PATH}", file=sys.stderr)
        sys.exit(1)

    py_docs = parse_python_stubs(STUB_PATH)

    if args.list:
        print(f"Found {len(py_docs)} documented symbols in {STUB_PATH.name}:")
        for sym, data in sorted(py_docs.items()):
            has_doc = "✅" if data.docstring else "❌"
            print(f"  {has_doc} {sym:35} -> {data.signature}")
        return

    if args.audit:
        print("=========================================================")
        print("          Swiftuna Docstring Parity Audit               ")
        print("=========================================================")
        found = 0
        for py_sym, sw_sym in GOLDEN_PATH_SYMBOLS:
            doc = py_docs.get(py_sym)
            if doc and doc.docstring:
                print(f"  ✅ {py_sym:26} ➔  {sw_sym}")
                found += 1
            else:
                print(f"  ⚠️  {py_sym:26} ➔  {sw_sym} (No Python docstring)")
        pct = (found / len(GOLDEN_PATH_SYMBOLS)) * 100
        print("---------------------------------------------------------")
        print(f"Golden Path Coverage: {found}/{len(GOLDEN_PATH_SYMBOLS)} ({pct:.1f}%)")
        print("=========================================================")
        return

    if args.symbol:
        doc = py_docs.get(args.symbol)
        if not doc:
            print(f"Symbol '{args.symbol}' not found in Python stubs.", file=sys.stderr)
            sys.exit(1)
        print(f"// --- Generated DocC for {args.symbol} ---")
        print(python_to_swift_docc(doc.docstring, args.symbol))
        return

    if args.golden_path:
        for py_sym, sw_sym in GOLDEN_PATH_SYMBOLS:
            doc = py_docs.get(py_sym)
            if doc and doc.docstring:
                print(f"\n// ========================================================")
                print(f"// Swift Symbol: {sw_sym} (from {py_sym})")
                print(f"// ========================================================")
                print(python_to_swift_docc(doc.docstring, py_sym))
        return

    parser.print_help()


if __name__ == "__main__":
    main()
