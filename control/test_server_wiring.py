"""Catch CLI/constructor drift in robot_server.

A string-replacement edit once added `compliance` to the CLI and to the
constructor *body* but not to its signature, so the server died at startup with
`unexpected keyword argument`. Nothing caught it: the module imported fine and
--help worked. This checks the three things that must agree.

    python -m control.test_server_wiring
"""

import ast
import inspect
from pathlib import Path

SRC = Path(__file__).resolve().parent / "robot_server.py"


def kwargs_passed_to(tree, func_name: str) -> set[str]:
    """Keyword names in every `func_name(...)` call in the module."""
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            fn = node.func
            name = getattr(fn, "id", None) or getattr(fn, "attr", None)
            if name == func_name:
                found |= {kw.arg for kw in node.keywords if kw.arg}
    return found


def names_used_in_init(tree) -> tuple[set[str], set[str]]:
    """(names read, names bound) inside RobotServer.__init__.

    Bound covers assignments, loop targets, comprehensions and imports made
    inside the function -- all legitimate, and all of which a naive "is this
    a parameter?" check would otherwise flag.
    """
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "RobotServer":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "__init__":
                    read, bound = set(), set()
                    for n in ast.walk(item):
                        if isinstance(n, ast.Name):
                            (read if isinstance(n.ctx, ast.Load) else bound).add(n.id)
                        elif isinstance(n, (ast.Import, ast.ImportFrom)):
                            bound |= {a.asname or a.name.split(".")[0] for a in n.names}
                        elif isinstance(n, ast.ExceptHandler) and n.name:
                            bound.add(n.name)
                    return read, bound
    return set(), set()


def main():
    import control.robot_server as rs

    tree = ast.parse(SRC.read_text())
    sig = set(inspect.signature(rs.RobotServer.__init__).parameters) - {"self"}
    ok = True

    print("=== every kwarg main() passes exists in the signature ===")
    passed = kwargs_passed_to(tree, "RobotServer")
    missing = sorted(passed - sig)
    print(f"  {'PASS' if not missing else 'FAIL'}  passes {len(passed)}"
          + (f", NOT in signature: {missing}" if missing else ", all accepted"))
    ok &= not missing

    print("\n=== every name __init__ uses is a parameter or an attribute ===")
    read, bound = names_used_in_init(tree)
    # A forgotten parameter is a name that is READ but never bound locally,
    # is not a parameter, not a module global, and not a builtin.
    import builtins
    globals_ = {n for n in dir(rs) if not n.startswith("__")}
    suspects = sorted(n for n in read
                      if n not in sig and n not in bound and n not in globals_
                      and not hasattr(builtins, n) and n != "self")
    print(f"  {'PASS' if not suspects else 'FAIL'}  "
          + (f"unresolved: {suspects}" if suspects else "no unresolved names"))
    ok &= not suspects

    print("\n=== every impedance profile file exists ===")
    root = SRC.parent.parent
    for key, rel in rs.RobotServer.IMPEDANCE_CONFIGS.items():
        exists = (root / rel).exists()
        print(f"  {'PASS' if exists else 'FAIL'}  {key[0]}/{key[1]} -> {rel}")
        ok &= exists

    print("\n" + ("SERVER WIRING OK" if ok else "FAILURES ABOVE"))
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
