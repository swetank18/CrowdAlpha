"""
engine/agents/user_strategy_sandbox.py

Static safety checks for user-submitted strategy source code.

This is not a full security proof. It is a hardening layer used before
running strategy code in a subprocess with strict request timeouts.
"""

from __future__ import annotations

import ast


MAX_CODE_SIZE = 32_000

_BANNED_NAMES = {
    "open",
    "exec",
    "eval",
    "compile",
    "__import__",
    "globals",
    "locals",
    "vars",
    "input",
    "help",
    "dir",
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "ctypes",
}


class StrategySafetyError(ValueError):
    pass


class _SafetyVisitor(ast.NodeVisitor):
    def __init__(self) -> None:
        self._saw_base_agent_class = False

    def visit_Import(self, node: ast.Import) -> None:
        raise StrategySafetyError("Imports are not allowed in user strategies.")

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        raise StrategySafetyError("Imports are not allowed in user strategies.")

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if node.attr.startswith("__") and node.attr != "__init__":
            raise StrategySafetyError("Dunder attribute access is not allowed.")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        if isinstance(node.func, ast.Name) and node.func.id in _BANNED_NAMES:
            raise StrategySafetyError(f"Call to banned name '{node.func.id}' is not allowed.")
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if isinstance(node.ctx, ast.Load) and node.id in _BANNED_NAMES:
            raise StrategySafetyError(f"Use of banned name '{node.id}' is not allowed.")
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        for base in node.bases:
            if isinstance(base, ast.Name) and base.id == "BaseAgent":
                self._saw_base_agent_class = True
        self.generic_visit(node)

    @property
    def saw_base_agent_class(self) -> bool:
        return self._saw_base_agent_class


def validate_user_strategy_source(code: str) -> ast.Module:
    if not isinstance(code, str) or not code.strip():
        raise StrategySafetyError("Strategy code must be a non-empty string.")
    if len(code) > MAX_CODE_SIZE:
        raise StrategySafetyError(f"Strategy code exceeds max size ({MAX_CODE_SIZE} chars).")

    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        raise StrategySafetyError(f"Syntax error: {exc}") from exc

    visitor = _SafetyVisitor()
    visitor.visit(tree)
    if not visitor.saw_base_agent_class:
        raise StrategySafetyError("Strategy must define a class inheriting from BaseAgent.")
    return tree
