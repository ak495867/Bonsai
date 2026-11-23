from __future__ import annotations

import ast
from typing import Any, Mapping
import numpy as np
import pandas as pd

from .engine import _feature_frame
from .models import MarketPanel


def ts_mean(df: pd.DataFrame, window: int) -> pd.DataFrame:
    w = max(1, int(window))
    return df.rolling(w, min_periods=max(1, w // 2)).mean()


def ts_std(df: pd.DataFrame, window: int) -> pd.DataFrame:
    w = max(2, int(window))
    return df.rolling(w, min_periods=max(2, w // 2)).std().fillna(0.0)


def ts_delta(df: pd.DataFrame, window: int) -> pd.DataFrame:
    w = max(1, int(window))
    return df.diff(w).fillna(0.0)


def ts_decay_linear(df: pd.DataFrame, window: int) -> pd.DataFrame:
    w = max(1, int(window))
    if w == 1:
        return df.copy()
    weights = np.arange(1, w + 1, dtype=float)
    weights /= weights.sum()

    def _conv1d(col: pd.Series) -> pd.Series:
        vals = col.to_numpy(dtype=float)
        res = np.full_like(vals, np.nan)
        for i in range(len(vals)):
            start = max(0, i - w + 1)
            sub = vals[start : i + 1]
            sub_w = weights[-len(sub) :]
            sub_w = sub_w / sub_w.sum()
            res[i] = np.dot(sub, sub_w)
        return pd.Series(res, index=col.index)

    return df.apply(_conv1d)


def ts_rank(df: pd.DataFrame, window: int) -> pd.DataFrame:
    w = max(2, int(window))
    mat = df.to_numpy(dtype=float)
    n_rows, n_cols = mat.shape
    res = np.full_like(mat, np.nan)

    for i in range(n_rows):
        start = max(0, i - w + 1)
        sub = mat[start : i + 1]
        cur = mat[i]
        valid_counts = np.sum(np.isfinite(sub), axis=0)
        less_counts = np.sum(sub < cur[None, :], axis=0)
        rank_pct = np.where(valid_counts > 1, less_counts / np.maximum(valid_counts - 1, 1), 0.5)
        res[i] = rank_pct

    return pd.DataFrame(res, index=df.index, columns=df.columns)


def ts_corr(x: pd.DataFrame, y: pd.DataFrame, window: int) -> pd.DataFrame:
    w = max(3, int(window))
    res = x.rolling(w, min_periods=max(3, w // 2)).corr(y)
    return res.replace([np.inf, -np.inf], np.nan).fillna(0.0)


def cs_rank(df: pd.DataFrame) -> pd.DataFrame:
    return df.rank(axis=1, pct=True)


def cs_zscore(df: pd.DataFrame) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0.0, np.nan)
    return df.sub(mean, axis=0).div(std, axis=0).fillna(0.0)


def scale(df: pd.DataFrame, target: float = 1.0) -> pd.DataFrame:
    abs_sum = df.abs().sum(axis=1).replace(0.0, np.nan)
    return df.div(abs_sum, axis=0).fillna(0.0) * float(target)


def signed_power(df: pd.DataFrame | float, p: float) -> pd.DataFrame | float:
    if isinstance(df, pd.DataFrame):
        return np.sign(df) * (df.abs() ** float(p))
    return float(np.sign(df) * (abs(df) ** float(p)))


OPERATORS = {
    "ts_mean": ts_mean,
    "ts_std": ts_std,
    "ts_delta": ts_delta,
    "ts_decay_linear": ts_decay_linear,
    "ts_rank": ts_rank,
    "ts_corr": ts_corr,
    "cs_rank": cs_rank,
    "cs_zscore": cs_zscore,
    "scale": scale,
    "signed_power": signed_power,
    "abs": lambda x: x.abs() if isinstance(x, pd.DataFrame) else abs(x),
    "sign": lambda x: np.sign(x),
}


class FormulaicAlphaEvaluator:
    """Safe AST evaluator for quantitative formulaic alpha expressions."""

    def __init__(self, panel: MarketPanel) -> None:
        self.panel = panel
        self._cache: dict[str, pd.DataFrame] = {}

    def _resolve_variable(self, name: str) -> pd.DataFrame:
        if name in self._cache:
            return self._cache[name]

        if name in {"prices", "close"}:
            res = self.panel.prices.copy()
        elif name in {"returns", "ret"}:
            res = self.panel.returns.copy()
        else:
            try:
                res = _feature_frame(self.panel, name)
            except ValueError:
                raise NameError(f"Variable '{name}' not found in MarketPanel features or prices")

        self._cache[name] = res
        return res

    def evaluate_node(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Expression):
            return self.evaluate_node(node.body)

        if isinstance(node, ast.Constant):
            return node.value

        if isinstance(node, ast.Name):
            return self._resolve_variable(node.id)

        if isinstance(node, ast.UnaryOp):
            operand = self.evaluate_node(node.operand)
            if isinstance(node.op, ast.USub):
                return -operand
            if isinstance(node.op, ast.UAdd):
                return operand
            raise ValueError(f"Unsupported unary operator: {type(node.op)}")

        if isinstance(node, ast.BinOp):
            left = self.evaluate_node(node.left)
            right = self.evaluate_node(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            if isinstance(node.op, ast.Div):
                if isinstance(left, pd.DataFrame) and isinstance(right, pd.DataFrame):
                    return left.div(right.replace(0.0, np.nan)).fillna(0.0)
                return left / right
            if isinstance(node.op, ast.Pow):
                return left**right
            raise ValueError(f"Unsupported binary operator: {type(node.op)}")

        if isinstance(node, ast.Call):
            if not isinstance(node.func, ast.Name):
                raise ValueError("Only direct function calls are supported")
            func_name = node.func.id
            if func_name not in OPERATORS:
                raise NameError(f"Unsupported alpha operator '{func_name}'")
            func = OPERATORS[func_name]
            args = [self.evaluate_node(arg) for arg in node.args]
            return func(*args)

        raise ValueError(f"Forbidden syntax tree node: {type(node).__name__}")


def evaluate_formulaic_alpha(
    expression: str,
    panel: MarketPanel,
    signal_lag: int = 1,
) -> pd.DataFrame:
    """
    Parses and evaluates a formulaic alpha expression across a MarketPanel.

    Parameters:
        expression: Symbolic expression, e.g. "cs_rank(ts_decay_linear(return_5d, 5)) - cs_rank(volatility_20d)"
        panel: Validated MarketPanel instance
        signal_lag: Number of days to lag the final signal to prevent lookahead bias (default: 1)
    """
    panel.validate()
    clean_expr = expression.strip()
    if not clean_expr:
        raise ValueError("Expression cannot be empty")

    try:
        tree = ast.parse(clean_expr, mode="eval")
    except SyntaxError as e:
        raise ValueError(f"Invalid alpha expression syntax: {e}")

    evaluator = FormulaicAlphaEvaluator(panel)
    raw_signal = evaluator.evaluate_node(tree)

    if not isinstance(raw_signal, pd.DataFrame):
        raise TypeError(f"Expression evaluated to {type(raw_signal).__name__}, expected pd.DataFrame")

    if panel.mask is not None:
        raw_signal = raw_signal.where(panel.mask, np.nan)

    raw_signal = raw_signal.replace([np.inf, -np.inf], np.nan)
    return raw_signal.shift(signal_lag)
