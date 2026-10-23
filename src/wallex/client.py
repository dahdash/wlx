"""Thin synchronous client for the Wallex REST API (https://api.wallex.ir).

Verified against the Wallex Swagger files in ``docs/qa/`` and live read-only
calls (2026-10-22): base URL, the ``X-API-Key`` header, the
``{"success", "message", "result"}`` envelope, ``GET /v1/markets``,
``GET /v1/depth``, ``GET /v1/account/balances``, and the spot
create/cancel/open-orders endpoints. Order submission has not been exercised.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx

BASE_URL = "https://api.wallex.ir"
ENV_FILE = Path(__file__).resolve().parents[2] / ".env"


def load_env(path: str | Path = ENV_FILE) -> None:
    """Read ``KEY=value`` lines from an untracked ``.env``; variables already set win."""
    p = Path(path)
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        k, sep, v = line.strip().partition("=")
        if sep and k and not k.startswith("#"):
            os.environ.setdefault(k.removeprefix("export ").strip(), v.strip().strip("'\""))


class WallexError(Exception):
    """API returned success=false, or a non-2xx HTTP status."""

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class WallexClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        base_url: str = BASE_URL,
        timeout: float = 10.0,
        transport: httpx.BaseTransport | None = None,
    ):
        self.api_key = api_key or os.environ.get("WALLEX_API_KEY")
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        self._http = httpx.Client(
            base_url=base_url, headers=headers, timeout=timeout, transport=transport
        )

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> WallexClient:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        resp = self._http.request(method, path, **kwargs)
        try:
            body = resp.json()
        except ValueError:
            body = None
        if resp.status_code >= 400 or not isinstance(body, dict) or not body.get("success"):
            msg = body.get("message") if isinstance(body, dict) else resp.text[:200]
            if isinstance(body, dict) and isinstance(body.get("result"), dict) and body["result"]:
                msg = f"{msg} {json.dumps(body['result'], ensure_ascii=False)}"  # field errors
            raise WallexError(f"{method} {path}: {msg}", resp.status_code)
        return body.get("result")

    # --- public market data -------------------------------------------------

    def markets(self) -> dict[str, dict[str, Any]]:
        """``{"symbols": {symbol: {...}}}`` with per-market order constraints:
        ``minQty``, ``minNotional``, ``stepSize``/``tickSize`` (decimals) and
        ``isMarketTypeEnable``. The richer ``/hector/web/v1/markets`` endpoint
        adds volumes, ``fair_price`` and ``is_spot``/``is_otc`` flags.
        """
        return self._request("GET", "/v1/markets")

    def depth(self, symbol: str) -> dict[str, list[dict[str, str]]]:
        """Order book: ``{"ask": [{price, quantity, sum}], "bid": [...]}``.

        Docs advise against polling; use the WebSocket feed for live depth.
        """
        return self._request("GET", "/v1/depth", params={"symbol": symbol})

    def candles(self, symbol: str, resolution: str, start: int, end: int) -> dict[str, list]:
        """UDF ``/v1/udf/history`` (no envelope): ``{"s", "t", "o", "h", "l", "c", "v"}``,
        ``t`` = bar open time (Unix s). Verified 2026-10-22: resolutions 1/15/60/240/D (not
        5/30); one 1-min request covers ≤ 1 day (~14 d of history); 60-min ≈ 29 d.
        ``s == "no_data"`` returns empty lists."""
        params = {"symbol": symbol, "resolution": resolution, "from": start, "to": end}
        resp = self._http.get("/v1/udf/history", params=params)
        if resp.status_code >= 400:
            raise WallexError(f"GET /v1/udf/history: {resp.text[:200]}", resp.status_code)
        body = resp.json()
        if body.get("s") == "no_data":
            return {k: [] for k in "tohlcv"}
        if body.get("s") != "ok":
            raise WallexError(f"GET /v1/udf/history: {body}", resp.status_code)
        return body

    # --- account (require API key) ------------------------------------------

    def balances(self) -> dict[str, dict[str, Any]]:
        """``{"balances": {asset: {"value", "locked", ...}}}``."""
        return self._request("GET", "/v1/account/balances")

    def open_orders(
        self, symbol: str | None = None, page: int | None = None, per_page: int | None = None
    ) -> dict[str, list[dict[str, Any]]]:
        """Unfilled orders. Live (2026-10-22): ``{"orders": [...]}``; docs say keyed by symbol."""
        params = {
            k: v for k, v in {"symbol": symbol, "page": page, "per_page": per_page}.items() if v
        }
        return self._request("GET", "/v1/account/openOrders", params=params)

    def place_order(
        self,
        symbol: str,
        side: str,
        type: str,
        quantity: str,
        price: str | None = None,
        *,
        stop_price: str | None = None,
        client_id: str | None = None,
        live: bool = False,
    ) -> dict[str, Any]:
        """Place an order. Dry-run by default: returns the payload without sending.

        ``type``: LIMIT, MARKET, STOP_LIMIT or STOP_MARKET (stop types need
        ``stop_price``). Docs limit order creation to 20 requests / 10 s.

        Pass ``live=True`` to actually submit. Quantity/price are strings to
        avoid float rounding.
        """
        payload: dict[str, Any] = {
            "symbol": symbol,
            "side": side.upper(),
            "type": type.upper(),
            "quantity": quantity,
        }
        if price is not None:
            payload["price"] = price
        if stop_price is not None:
            payload["stop_Price"] = stop_price  # sic: capital P per the docs
        if client_id is not None:
            payload["client_id"] = client_id
        if not live:
            return {"dry_run": True, "payload": payload}
        return self._request("POST", "/v1/account/orders", json=payload)

    def order(self, client_order_id: str) -> dict[str, Any]:
        """Order details (``active``, ``executedQty``, ``executedPrice``, ``fee``, ``fills``)."""
        return self._request("GET", f"/v1/account/orders/{client_order_id}")

    def cancel_order(self, client_order_id: str, *, live: bool = False) -> Any:
        if not live:
            return {"dry_run": True, "clientOrderId": client_order_id}
        return self._request("DELETE", f"/v1/account/orders/{client_order_id}")
