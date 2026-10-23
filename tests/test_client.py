import os

import httpx
import pytest

from wallex import WallexClient, WallexError


def make(handler):
    return WallexClient("k", transport=httpx.MockTransport(handler))


def test_depth_sends_api_key_and_unwraps_result():
    def handler(req: httpx.Request) -> httpx.Response:
        assert req.url.path == "/v1/depth"
        assert req.url.params["symbol"] == "BTCUSDT"
        assert req.headers["X-API-Key"] == "k"
        return httpx.Response(
            200, json={"success": True, "message": "", "result": {"ask": [], "bid": []}}
        )

    assert make(handler).depth("BTCUSDT") == {"ask": [], "bid": []}


def test_success_false_raises():
    c = make(
        lambda r: httpx.Response(200, json={"success": False, "message": "bad", "result": None})
    )
    with pytest.raises(WallexError, match="bad"):
        c.markets()


def test_error_includes_field_errors():
    body = {"success": False, "message": "invalid", "code": 422, "result": {"symbol": ["closed"]}}
    c = make(lambda r: httpx.Response(422, json=body))
    with pytest.raises(WallexError, match=r'invalid \{"symbol": \["closed"\]\}') as e:
        c.markets()
    assert e.value.status_code == 422


def test_place_order_dry_run_does_not_hit_network():
    def handler(req):
        raise AssertionError("network used in dry-run")

    out = make(handler).place_order("BTCUSDT", "buy", "limit", "0.01", "50000")
    assert out["dry_run"] and out["payload"]["side"] == "BUY"


def test_live_order_and_cancel_use_documented_paths():
    seen = []

    def handler(req):
        seen.append((req.method, req.url.path, req.content))
        return httpx.Response(201, json={"success": True, "message": "", "result": {}})

    c = make(handler)
    c.place_order("BTCUSDT", "sell", "stop_limit", "1", "9", stop_price="8", live=True)
    c.cancel_order("abc", live=True)
    assert seen[0][:2] == ("POST", "/v1/account/orders") and b"stop_Price" in seen[0][2]
    assert seen[1][:2] == ("DELETE", "/v1/account/orders/abc")


def test_load_env_keeps_existing(tmp_path, monkeypatch):
    from wallex.client import load_env

    f = tmp_path / ".env"
    f.write_text("# c\nWLX_A='x1'\nexport WLX_B=y2\nWLX_C=z\n")
    monkeypatch.delenv("WLX_A", raising=False)
    monkeypatch.delenv("WLX_B", raising=False)
    monkeypatch.setenv("WLX_C", "keep")
    load_env(f)
    assert (os.environ["WLX_A"], os.environ["WLX_B"], os.environ["WLX_C"]) == ("x1", "y2", "keep")
    monkeypatch.delenv("WLX_A")
    monkeypatch.delenv("WLX_B")
