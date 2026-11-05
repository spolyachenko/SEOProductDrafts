"""Run with a live LLM at LLM_BASE_URL: the example tests await the real model."""
import asyncio
import json
from pathlib import Path

import httpx
import pytest

import main

EXAMPLES = json.loads(Path(__file__).with_name("examples.json").read_text(encoding="utf-8"))


@pytest.fixture
async def client():
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=main.app),
                                 base_url="http://test", timeout=180) as c:
        yield c


async def test_generate(client):
    res = await asyncio.gather(*(client.post("/generate", json=p) for p in EXAMPLES))
    for product, r in zip(EXAMPLES, res):
        assert r.status_code == 200, f"{product['name']}: {r.text}"


@pytest.mark.parametrize("body", [{}, {"name": "  "}, {"name": "X", "specs": "abc"}])
async def test_invalid_body(client, body):
    assert (await client.post("/generate", json=body)).status_code == 422


async def test_malformed_json(client):
    r = await client.post("/generate", content=b'{"name": ', headers={"content-type": "application/json"})
    assert r.status_code == 422
