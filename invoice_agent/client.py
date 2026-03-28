"""WebSocket-based client for InvoiceAgent environment."""

from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, Optional, Tuple

from invoice_agent.models import InvoiceAction, InvoiceObservation


class InvoiceAgentClient:
    """Async WebSocket client for the InvoiceAgent environment."""

    def __init__(self, base_url: str = "http://localhost:8000") -> None:
        self.base_url = base_url.rstrip("/")
        self._ws_url = self.base_url.replace("http://", "ws://").replace("https://", "wss://") + "/ws"
        self._ws = None

    async def __aenter__(self) -> "InvoiceAgentClient":
        import websockets
        self._ws = await websockets.connect(self._ws_url)
        return self

    async def __aexit__(self, *args: Any) -> None:
        if self._ws:
            await self._ws.close()

    async def reset(
        self, task_id: str = "easy", seed: int = 42
    ) -> Tuple[InvoiceObservation, float, bool, Dict[str, Any]]:
        """Reset the environment."""
        await self._ws.send(json.dumps({
            "method": "reset",
            "params": {"task_id": task_id, "seed": seed},
        }))
        resp = json.loads(await self._ws.recv())
        obs = InvoiceObservation(**resp["observation"])
        return obs, resp["reward"], resp["done"], resp["info"]

    async def step(
        self, action: InvoiceAction
    ) -> Tuple[InvoiceObservation, float, bool, Dict[str, Any]]:
        """Take a step in the environment."""
        await self._ws.send(json.dumps({
            "method": "step",
            "params": action.model_dump(),
        }))
        resp = json.loads(await self._ws.recv())
        if "error" in resp:
            raise RuntimeError(resp["error"])
        obs = InvoiceObservation(**resp["observation"])
        return obs, resp["reward"], resp["done"], resp["info"]

    async def state(self) -> Dict[str, Any]:
        """Get current environment state."""
        await self._ws.send(json.dumps({"method": "state", "params": {}}))
        resp = json.loads(await self._ws.recv())
        return resp.get("state", resp)

    def sync(self) -> "SyncInvoiceAgentClient":
        """Return a synchronous wrapper."""
        return SyncInvoiceAgentClient(self)


class SyncInvoiceAgentClient:
    """Synchronous wrapper around the async client."""

    def __init__(self, async_client: InvoiceAgentClient) -> None:
        self._client = async_client
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def __enter__(self) -> "SyncInvoiceAgentClient":
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._client.__aenter__())
        return self

    def __exit__(self, *args: Any) -> None:
        if self._loop:
            self._loop.run_until_complete(self._client.__aexit__(*args))
            self._loop.close()

    def reset(self, task_id: str = "easy", seed: int = 42):
        return self._loop.run_until_complete(self._client.reset(task_id, seed))

    def step(self, action: InvoiceAction):
        return self._loop.run_until_complete(self._client.step(action))

    def state(self):
        return self._loop.run_until_complete(self._client.state())
