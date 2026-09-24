"""A transport that times the vendor, and nothing else.

Latency is a headline number, so what the clock covers has to be decided
rather than inherited. Three things could contaminate it, and each is
handled here rather than hoped about.

*Our own queuing.* The ``decisions`` client starts its clock before it
acquires the shared in-flight semaphore, so a request that waits on our gate
reports that wait as vendor latency. At concurrency 1 the gate is never
contended and it costs nothing; at concurrency 16, or with
``--permutations 3``, it would quietly inflate whichever arm happened to
arrive when the gate was full. Here the clock starts *after* the gate, and
the wait is recorded separately so "the gate was never contended" is a
measurement rather than an assumption.

*Vendor throttling.* A 429 or a 503 is not a slow model, it is a closed
door. Those responses are counted per client and surfaced in the report, and
the retry that follows is excluded from the reported latency -- only the
successful attempt's time is kept. An arm that saw throttling gets its
latency flagged rather than silently averaged.

*Connection setup.* Inherited: the pooled keep-alive connection is the
parent's, and both hosted arms share the same transport, because a fresh TLS
handshake costs more than either forward pass.

``post`` keeps the parent's ``(body, seconds)`` signature; the extra
per-call detail is read afterwards from :meth:`last`, which is thread-local
because the runner fans out over threads.
"""

from __future__ import annotations

import json
import threading
import time

from decisions.client import APIError, OpenAIClient, _gate

#: Statuses that mean "not now" rather than "here is your answer".
THROTTLE_STATUS = {429, 503, 529}


class TimedClient(OpenAIClient):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.throttled = 0
        self._local = threading.local()

    def last(self) -> dict:
        """Detail of this thread's most recent call."""
        return getattr(self._local, "detail", {"gate_wait_s": 0.0,
                                               "throttled": 0})

    def post(self, path: str, payload: dict) -> tuple[dict, float]:
        body = json.dumps(payload).encode()
        headers = {"Content-Type": "application/json",
                   "Content-Length": str(len(body)),
                   "Accept": "application/json",
                   "Connection": "keep-alive"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        target = self._path + path
        throttled = 0
        for attempt in (0, 1):
            queued = time.perf_counter()
            try:
                with _gate:
                    # The clock starts here: everything before it was ours.
                    waited = time.perf_counter() - queued
                    started = time.perf_counter()
                    with self._connection() as conn:
                        conn.request("POST", target, body=body, headers=headers)
                        response = conn.getresponse()
                        raw = response.read().decode("utf-8", "replace")
                        status = response.status
                    elapsed = time.perf_counter() - started
            except (OSError, ConnectionError):
                if attempt:
                    raise
                continue
            if status in THROTTLE_STATUS:
                throttled += 1
                self.throttled += 1
            self._local.detail = {"gate_wait_s": waited, "throttled": throttled}
            if status >= 400:
                raise APIError(status, raw)
            return json.loads(raw), elapsed
        raise RuntimeError("unreachable")
