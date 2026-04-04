"""
memory/memory.py — Session-scoped, thread-safe in-memory context store.

Design goals:
  • Each session gets an isolated namespace (keyed by session_id).
  • A rolling window of turns prevents unbounded memory growth.
  • TTL-based expiry simulates the behaviour of a real cache (Redis / DynamoDB).
  • The interface is intentionally storage-agnostic so a persistent backend
    (PostgreSQL, Redis) can be dropped in without changing call-sites.

Concurrency:
  A single RLock per store instance makes it safe for async FastAPI handlers
  that run in a thread pool — no coroutine-level issues arise.
"""

import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

from config import settings
from utils.logger import get_logger

logger = get_logger(__name__)


# ── Data structures ───────────────────────────────────────────────────────────

@dataclass
class Turn:
    """One round-trip within a session: input → agent output."""

    role: str                       # "user" | "agent" | "system"
    content: str
    agent_name: str = "system"
    timestamp: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Session:
    """
    Container for all state belonging to a single user session.

    `turns` is a deque with a fixed max-length — when full, the oldest
    turn is automatically discarded (FIFO eviction).
    """

    session_id: str
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    turns: deque = field(init=False)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.turns = deque(maxlen=settings.memory.max_turns_per_session)

    def add_turn(self, turn: Turn) -> None:
        self.turns.append(turn)
        self.last_accessed = time.time()

    def is_expired(self) -> bool:
        age = time.time() - self.last_accessed
        return age > settings.memory.session_ttl_seconds

    def get_context_window(self, n: int = 5) -> list[Turn]:
        """Return the *n* most recent turns for prompt context injection."""
        return list(self.turns)[-n:]

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "turn_count": len(self.turns),
            "turns": [
                {
                    "role": t.role,
                    "content": t.content,
                    "agent_name": t.agent_name,
                    "timestamp": t.timestamp,
                    "metadata": t.metadata,
                }
                for t in self.turns
            ],
            "metadata": self.metadata,
        }


# ── Store ─────────────────────────────────────────────────────────────────────

class MemoryStore:
    """
    Thread-safe in-memory session store.

    Public interface:
        get_or_create_session(session_id) → Session
        add_turn(session_id, turn)        → None
        get_context(session_id, n)        → str   (formatted for LLM prompts)
        get_session(session_id)           → Optional[Session]
        delete_session(session_id)        → bool
        list_sessions()                   → list[str]
        stats()                           → dict
    """

    def __init__(self) -> None:
        self._sessions: dict[str, Session] = {}
        self._lock = threading.RLock()
        logger.info("MemoryStore initialised (max_sessions=%d)", settings.memory.max_sessions)

    # ── Session lifecycle ─────────────────────────────────────────────────────

    def get_or_create_session(self, session_id: str) -> Session:
        """Return an existing session or create a new one."""
        with self._lock:
            self._evict_if_needed()

            if session_id not in self._sessions:
                if len(self._sessions) >= settings.memory.max_sessions:
                    raise MemoryError(
                        f"Session limit ({settings.memory.max_sessions}) reached. "
                        "Cannot create a new session."
                    )
                self._sessions[session_id] = Session(session_id=session_id)
                logger.info("Session created | session_id=%s", session_id)
            else:
                session = self._sessions[session_id]
                if session.is_expired():
                    # Re-create rather than serve stale data
                    logger.info("Session expired, resetting | session_id=%s", session_id)
                    self._sessions[session_id] = Session(session_id=session_id)

            return self._sessions[session_id]

    def get_session(self, session_id: str) -> Optional[Session]:
        """Return session if it exists and is not expired, else None."""
        with self._lock:
            session = self._sessions.get(session_id)
            if session is None or session.is_expired():
                return None
            return session

    def delete_session(self, session_id: str) -> bool:
        """Delete a session. Returns True if it existed."""
        with self._lock:
            existed = session_id in self._sessions
            self._sessions.pop(session_id, None)
            if existed:
                logger.info("Session deleted | session_id=%s", session_id)
            return existed

    # ── Turn management ───────────────────────────────────────────────────────

    def add_turn(self, session_id: str, turn: Turn) -> None:
        """Append a turn to the session's rolling history."""
        with self._lock:
            session = self.get_or_create_session(session_id)
            session.add_turn(turn)
            logger.debug(
                "Turn added | session_id=%s role=%s agent=%s",
                session_id, turn.role, turn.agent_name,
            )

    def get_context(self, session_id: str, n: int = 5) -> str:
        """
        Return the last *n* turns formatted as a plain-text context block
        suitable for injection into an LLM prompt.
        """
        with self._lock:
            session = self._sessions.get(session_id)
            if not session:
                return ""

            turns = session.get_context_window(n)
            if not turns:
                return ""

            lines = []
            for t in turns:
                prefix = f"[{t.role.upper()} via {t.agent_name}]"
                lines.append(f"{prefix} {t.content}")

            return "\n".join(lines)

    # ── Housekeeping ──────────────────────────────────────────────────────────

    def list_sessions(self) -> list[str]:
        with self._lock:
            return list(self._sessions.keys())

    def stats(self) -> dict[str, Any]:
        with self._lock:
            active = [s for s in self._sessions.values() if not s.is_expired()]
            return {
                "total_sessions": len(self._sessions),
                "active_sessions": len(active),
                "max_sessions": settings.memory.max_sessions,
                "total_turns": sum(len(s.turns) for s in active),
            }

    def _evict_if_needed(self) -> None:
        """Remove expired sessions to free capacity (called under lock)."""
        expired = [sid for sid, s in self._sessions.items() if s.is_expired()]
        for sid in expired:
            del self._sessions[sid]
            logger.debug("Session evicted (TTL) | session_id=%s", sid)


# ── Module-level singleton ────────────────────────────────────────────────────

memory_store = MemoryStore()
