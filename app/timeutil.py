"""RFC3339 时间解析与数据库时间戳格式化。

历史时点校验 (GET /verify/{digest}?at=...) 使用:
- 证明/撤销记录的时间戳由 SQLite 以 UTC 毫秒精度字符串存储
  (strftime('%Y-%m-%dT%H:%M:%fZ', 'now')), 同格式字符串可按字典序直接比较。
- at 参数必须携带时区偏移 (Z 或 ±HH:MM), 统一归一化为 UTC 后再比较。
"""
from __future__ import annotations

from datetime import datetime, timezone


def parse_rfc3339(value: str) -> datetime:
    """解析 RFC3339 时间戳并归一化为 UTC。

    要求显式携带 UTC 偏移 (如 ``Z`` / ``+00:00``); 裸时间 (无时区) 拒绝。
    解析失败抛出 ValueError。
    """
    text = (value or "").strip()
    if not text:
        raise ValueError("时间戳为空")
    try:
        dt = datetime.fromisoformat(text)
    except ValueError as exc:
        raise ValueError(f"不是合法的 RFC3339 时间戳: {value!r}") from exc
    if dt.tzinfo is None:
        raise ValueError("时间戳必须携带 UTC 偏移 (如 'Z' 或 '+00:00')")
    return dt.astimezone(timezone.utc)


def format_db_timestamp(dt: datetime) -> str:
    """格式化为与 SQLite 默认时间戳一致的 UTC 毫秒字符串。

    与 proofs.created_at / revocations.revoked_at 同格式, 可直接做字符串比较。
    微秒截断到毫秒 (不四舍五入), 保证 created_at <= at 的边界语义。
    """
    dt = dt.astimezone(timezone.utc)
    return f"{dt.strftime('%Y-%m-%dT%H:%M:%S')}.{dt.microsecond // 1000:03d}Z"


def utcnow() -> datetime:
    """当前 UTC 时间 (aware)。"""
    return datetime.now(timezone.utc)
