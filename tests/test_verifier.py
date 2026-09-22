"""端到端测试: 通过 FastAPI TestClient + 内存 SQLite 覆盖全部校验分支。"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app import crypto  # noqa: E402
from app.config import Settings  # noqa: E402
from app.database import init_db  # noqa: E402
from app.main import app, db  # noqa: E402


@pytest.fixture()
def client():
    import sqlite3

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)

    def _db():
        yield conn

    app.dependency_overrides[db] = _db
    # 开发模式: 不配置信任根, 任意自签名根均可接受
    cfg = Settings(
        database_path=":memory:",
        trusted_root_issuers=frozenset(),
        max_chain_depth=256,
    )
    import app.main as main_mod

    main_mod.settings = cfg
    try:
        with TestClient(app) as c:
            yield c, conn
    finally:
        app.dependency_overrides.clear()


def _keypair():
    priv = Ed25519PrivateKey.generate()
    pub = crypto.encode_public_key(priv.public_key())
    return priv, pub


def _sign(priv, artifact, parent=None):
    return crypto.sign(priv, artifact, parent)


def _post_proof(client, priv, pub, artifact, parent=None, signature=None):
    sig = signature if signature is not None else _sign(priv, artifact, parent)
    return client.post(
        "/proofs",
        json={
            "artifact_digest": artifact,
            "issuer": pub,
            "parent_digest": parent,
            "signature": sig,
        },
    )


def _build_chain(client, root_priv, root_pub, n=3):
    """root -> a1 -> a2 ... 全部由同一密钥签名。"""
    digests = ["sha256:root"]
    r = _post_proof(client, root_priv, root_pub, digests[0])
    assert r.status_code == 201, r.text
    for i in range(1, n):
        d = f"sha256:a{i}"
        r = _post_proof(client, root_priv, root_pub, d, parent=digests[-1])
        assert r.status_code == 201, r.text
        digests.append(d)
    return digests


def _ts(dt: datetime) -> str:
    """格式化为与 SQLite 默认时间戳一致的 UTC 毫秒字符串。"""
    dt = dt.astimezone(timezone.utc)
    return dt.strftime("%Y-%m-%dT%H:%M:%S.") + f"{dt.microsecond // 1000:03d}Z"


def _set_created_at(conn, artifact, ts, issuer=None):
    """直接把证明的 created_at 改成指定时间, 构造确定性的历史时间线。"""
    if issuer is None:
        conn.execute(
            "UPDATE proofs SET created_at = ? WHERE artifact_digest = ?",
            (ts, artifact),
        )
    else:
        conn.execute(
            "UPDATE proofs SET created_at = ? WHERE artifact_digest = ? AND issuer = ?",
            (ts, artifact, issuer),
        )
    conn.commit()


def _set_revoked_at(conn, artifact, ts):
    conn.execute(
        "UPDATE revocations SET revoked_at = ? WHERE artifact_digest = ?",
        (ts, artifact),
    )
    conn.commit()


# ---------- 基础 ----------


def test_health(client):
    c, _ = client
    assert c.get("/health").json() == {"status": "ok"}


def test_crypto_roundtrip():
    priv, pub = _keypair()
    sig = crypto.sign(priv, "sha256:x", "sha256:y")
    assert crypto.verify(pub, "sha256:x", "sha256:y", sig)
    assert not crypto.verify(pub, "sha256:x", "sha256:other", sig)
    assert not crypto.verify(pub, "sha256:z", "sha256:y", sig)


# ---------- 成功路径 ----------


def test_valid_chain(client):
    c, _ = client
    priv, pub = _keypair()
    digests = _build_chain(c, priv, pub, n=3)

    res = c.get(f"/verify/{digests[-1]}")
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["length"] == 3
    assert [e["artifact_digest"] for e in body["chain"]] == list(reversed(digests))
    assert body["chain"][0]["relation"] == "target"
    assert body["chain"][-1]["relation"] == "root"
    assert body["root_issuer"] == pub


def test_root_self_signed(client):
    c, _ = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:only")
    body = c.get("/verify/sha256:only").json()
    assert body["status"] == "OK"
    assert body["chain"][0]["relation"] == "root"


# ---------- 写入语义 ----------


def test_duplicate_write_is_idempotent(client):
    c, _ = client
    priv, pub = _keypair()
    r1 = _post_proof(c, priv, pub, "sha256:d")
    r2 = _post_proof(c, priv, pub, "sha256:d")
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.json()["deduplicated"] is True
    assert r1.json()["proof_id"] == r2.json()["proof_id"]


def test_bad_issuer_key_rejected(client):
    c, _ = client
    r = c.post(
        "/proofs",
        json={
            "artifact_digest": "sha256:x",
            "issuer": "not-base64!!!",
            "parent_digest": None,
            "signature": "AAAA",
        },
    )
    assert r.status_code == 400


# ---------- 分叉 ----------


def test_fork_at_target(client):
    c, _ = client
    k1, p1 = _keypair()
    k2, p2 = _keypair()
    _post_proof(c, k1, p1, "sha256:root")
    _post_proof(c, k1, p1, "sha256:child", parent="sha256:root")
    # 同一 child, 另一个签发者 -> 分叉
    _post_proof(c, k2, p2, "sha256:child", parent="sha256:root")

    body = c.get("/verify/sha256:child").json()
    assert body["status"] == "FORK"
    assert body["failure"]["at"]["artifact_digest"] == "sha256:child"
    assert len(body["failure"]["branches"]) == 2


def test_fork_at_ancestor_has_failure_path(client):
    c, _ = client
    k1, p1 = _keypair()
    k2, p2 = _keypair()
    _post_proof(c, k1, p1, "sha256:root")
    # mid 两条分叉证明
    _post_proof(c, k1, p1, "sha256:mid", parent="sha256:root")
    _post_proof(c, k2, p2, "sha256:mid", parent="sha256:root")
    _post_proof(c, k1, p1, "sha256:leaf", parent="sha256:mid")

    body = c.get("/verify/sha256:leaf").json()
    assert body["status"] == "FORK"
    path = body["failure_path"]
    assert [e["artifact_digest"] for e in path] == ["sha256:leaf", "sha256:mid"]
    assert path[-1]["relation"] == "fork_point"
    assert len(body["failure"]["branches"]) == 2


# ---------- 撤销 ----------


def test_revoked_target(client):
    c, _ = client
    priv, pub = _keypair()
    _build_chain(c, priv, pub, n=2)
    r = c.post("/revocations", json={"artifact_digest": "sha256:a1", "reason": "CVE"})
    assert r.status_code == 201
    body = c.get("/verify/sha256:a1").json()
    assert body["status"] == "REVOKED"
    assert body["failure"]["at"]["relation"] == "revoked"
    assert body["failure"]["at"]["revocation"]["reason"] == "CVE"


def test_revoked_ancestor_locatable_path(client):
    c, _ = client
    priv, pub = _keypair()
    digests = _build_chain(c, priv, pub, n=4)
    c.post("/revocations", json={"artifact_digest": "sha256:a1", "reason": "ROTATED"})

    body = c.get(f"/verify/{digests[-1]}").json()
    assert body["status"] == "REVOKED"
    path = body["failure_path"]
    assert [e["artifact_digest"] for e in path] == [
        "sha256:a3",
        "sha256:a2",
        "sha256:a1",
    ]
    assert path[-1]["relation"] == "revoked_ancestor"
    assert path[-1]["revocation"]["reason"] == "ROTATED"


def test_revocation_idempotent(client):
    c, _ = client
    r1 = c.post("/revocations", json={"artifact_digest": "sha256:x", "reason": "a"})
    r2 = c.post("/revocations", json={"artifact_digest": "sha256:x", "reason": "b"})
    assert r1.status_code == 201
    assert r2.status_code == 200
    assert r2.json()["already_revoked"] is True
    assert r2.json()["reason"] == "a"


# ---------- 签名 / 父链 / 根 ----------


def test_invalid_signature_target(client):
    c, _ = client
    priv, pub = _keypair()
    other, _ = _keypair()
    _post_proof(c, priv, pub, "sha256:root")
    bad_sig = _sign(other, "sha256:child", "sha256:root")
    _post_proof(c, priv, pub, "sha256:child", parent="sha256:root", signature=bad_sig)

    body = c.get("/verify/sha256:child").json()
    assert body["status"] == "INVALID_SIGNATURE"
    assert body["failure"]["at"]["artifact_digest"] == "sha256:child"


def test_invalid_signature_ancestor(client):
    c, _ = client
    k1, p1 = _keypair()
    k2, _ = _keypair()
    _post_proof(c, k1, p1, "sha256:root")
    bad = _sign(k2, "sha256:mid", "sha256:root")
    _post_proof(c, k1, p1, "sha256:mid", parent="sha256:root", signature=bad)
    _post_proof(c, k1, p1, "sha256:leaf", parent="sha256:mid")

    body = c.get("/verify/sha256:leaf").json()
    assert body["status"] == "INVALID_SIGNATURE"
    assert [e["artifact_digest"] for e in body["failure_path"]] == [
        "sha256:leaf",
        "sha256:mid",
    ]


def test_missing_parent(client):
    c, _ = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:orphan", parent="sha256:ghost")
    body = c.get("/verify/sha256:orphan").json()
    assert body["status"] == "MISSING_PARENT"
    assert body["failure"]["missing_artifact"] == "sha256:ghost"
    assert body["failure_path"][0]["artifact_digest"] == "sha256:orphan"


def test_cycle_detected(client):
    c, conn = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:a", parent="sha256:b")
    _post_proof(c, priv, pub, "sha256:b", parent="sha256:a")

    body = c.get("/verify/sha256:a").json()
    assert body["status"] == "CYCLE"
    assert body["failure"]["at"]["relation"] == "cycle_point"


def test_verify_unknown_404(client):
    c, _ = client
    assert c.get("/verify/sha256:nope").status_code == 404


# ---------- 信任根 ----------


def test_untrusted_root_enforced():
    import sqlite3

    from fastapi.testclient import TestClient

    from app.main import app as app2, db as db_dep

    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    app2.dependency_overrides[db_dep] = lambda: (yield conn)

    trusted_priv = Ed25519PrivateKey.generate()
    trusted_pub = crypto.encode_public_key(trusted_priv.public_key())
    rogue_priv, rogue_pub = _keypair()

    import app.main as main_mod

    main_mod.settings = Settings(
        ":memory:", frozenset({trusted_pub}), 256
    )

    with TestClient(app2) as c:
        # 非信任根
        _post_proof(c, rogue_priv, rogue_pub, "sha256:r")
        body = c.get("/verify/sha256:r").json()
        assert body["status"] == "UNTRUSTED_ROOT"

        # 信任根 + 合法下游
        _post_proof(c, trusted_priv, trusted_pub, "sha256:good-root")
        _post_proof(
            c, trusted_priv, trusted_pub, "sha256:good-child",
            parent="sha256:good-root",
        )
        body = c.get("/verify/sha256:good-child").json()
        assert body["status"] == "OK"
        assert body["length"] == 2

    app2.dependency_overrides.clear()


# ---------- 历史时点校验 (?at=) ----------


def test_at_invalid_returns_400(client):
    c, _ = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:x")

    for bad in [
        "not-a-time",
        "2026-13-01T00:00:00Z",        # 非法月份
        "2026-01-01T00:00:00",         # 裸时间, 缺少 UTC 偏移
        "2026-01-01 00:00:00",         # 非 RFC3339 分隔符且无时区
        "",                            # 空串
    ]:
        res = c.get("/verify/sha256:x", params={"at": bad})
        assert res.status_code == 400, bad


def test_at_future_returns_400(client):
    c, _ = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:x")

    future = _ts(datetime.now(timezone.utc) + timedelta(days=1))
    res = c.get("/verify/sha256:x", params={"at": future})
    assert res.status_code == 400


def test_at_before_target_proof_returns_404(client):
    c, conn = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:x")

    now = datetime.now(timezone.utc)
    _set_created_at(conn, "sha256:x", _ts(now - timedelta(days=1)))

    # 该时点目标证明尚未写入 -> 与"不存在"语义一致, 404
    res = c.get("/verify/sha256:x", params={"at": _ts(now - timedelta(days=2))})
    assert res.status_code == 404


def test_created_at_boundary_inclusive(client):
    c, conn = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:only")

    t0 = datetime.now(timezone.utc) - timedelta(days=1)
    _set_created_at(conn, "sha256:only", _ts(t0))

    # at 恰好等于 created_at -> 记录可见 (边界取等)
    res = c.get("/verify/sha256:only", params={"at": _ts(t0)})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["evaluated_at"] == _ts(t0)

    # 早 1 毫秒 -> 记录尚不可见
    res = c.get("/verify/sha256:only", params={"at": _ts(t0 - timedelta(milliseconds=1))})
    assert res.status_code == 404


def test_at_offset_normalized_to_utc(client):
    c, conn = client
    priv, pub = _keypair()
    _post_proof(c, priv, pub, "sha256:only")

    t0 = datetime.now(timezone.utc) - timedelta(days=1)
    _set_created_at(conn, "sha256:only", _ts(t0))

    # 带 +08:00 偏移, 与 UTC 边界同一瞬间 -> 可见, evaluated_at 归一化为 Z
    t0_ms = t0.replace(microsecond=(t0.microsecond // 1000) * 1000)
    local = t0_ms.astimezone(timezone(timedelta(hours=8)))
    at_param = (
        local.strftime("%Y-%m-%dT%H:%M:%S.")
        + f"{local.microsecond // 1000:03d}+08:00"
    )
    res = c.get("/verify/sha256:only", params={"at": at_param})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["evaluated_at"] == _ts(t0)


def test_late_fork_does_not_change_earlier_verdict(client):
    c, conn = client
    k1, p1 = _keypair()
    k2, p2 = _keypair()
    _post_proof(c, k1, p1, "sha256:root")
    _post_proof(c, k1, p1, "sha256:child", parent="sha256:root")
    # 晚到的分叉证明 (同一 child, 另一个签发者)
    _post_proof(c, k2, p2, "sha256:child", parent="sha256:root")

    now = datetime.now(timezone.utc)
    t_root, t_child = now - timedelta(days=3), now - timedelta(days=3, minutes=-1)
    t_fork = now - timedelta(days=1)
    _set_created_at(conn, "sha256:root", _ts(t_root))
    _set_created_at(conn, "sha256:child", _ts(t_child), issuer=p1)
    _set_created_at(conn, "sha256:child", _ts(t_fork), issuer=p2)

    # 分叉写入之前的时点: 结论仍是 OK, 且带 evaluated_at
    t_before = now - timedelta(days=2)
    res = c.get("/verify/sha256:child", params={"at": _ts(t_before)})
    assert res.status_code == 200
    body = res.json()
    assert body["status"] == "OK"
    assert body["length"] == 2
    assert body["evaluated_at"] == _ts(t_before)

    # 分叉生效瞬间 (边界取等): FORK
    body = c.get("/verify/sha256:child", params={"at": _ts(t_fork)}).json()
    assert body["status"] == "FORK"
    assert len(body["failure"]["branches"]) == 2
    assert body["evaluated_at"] == _ts(t_fork)

    # 不传 at: 行为不变, 仍是 FORK, 且无 evaluated_at
    body = c.get("/verify/sha256:child").json()
    assert body["status"] == "FORK"
    assert "evaluated_at" not in body


def test_revocation_effective_only_after_revoked_at(client):
    c, conn = client
    priv, pub = _keypair()
    digests = _build_chain(c, priv, pub, n=3)  # root -> a1 -> a2
    c.post("/revocations", json={"artifact_digest": "sha256:a1", "reason": "CVE"})

    now = datetime.now(timezone.utc)
    _set_created_at(conn, "sha256:root", _ts(now - timedelta(days=3)))
    _set_created_at(conn, "sha256:a1", _ts(now - timedelta(days=3, minutes=-1)))
    _set_created_at(conn, "sha256:a2", _ts(now - timedelta(days=3, minutes=-2)))
    t_revoked = now - timedelta(days=1)
    _set_revoked_at(conn, "sha256:a1", _ts(t_revoked))

    # 撤销生效前: 下游完整链 OK
    t_before = now - timedelta(days=2)
    body = c.get(f"/verify/{digests[-1]}", params={"at": _ts(t_before)}).json()
    assert body["status"] == "OK"
    assert body["length"] == 3
    assert body["evaluated_at"] == _ts(t_before)

    # 目标自身在撤销生效前同样 OK
    body = c.get("/verify/sha256:a1", params={"at": _ts(t_before)}).json()
    assert body["status"] == "OK"

    # 撤销生效瞬间 (边界取等): 下游 REVOKED, 失败路径定位到被撤销祖先
    body = c.get(f"/verify/{digests[-1]}", params={"at": _ts(t_revoked)}).json()
    assert body["status"] == "REVOKED"
    assert body["failure_path"][-1]["artifact_digest"] == "sha256:a1"
    assert body["failure_path"][-1]["relation"] == "revoked_ancestor"
    assert body["evaluated_at"] == _ts(t_revoked)

    # 目标自身在撤销生效后 REVOKED
    body = c.get("/verify/sha256:a1", params={"at": _ts(t_revoked)}).json()
    assert body["status"] == "REVOKED"
    assert body["failure"]["at"]["relation"] == "revoked"


def test_late_parent_arrival_keeps_earlier_missing_parent(client):
    c, conn = client
    priv, pub = _keypair()
    # 子证明先到, 父证明尚未写入
    _post_proof(c, priv, pub, "sha256:child", parent="sha256:root")

    now = datetime.now(timezone.utc)
    _set_created_at(conn, "sha256:child", _ts(now - timedelta(days=3)))
    t_mid = now - timedelta(days=2)

    body = c.get("/verify/sha256:child", params={"at": _ts(t_mid)}).json()
    assert body["status"] == "MISSING_PARENT"
    assert body["failure"]["missing_artifact"] == "sha256:root"

    # 父证明晚到
    _post_proof(c, priv, pub, "sha256:root")
    _set_created_at(conn, "sha256:root", _ts(now - timedelta(days=1)))

    # 更早时点结论不变: 仍是 MISSING_PARENT
    body = c.get("/verify/sha256:child", params={"at": _ts(t_mid)}).json()
    assert body["status"] == "MISSING_PARENT"

    # 父证明可见之后: OK
    body = c.get(
        "/verify/sha256:child", params={"at": _ts(now - timedelta(hours=12))}
    ).json()
    assert body["status"] == "OK"
    assert body["length"] == 2
