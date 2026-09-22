"""端到端测试: 通过 FastAPI TestClient + 内存 SQLite 覆盖全部校验分支。"""
from __future__ import annotations

import sys
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
