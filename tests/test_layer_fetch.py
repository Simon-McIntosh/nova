"""Fetch, digest refusal and clean skip for the pinned validator layers."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from nova.database import layer_fetch

_MANIFEST = Path(__file__).with_name("data-manifest.json")
_RECORD = json.loads(_MANIFEST.read_text())["efitpp_validator_layers"]
_REGISTRY = _RECORD["registry"]


def _layer(case: str, role: str) -> dict[str, object]:
    return next(
        layer for layer in _RECORD["cases"][case]["layers"] if layer["role"] == role
    )


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _flip_digest(digest: str) -> str:
    prefix, hex_digest = digest.split(":", 1)
    flipped = "1" if hex_digest[-1] != "1" else "0"
    return f"{prefix}:{hex_digest[:-1]}{flipped}"


def _stored_runner(payload: bytes):
    def runner(argv):
        output = Path(argv[list(argv).index("--output") + 1])
        output.write_bytes(payload)
        return subprocess.CompletedProcess(
            args=list(argv), returncode=0, stdout="", stderr=""
        )

    return runner


def _unreachable_runner(reason: str = "dial tcp: connection refused"):
    def runner(argv):
        return subprocess.CompletedProcess(
            args=list(argv), returncode=1, stdout="", stderr=reason
        )

    return runner


# --------------------------------------------------------------------------
# fetch every pinned layer, then verify its size and digest
# --------------------------------------------------------------------------
def test_every_pinned_layer_is_fetched_and_verified(tmp_path: Path) -> None:
    cache = tmp_path / "store"
    for case in _RECORD["cases"].values():
        for layer in case["layers"]:
            digest = str(layer["digest"])
            try:
                directory = layer_fetch.fetch_layer(
                    digest,
                    cache_directory=cache,
                    name=str(layer["name"]),
                    registry=_REGISTRY,
                )
            except layer_fetch.LayerRegistryUnreachable as error:
                pytest.skip(f"registry unreachable, pinned layer left unknown: {error}")

            blob = directory / str(layer["name"])
            payload = blob.read_bytes()
            assert len(payload) == int(layer["size"]), layer["name"]
            assert hashlib.sha256(payload).hexdigest() == digest.removeprefix(
                "sha256:"
            ), layer["name"]
            assert (directory / "manifest.json").is_file()


def test_a_published_layer_is_reused_without_contacting_the_registry(
    tmp_path: Path,
) -> None:
    payload = b"synthetic-analytic-layer-bytes"
    cache = tmp_path / "store"
    directory = layer_fetch.fetch_layer(
        _digest(payload),
        cache_directory=cache,
        name="analytic.nc",
        runner=_stored_runner(payload),
    )
    assert (directory / "analytic.nc").read_bytes() == payload

    def refuse(argv):
        raise AssertionError("a published layer must not re-contact the registry")

    assert (
        layer_fetch.fetch_layer(
            _digest(payload),
            cache_directory=cache,
            name="analytic.nc",
            runner=refuse,
        )
        == directory
    )


# --------------------------------------------------------------------------
# refusal when one manifest digest is altered
# --------------------------------------------------------------------------
def test_altered_manifest_digest_refuses_the_layer(tmp_path: Path) -> None:
    payload = b"the certified analytic layer"
    altered = _flip_digest(_digest(payload))
    assert altered != _digest(payload)
    cache = tmp_path / "store"

    with pytest.raises(layer_fetch.LayerDigestMismatch):
        layer_fetch.fetch_layer(
            altered,
            cache_directory=cache,
            name="analytic.nc",
            runner=_stored_runner(payload),
        )

    published = cache / "sha256" / altered.removeprefix("sha256:")
    assert not published.exists()


def test_the_recorded_digests_are_well_formed() -> None:
    for case in _RECORD["cases"].values():
        for layer in case["layers"]:
            digest = str(layer["digest"])
            assert digest.startswith("sha256:")
            hex_digest = digest.removeprefix("sha256:")
            assert len(hex_digest) == 64
            int(hex_digest, 16)


# --------------------------------------------------------------------------
# distinguish an unreachable registry, credential failure and unknown digest
# --------------------------------------------------------------------------
def test_unreachable_registry_is_classified_separately(tmp_path: Path) -> None:
    layer = _layer("iter_corsica_130506", "reference")
    with pytest.raises(layer_fetch.LayerRegistryUnreachable):
        layer_fetch.fetch_layer(
            str(layer["digest"]),
            cache_directory=tmp_path / "store",
            runner=_unreachable_runner(),
        )


@pytest.mark.parametrize(
    "message",
    [
        (
            'Error response from registry: HEAD "https://ghcr.io/v2/example/'
            'manifests/sha256:abc": unauthorized: authentication required'
        ),
        (
            "Error response from registry: denied: requested access to the "
            "resource is denied"
        ),
        (
            'Error: GET "https://ghcr.io/token": response status code 401: '
            "token has expired"
        ),
    ],
    ids=["authentication-required", "access-denied", "expired-token"],
)
def test_registry_credential_failures_are_not_skips(
    message: str, tmp_path: Path
) -> None:
    layer = _layer("iter_corsica_130506", "reference")

    with pytest.raises(layer_fetch.LayerRegistryCredentialError) as caught:
        layer_fetch.fetch_layer(
            str(layer["digest"]),
            cache_directory=tmp_path / "store",
            runner=_unreachable_runner(message),
        )

    assert not isinstance(caught.value, layer_fetch.LayerRegistryUnreachable)


def test_an_unknown_digest_is_a_fetch_error_not_a_skip(tmp_path: Path) -> None:
    def runner(argv):
        return subprocess.CompletedProcess(
            args=list(argv),
            returncode=1,
            stdout="",
            stderr="Error: not found: ghcr.io/...: unknown digest",
        )

    with pytest.raises(layer_fetch.LayerFetchError) as caught:
        layer_fetch.fetch_layer(
            "sha256:" + "0" * 64,
            cache_directory=tmp_path / "store",
            runner=runner,
        )
    assert not isinstance(caught.value, layer_fetch.LayerRegistryUnreachable)


def test_a_malformed_digest_is_refused_offline(tmp_path: Path) -> None:
    with pytest.raises(layer_fetch.LayerFetchError):
        layer_fetch.fetch_layer(
            "md5:cafe",
            cache_directory=tmp_path / "store",
            runner=_unreachable_runner(),
        )
