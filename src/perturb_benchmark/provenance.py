"""Data provenance and checksum utilities."""

import hashlib
import json
import os
import urllib.request
from datetime import UTC, datetime
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Return the SHA256 digest of a non-empty file."""

    if not path.is_file():
        raise FileNotFoundError(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_with_provenance(
    url: str,
    destination: Path,
    provenance_path: Path,
) -> dict[str, object]:
    """Stream a source artifact and atomically record its provenance."""

    if not url.startswith(("https://", "http://")):
        raise ValueError("download URL must use HTTP or HTTPS")
    destination.parent.mkdir(parents=True, exist_ok=True)
    provenance_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")

    request = urllib.request.Request(url, headers={"User-Agent": "perturb-benchmark/0.1"})
    try:
        with (
            urllib.request.urlopen(request, timeout=120) as response,
            temporary.open("wb") as output,
        ):
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        size = temporary.stat().st_size
        if size == 0:
            raise ValueError("downloaded artifact is empty")
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)

    record: dict[str, object] = {
        "url": url,
        "downloaded_at": datetime.now(UTC).isoformat(),
        "path": destination.as_posix(),
        "bytes": destination.stat().st_size,
        "sha256": sha256_file(destination),
    }
    provenance_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    return record
