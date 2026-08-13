"""Read-only artifact identity and path-containment helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
from pathlib import Path
from typing import Any, Mapping

from .errors import ControlPlaneError, ErrorRecord
from .schemas import AGENT_SCHEMA_VERSION, canonical_json, new_id


class IdentityStrength(str, Enum):
    CONTENT = "content"
    METADATA = "metadata"
    UNKNOWN = "unknown"


def canonical_path(path: str | Path, *, require_exists: bool = False) -> Path:
    """Normalize a path without creating it; resolve existing symlinks where possible."""
    candidate = Path(path).expanduser().resolve(strict=require_exists)
    return candidate


def path_within(path: str | Path, root: str | Path) -> bool:
    """Whether the resolved path is contained by the resolved root."""
    try:
        canonical_path(path).relative_to(canonical_path(root, require_exists=True))
    except (ValueError, FileNotFoundError):
        return False
    return True


def require_path_within(path: str | Path, root: str | Path, *, require_exists: bool = False) -> Path:
    """Return a canonical contained path or raise a structured control-plane error."""
    try:
        resolved_root = canonical_path(root, require_exists=True)
        resolved_path = canonical_path(path, require_exists=require_exists)
    except FileNotFoundError as error:
        raise ControlPlaneError(ErrorRecord("FILESYSTEM.NOT_FOUND", "Path does not exist", True,
                                            path=str(error.filename or path))) from error
    try:
        resolved_path.relative_to(resolved_root)
    except ValueError as error:
        raise ControlPlaneError(ErrorRecord("FILESYSTEM.OUTSIDE_ROOT", "Path escapes permitted root", False,
                                            path=str(resolved_path), details={"root": str(resolved_root)})) from error
    return resolved_path


@dataclass(frozen=True)
class IdentityRecord:
    """Identity evidence. Metadata fingerprints are explicitly weaker than content digests."""

    algorithm: str
    digest: str
    strength: IdentityStrength
    basis: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = AGENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.algorithm or not self.digest:
            raise ValueError("IdentityRecord requires algorithm and digest")
        if self.strength is IdentityStrength.METADATA and self.algorithm != "sha256-canonical-metadata-v1":
            raise ValueError("Metadata identity must use the canonical metadata algorithm")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "algorithm": self.algorithm,
                "digest": self.digest, "strength": self.strength.value, "basis": dict(self.basis)}


def metadata_identity(path: str | Path, *, root: str | Path | None = None) -> IdentityRecord:
    """Fast fingerprint; it reads stat metadata and never hashes file contents or writes cache state."""
    resolved = canonical_path(path, require_exists=True)
    stat = resolved.stat()
    location = str(resolved.relative_to(canonical_path(root, require_exists=True))) if root is not None else str(resolved)
    basis = {"path": location, "size_bytes": stat.st_size, "mtime_ns": stat.st_mtime_ns,
             "kind": "directory" if resolved.is_dir() else "file"}
    return IdentityRecord("sha256-canonical-metadata-v1", hashlib.sha256(canonical_json(basis).encode("utf-8")).hexdigest(),
                          IdentityStrength.METADATA, basis)


def content_identity(path: str | Path, *, chunk_size: int = 1024 * 1024) -> IdentityRecord:
    """Opt-in full file digest. Callers must choose this cost; it is never implicit."""
    resolved = canonical_path(path, require_exists=True)
    if not resolved.is_file():
        raise ControlPlaneError(ErrorRecord("FILESYSTEM.INVALID_PATH", "Content digest requires a regular file", False,
                                            path=str(resolved)))
    digest = hashlib.sha256()
    with resolved.open("rb") as stream:
        for block in iter(lambda: stream.read(chunk_size), b""):
            digest.update(block)
    return IdentityRecord("sha256", digest.hexdigest(), IdentityStrength.CONTENT,
                          {"size_bytes": resolved.stat().st_size, "chunk_size": chunk_size})


@dataclass(frozen=True)
class ArtifactRecord:
    """An immutable published product record; location and identity are separate."""

    artifact_type: str
    artifact_schema: str
    producer_stage: str
    identity: IdentityRecord
    location: str
    size_bytes: int
    artifact_id: str = field(default_factory=lambda: new_id("art"))
    compatibility: Mapping[str, Any] = field(default_factory=dict)
    schema_version: str = AGENT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not all((self.artifact_type, self.artifact_schema, self.producer_stage, self.location)):
            raise ValueError("ArtifactRecord requires type, schema, producer stage, and location")
        if self.size_bytes < 0:
            raise ValueError("ArtifactRecord.size_bytes must be nonnegative")

    def to_dict(self) -> dict[str, Any]:
        return {"schema_version": self.schema_version, "artifact_id": self.artifact_id,
                "artifact_type": self.artifact_type, "artifact_schema": self.artifact_schema,
                "producer_stage": self.producer_stage, "identity": self.identity.to_dict(),
                "location": self.location, "size_bytes": self.size_bytes,
                "compatibility": dict(self.compatibility)}

    @classmethod
    def from_file(cls, path: str | Path, *, artifact_type: str, artifact_schema: str,
                  producer_stage: str, root: str | Path | None = None,
                  compatibility: Mapping[str, Any] | None = None) -> "ArtifactRecord":
        """Adapt one existing file to v1 without workflow inspection or writes."""
        resolved = canonical_path(path, require_exists=True)
        if not resolved.is_file():
            raise ControlPlaneError(ErrorRecord("ARTIFACT.INVALID", "Artifact must be a regular file", False,
                                                path=str(resolved)))
        location = str(resolved.relative_to(canonical_path(root, require_exists=True))) if root is not None else str(resolved)
        return cls(artifact_type=artifact_type, artifact_schema=artifact_schema, producer_stage=producer_stage,
                   identity=metadata_identity(resolved, root=root), location=location,
                   size_bytes=resolved.stat().st_size, compatibility=compatibility or {})
