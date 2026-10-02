#!/usr/bin/env python3
"""Verify the pinned GTZAN mirror archive and extract without replacing raw files.

Python 3.12+; standard library only. This does not transform audio.
"""

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath

ARCHIVE_SHA256 = "b28cc067ff6199bd826f5d1a6931458586d64acae7f44b2e882b7a97af057531"
ARCHIVE_BYTES = 1226192050
SOURCE_REVISION = "d2146561ecc7df707d9e6b8318885fe6a39668a2"
SOURCE_URL = (
    "https://huggingface.co/datasets/marsyas/gtzan/resolve/"
    + SOURCE_REVISION + "/data/genres.tar.gz?download=true"
)


def digest_file(filename: Path) -> str:
    with filename.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def extract(archive: Path, destination: Path) -> dict:
    if archive.stat().st_size != ARCHIVE_BYTES:
        raise ValueError("Archive size differs from pinned source; resume/verify download.")
    actual_hash = digest_file(archive)
    if actual_hash != ARCHIVE_SHA256:
        raise ValueError("Archive SHA-256 mismatch; raw files have not been changed.")
    destination.mkdir(parents=True, exist_ok=True)
    root = destination.resolve()
    created = identical = 0
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        seen = set()
        # Validate every member before extraction, disallow links/devices/traversal.
        for member in members:
            relative = PurePosixPath(member.name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe archive path: {member.name}")
            if not (member.isdir() or member.isfile()):
                raise ValueError(f"Unsupported archive member: {member.name}")
            target = root.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(root):
                raise ValueError(f"Archive destination escapes root: {member.name}")
            # A pre-existing symlink, even one pointing inside root, is not raw data.
            if any(p.is_symlink() for p in [target, *target.parents] if p.is_relative_to(root)):
                raise ValueError(f"Symlink in destination: {member.name}")
            if member.isfile():
                if relative in seen:
                    raise ValueError(f"Duplicate archive path: {member.name}")
                seen.add(relative)
                if target.exists():
                    if not target.is_file():
                        raise ValueError(f"Existing non-file: {target}")
                    with bundle.extractfile(member) as source:
                        source_hash = hashlib.file_digest(source, "sha256").hexdigest()
                    if target.stat().st_size != member.size or digest_file(target) != source_hash:
                        raise ValueError(f"Refusing to replace differing raw file: {target}")
        for member in members:
            target = root.joinpath(*PurePosixPath(member.name).parts)
            if member.isfile() and target.exists():
                identical += 1
                continue
            bundle.extract(member, path=root, filter="data")
            created += int(member.isfile())
    return {
        "source_url": SOURCE_URL,
        "source_revision": SOURCE_REVISION,
        "archive_sha256": actual_hash,
        "archive_bytes": archive.stat().st_size,
        "files_extracted": created,
        "existing_identical_files": identical,
        "destination": destination.as_posix(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=Path("data/downloads/genres.tar.gz.part"))
    parser.add_argument("--destination", type=Path, default=Path("data/raw/gtzan"))
    args = parser.parse_args()
    result = extract(args.archive, args.destination)
    # A .part file becomes the completed archive only after successful verification.
    if args.archive.suffix == ".part":
        completed = args.archive.with_suffix("")
        if completed.exists():
            raise FileExistsError(f"Completed archive already exists; retained both: {completed}")
        args.archive.rename(completed)
        result["archive"] = completed.as_posix()
    else:
        result["archive"] = args.archive.as_posix()
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
