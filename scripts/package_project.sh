#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_NAME="$(basename "${PROJECT_ROOT}")"

MAX_FILE_SIZE_MB="${MAX_FILE_SIZE_MB:-200}"
OUTPUT_DIR="${OUTPUT_DIR:-${PROJECT_ROOT}/release}"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
STAGING_DIR="${OUTPUT_DIR}/${PROJECT_NAME}-${TIMESTAMP}"
ARCHIVE_PATH="${OUTPUT_DIR}/${PROJECT_NAME}-${TIMESTAMP}.zip"
MANIFEST_PATH="${OUTPUT_DIR}/${PROJECT_NAME}-${TIMESTAMP}.manifest.txt"

mkdir -p "${OUTPUT_DIR}"
rm -rf "${STAGING_DIR}"
mkdir -p "${STAGING_DIR}"

export PROJECT_ROOT STAGING_DIR MAX_FILE_SIZE_MB

python3 - <<'PY'
import os
import shutil
from pathlib import Path

project_root = Path(os.environ["PROJECT_ROOT"]).resolve()
staging_dir = Path(os.environ["STAGING_DIR"]).resolve()
max_bytes = int(os.environ["MAX_FILE_SIZE_MB"]) * 1024 * 1024

skip_dir_names = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "venv",
    "node_modules",
}

skip_dir_paths = {
    project_root / "release",
}

skip_file_suffixes = {
    ".pyc",
    ".pyo",
    ".swp",
    ".swo",
}

included = []
skipped_large = []
skipped_other = []

for root, dirs, files in os.walk(project_root):
    root_path = Path(root)
    rel_root = root_path.relative_to(project_root)

    filtered_dirs = []
    for dirname in dirs:
        src_dir = root_path / dirname
        if dirname in skip_dir_names or src_dir in skip_dir_paths:
            skipped_other.append((src_dir.relative_to(project_root).as_posix() + "/", "excluded directory"))
            continue
        filtered_dirs.append(dirname)
    dirs[:] = filtered_dirs

    for filename in files:
        src_path = root_path / filename
        rel_path = src_path.relative_to(project_root)

        if src_path.suffix in skip_file_suffixes:
            skipped_other.append((rel_path.as_posix(), "excluded cache file"))
            continue

        size = src_path.stat().st_size
        if size > max_bytes:
            skipped_large.append((rel_path.as_posix(), size))
            continue

        dst_path = staging_dir / rel_path
        dst_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_path, dst_path)
        included.append((rel_path.as_posix(), size))

included.sort()
skipped_large.sort()
skipped_other.sort()

manifest_path = staging_dir.parent / (staging_dir.name + ".manifest.txt")
with manifest_path.open("w", encoding="utf-8") as fh:
    fh.write(f"project_root: {project_root}\n")
    fh.write(f"max_file_size_mb: {os.environ['MAX_FILE_SIZE_MB']}\n\n")

    fh.write("[included files]\n")
    for rel_path, size in included:
        fh.write(f"{size:>12}  {rel_path}\n")

    fh.write("\n[skipped large files]\n")
    for rel_path, size in skipped_large:
        fh.write(f"{size:>12}  {rel_path}\n")

    fh.write("\n[skipped other paths]\n")
    for rel_path, reason in skipped_other:
        fh.write(f"{reason:<20}  {rel_path}\n")
PY

(cd "${OUTPUT_DIR}" && zip -r "$(basename "${ARCHIVE_PATH}")" "$(basename "${STAGING_DIR}")")
rm -rf "${STAGING_DIR}"

INCLUDED_COUNT="$(awk 'BEGIN{c=0;section=0} /^\[included files\]/{section=1;next} /^\[/{section=0} section && NF{c++} END{print c}' "${MANIFEST_PATH}")"
SKIPPED_LARGE_COUNT="$(awk 'BEGIN{c=0;section=0} /^\[skipped large files\]/{section=1;next} /^\[/{section=0} section && NF{c++} END{print c}' "${MANIFEST_PATH}")"

echo "打包完成:"
echo "  压缩包: ${ARCHIVE_PATH}"
echo "  清单:   ${MANIFEST_PATH}"
echo "  已打包文件数: ${INCLUDED_COUNT}"
echo "  跳过的大文件数: ${SKIPPED_LARGE_COUNT}"
echo
echo "可选参数:"
echo "  MAX_FILE_SIZE_MB=500 bash scripts/package_project.sh"
echo "  OUTPUT_DIR=/tmp/package-out bash scripts/package_project.sh"
