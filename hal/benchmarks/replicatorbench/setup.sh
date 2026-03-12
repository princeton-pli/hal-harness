#!/bin/bash

mkdir -p /workspace

if [ ! -f /workspace/run_agent.py ]; then
  for cand in \
    /root/environment/workspace/run_agent.py \
    /root/environment/workspace/hal/run_agent.py \
    /root/environment/workspace/hal-harness/run_agent.py
  do
    if [ -f "$cand" ]; then
      ln -sf "$cand" /workspace/run_agent.py
      break
    fi
  done
fi

set -euo pipefail

if [[ "${STUB_WEAVE:-0}" == "1" ]]; then
  cat > /workspace/weave.py <<'PY'
from contextlib import contextmanager

def init(*args, **kwargs):
    return None

@contextmanager
def attributes(*args, **kwargs):
    yield
PY
  echo "[replicatorbench] using weave stub at /workspace/weave.py"
else
  rm -f /workspace/weave.py
  echo "[replicatorbench] weave stub disabled; using real import behavior"
fi

BENCH_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TASKS_JSON="${BENCH_DIR}/tasks.json"

if [[ ! -f "${TASKS_JSON}" ]]; then
  echo "[replicatorbench] ERROR: tasks.json not found at: ${TASKS_JSON}"
  exit 1
fi

ROOT="/workspace"
CAPS_DIR="${ROOT}/capsules"
mkdir -p "${CAPS_DIR}"

TEMPLATE_SRC="${BENCH_DIR}/templates"
TEMPLATE_DST="${ROOT}/replicatorbench_templates"
rm -rf "${TEMPLATE_DST}"
mkdir -p "${TEMPLATE_DST}"
if [[ -d "${TEMPLATE_SRC}" ]]; then
  cp -R "${TEMPLATE_SRC}/." "${TEMPLATE_DST}/"
  echo "[replicatorbench] templates copied to ${TEMPLATE_DST}"
else
  echo "[replicatorbench] WARNING: templates dir not found at ${TEMPLATE_SRC} (skipping copy)"
fi

KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"

apt-get update -y
apt-get install -y --no-install-recommends \
  ca-certificates \
  curl \
  git \
  git-lfs \
  python3 \
  python3-pip \
  unzip \
  tar

git lfs install --system

python3 -m pip install --upgrade pip
python3 -m pip install --upgrade gdown

download_gdrive_folder() {
  local url="$1"
  local out_dir="$2"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  gdown --no-cookies --folder "${url}" -O "${out_dir}"
}

download_gdrive_file() {
  local url_or_id="$1"
  local out_path="$2"

  local id=""
  if [[ "${url_or_id}" =~ ^[0-9A-Za-z_-]{20,}$ ]]; then
    id="${url_or_id}"
  else
    id="$(python3 - <<'PY' "${url_or_id}"
import re,sys
u=sys.argv[1]
m = re.search(r"/file/d/([^/]+)", u) or re.search(r"[?&]id=([^&]+)", u)
print(m.group(1) if m else "")
PY
)"
  fi

  if [[ -z "${id}" ]]; then
    echo "[replicatorbench] ERROR: cannot parse Google Drive file id from: ${url_or_id}"
    return 1
  fi

  local base="https://drive.google.com/uc?export=download&id=${id}"
  local cookie="/tmp/gdrive_cookie_$$.txt"
  local tmp="${out_path}.tmp"

  rm -f "${cookie}" "${tmp}"

  curl -L -sS -c "${cookie}" -o "${tmp}" "${base}"

  if head -c 2 "${tmp}" | grep -q "PK"; then
    mv -f "${tmp}" "${out_path}"
    rm -f "${cookie}"
    return 0
  fi

  # virus-scan warning flow
  if grep -q "Google Drive can't scan this file for viruses" "${tmp}"; then
    local confirm uuid
    confirm="$(sed -n 's/.*name="confirm" value="\([^"]*\)".*/\1/p' "${tmp}" | head -n 1)"
    uuid="$(sed -n 's/.*name="uuid" value="\([^"]*\)".*/\1/p' "${tmp}" | head -n 1)"

    if [[ -z "${confirm}" ]]; then
      echo "[replicatorbench] ERROR: could not parse confirm token from virus-scan warning page."
      rm -f "${tmp}" "${cookie}"
      return 1
    fi

    echo "[replicatorbench]   virus-scan warning detected; downloading from drive.usercontent.google.com..."
    local dl="https://drive.usercontent.google.com/download?export=download&id=${id}&confirm=${confirm}"
    if [[ -n "${uuid}" ]]; then
      dl="${dl}&uuid=${uuid}"
    fi

    curl -L --fail -b "${cookie}" --progress-bar -o "${out_path}" "${dl}"
    rm -f "${tmp}" "${cookie}"
    return 0
  fi

  echo "[replicatorbench] ERROR: Google Drive returned HTML instead of a file, and it was not the virus-scan warning page."
  echo "[replicatorbench] First lines:"
  head -n 20 "${tmp}" | sed 's/^/[replicatorbench]   /'
  rm -f "${tmp}" "${cookie}"
  return 1
}

extract_zip_capsule() {
  local zip_path="$1"
  local out_dir="$2"

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"

  local tmp="/tmp/capsule_unzip_$$"
  rm -rf "${tmp}"
  mkdir -p "${tmp}"

  unzip -q "${zip_path}" -d "${tmp}"

  local top_count
  top_count="$(find "${tmp}" -mindepth 1 -maxdepth 1 | wc -l | tr -d ' ')"

  if [[ "${top_count}" == "1" ]] && [[ -d "$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -n 1)" ]]; then
    local topdir
    topdir="$(find "${tmp}" -mindepth 1 -maxdepth 1 -type d | head -n 1)"
    shopt -s dotglob
    cp -R "${topdir}"/* "${out_dir}"/
    shopt -u dotglob
  else
    shopt -s dotglob
    cp -R "${tmp}"/* "${out_dir}"/
    shopt -u dotglob
  fi

  rm -rf "${tmp}"

  # Fix double extension artifacts
  if compgen -G "${out_dir}/*.pdf.pdf" > /dev/null; then
    for f in "${out_dir}"/*.pdf.pdf; do
      mv -f "${f}" "${f%.pdf}"
    done
  fi
}

download_tarball_capsule() {
  local url="$1"
  local sha="$2"
  local out_dir="$3"
  local tgz_path="$4"

  echo "[replicatorbench] downloading tarball: ${url}"
  curl -L --fail -o "${tgz_path}" "${url}"

  if [[ -n "${sha}" && "${sha}" != "null" ]]; then
    echo "[replicatorbench] verifying sha256..."
    echo "${sha}  ${tgz_path}" | sha256sum -c -
  fi

  rm -rf "${out_dir}"
  mkdir -p "${out_dir}"
  tar -xzf "${tgz_path}" -C "${out_dir}" --strip-components=1
}

download_github_subdir_capsule() {
  local repo_url="$1"
  local ref="$2"
  local subdir="$3"
  local sha="$4"      
  local out_dir="$5"

  local tmp="/tmp/github_subdir_capsule_$$"
  rm -rf "${tmp}" "${out_dir}"
  mkdir -p "${out_dir}"

  echo "[replicatorbench] cloning github_subdir capsule: ${repo_url} @ ${ref}"
  git clone --depth 1 "${repo_url}" "${tmp}"

  git -C "${tmp}" fetch --depth 1 origin "${ref}" || true
  git -C "${tmp}" checkout "${ref}"

  if ! command -v git-lfs >/dev/null 2>&1; then
    echo "[replicatorbench] ERROR: git-lfs not installed or not in PATH"
    rm -rf "${tmp}"
    return 1
  fi

  git -C "${tmp}" lfs install --local
  git -C "${tmp}" lfs pull

  if [[ ! -d "${tmp}/${subdir}" ]]; then
    echo "[replicatorbench] ERROR: subdir not found after clone: ${subdir}"
    find "${tmp}" -maxdepth 3 | sed 's/^/[replicatorbench]   /'
    rm -rf "${tmp}"
    return 1
  fi

  shopt -s dotglob
  cp -R "${tmp}/${subdir}/"* "${out_dir}/"
  shopt -u dotglob

  rm -rf "${tmp}"
}

prepare_shared_study_dir() {
  local capsule_id="$1"
  local work_dir="${ROOT}/${capsule_id}"

  rm -rf "${work_dir}"
  mkdir -p "${work_dir}"

  echo "${CAPS_DIR}/${capsule_id}" > "${work_dir}/CAPSULE_PATH.txt"
}

make_capsule_readonly() {
  local capsule_dir="$1"

  find "${capsule_dir}" -type d -exec chmod 755 {} \; || true

  find "${capsule_dir}" -type f -exec chmod 444 {} \; || true
}

declare -A DOWNLOADED=()
declare -A PREPARED_STUDYDIR=()

CURRENT_TASK_ID="$(python3 - <<'PY'
import json
p = "/workspace/input.json"
try:
    with open(p, "r") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        for k in obj.keys():
            print(k)
            break
    else:
        print("")
except Exception:
    print("")
PY
)"

while IFS=$'\t' read -r task_id capsule_id capsule_type capsule_url capsule_ref capsule_subdir capsule_sha256; do
  if [[ -z "${task_id}" ]]; then
    continue
  fi

  if [[ -n "${CURRENT_TASK_ID}" && "${task_id}" != "${CURRENT_TASK_ID}" ]]; then
    continue
  fi

  if [[ -z "${capsule_id}" ]]; then
    capsule_id="${task_id}"
  fi

  if [[ -z "${PREPARED_STUDYDIR[${capsule_id}]+x}" ]]; then
    prepare_shared_study_dir "${capsule_id}"
    PREPARED_STUDYDIR["${capsule_id}"]=1
    echo "[replicatorbench] prepared shared study dir: ${ROOT}/${capsule_id}"
  fi

  # Download capsule only once per capsule_id
  if [[ -z "${DOWNLOADED[${capsule_id}]+x}" ]]; then
    if [[ -z "${capsule_url}" ]]; then
      echo "[replicatorbench] ${task_id}: no capsule_url (skipping download)"
      DOWNLOADED["${capsule_id}"]=1
      continue
    fi

    out_dir="${CAPS_DIR}/${capsule_id}"

    if [[ "${capsule_type}" == "gdrive_zip" ]]; then
      echo "[replicatorbench] ${capsule_id}: downloading ZIP capsule..."
      zip_path="${CAPS_DIR}/${capsule_id}.zip"
      download_gdrive_file "${capsule_url}" "${zip_path}"
      extract_zip_capsule "${zip_path}" "${out_dir}"
      if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
        rm -f "${zip_path}"
      fi

    elif [[ "${capsule_type}" == "gdrive_folder" ]]; then
      echo "[replicatorbench] ${capsule_id}: downloading folder capsule..."
      download_gdrive_folder "${capsule_url}" "${out_dir}"

    elif [[ "${capsule_type}" == "github_subdir" ]]; then
      echo "[replicatorbench] ${capsule_id}: downloading github_subdir capsule..."
      download_github_subdir_capsule \
        "${capsule_url}" \
        "${capsule_ref}" \
        "${capsule_subdir}" \
        "${capsule_sha256}" \
        "${out_dir}"


    else
      echo "[replicatorbench] ${capsule_id}: downloading tarball capsule..."
      tgz_path="${CAPS_DIR}/${capsule_id}.tar.gz"
      download_tarball_capsule "${capsule_url}" "${capsule_sha256}" "${out_dir}" "${tgz_path}"
      if [[ "${KEEP_ARCHIVES}" != "1" ]]; then
        rm -f "${tgz_path}"
      fi
    fi

    make_capsule_readonly "${out_dir}"
    DOWNLOADED["${capsule_id}"]=1

    echo "[replicatorbench] ${capsule_id}: capsule ready at ${out_dir}"
  fi

done < <(python3 - "${TASKS_JSON}" <<'PY'
import json, sys

tasks_path = sys.argv[1]
with open(tasks_path, "r") as f:
    obj = json.load(f)

tasks = obj["tasks"] if isinstance(obj, dict) and "tasks" in obj else obj
if not isinstance(tasks, list):
    raise SystemExit("tasks.json must be a dict with key 'tasks' holding a list.")

STAGE_SUFFIXES = ["_extract", "_web_search", "_design", "_execute", "_interpret"]

def derive_capsule_id(task_id: str) -> str:
    for s in STAGE_SUFFIXES:
        if task_id.endswith(s):
            return task_id[: -len(s)]
    return task_id

for t in tasks:
    task_id = str(t.get("task_id", "")).strip()
    if not task_id:
        continue
    capsule_id = str(t.get("capsule_id", "")).strip() or derive_capsule_id(task_id)
    ctype = (t.get("capsule_type") or "").strip()
    url = (t.get("capsule_url") or "").strip()
    ref = (t.get("capsule_ref") or "").strip()
    subdir = (t.get("capsule_subdir") or "").strip()
    sha = t.get("capsule_sha256", None)
    sha_s = "" if sha is None else str(sha).strip()
    print(f"{task_id}\t{capsule_id}\t{ctype}\t{url}\t{ref}\t{subdir}\t{sha_s}")
PY
)