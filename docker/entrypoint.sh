#!/bin/sh
# Container entrypoint: decide where the documents folder is, then run the command (the server).
#
# Uploads and Google Drive syncs write into the documents folder, so it has to be on a volume, or a
# rebuilt container loses them (and the next index update drops them from the index):
# - RAG_DOCS, if set, is used as it is;
# - a folder mounted at /app/docs (`-v /path/to/docs:/app/docs`) is used as it is;
# - otherwise /data/docs on the data volume, filled with the sample documents on first start.
set -eu
if [ -z "${RAG_DOCS:-}" ]; then
  if awk '$5 == "/app/docs" { found = 1 } END { exit !found }' /proc/self/mountinfo; then
    RAG_DOCS=/app/docs
  else
    RAG_DOCS=/data/docs
    if [ ! -d /data/docs ]; then
      rm -rf /data/.docs-seed
      cp -R /app/docs /data/.docs-seed
      mv /data/.docs-seed /data/docs
    fi
  fi
  export RAG_DOCS
fi
exec "$@"
