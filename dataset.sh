#!/bin/bash

set -euo pipefail

if [[ $# -lt 1 || ! "$1" =~ ^[0-9]+$ || "$1" -lt 1 ]]; then
    echo "Usage: $0 <nb> [output_file]" >&2
    exit 1
fi

NB="$1"
OUT="${2:-train.txt}"
DIR="dataset"
ZIP="dataset.zip"

if [[ ! -d "$DIR" ]]; then
    if [[ ! -f "$ZIP" ]]; then
        echo "Error: '$DIR' not found and '$ZIP' is missing." >&2
        exit 1
    fi
    echo "'$DIR' not found, unzipping $ZIP..."
    unzip -q "$ZIP"
    if [[ ! -d "$DIR" ]]; then
        echo "Error: unzip did not create '$DIR'." >&2
        exit 1
    fi
fi

TMP="$(mktemp)"
trap 'rm -f "$TMP"' EXIT

while IFS= read -r -d '' file; do
    total=$(wc -l < "$file")
    if (( total < NB )); then
        echo "Warning: $file has only $total lines (< $NB), taking all of them." >&2
    fi
    shuf -n "$NB" "$file" >> "$TMP"
done < <(find "$DIR" -type f -name '*.txt' -print0 | sort -z)

shuf "$TMP" > "$OUT"

echo "Done: $(wc -l < "$OUT") lines written to $OUT"