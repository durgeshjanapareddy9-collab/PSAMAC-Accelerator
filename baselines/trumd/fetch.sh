#!/usr/bin/env bash
# baselines/trumd/fetch.sh -- download the TruMD Verilog (not stored in our repo).
#
# TruMD (truncated Dadda multipliers, 2/4/6/8 dropped LSB columns) comes from
#   https://github.com/Hassan313/Approximate-Multiplier
# That repository has NO license file, so we may not copy its code into ours.
# Instead this script clones it at a fixed commit into baselines/trumd/upstream/
# (ignored by git). Every run gets exactly the same files.
#
# Usage:  bash baselines/trumd/fetch.sh        (normally via `make baselines`)

set -euo pipefail
URL=https://github.com/Hassan313/Approximate-Multiplier.git
COMMIT=ba73ab939e80501bc1ba83ce1ee4c8bca485ad2d     # pinned; checked 2026-10-22
DIR="$(cd "$(dirname "$0")" && pwd)/upstream"

if [ -d "$DIR/.git" ] && [ "$(git -C "$DIR" rev-parse HEAD)" = "$COMMIT" ]; then
    echo "TruMD already fetched at $COMMIT"
    exit 0
fi
rm -rf "$DIR"
git clone -q "$URL" "$DIR"
git -C "$DIR" checkout -q "$COMMIT"
echo "TruMD fetched into $DIR at $COMMIT"
