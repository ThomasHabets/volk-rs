#!/usr/bin/env bash
set -ueo pipefail
exit 0
cd "$TICKBOX_TEMPDIR/work"
export CARGO_TARGET_DIR="$TICKBOX_CWD/target/${TICKBOX_BRANCH}.deny"
exec cargo deny check
