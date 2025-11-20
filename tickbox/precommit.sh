#!/usr/bin/env bash
set -ueo pipefail
exec schedtool -D -e tickbox --dir tickbox/precommit
