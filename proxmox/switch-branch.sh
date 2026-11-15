#!/usr/bin/env bash
# Switch an existing EpexPredictor container to another branch of the repository, e.g. to test a pull request,
# and optionally switch on Timestral. Run it in the shell of the container:
#
#   bash -c "$(curl -fsSL https://raw.githubusercontent.com/jeremiahpslewis/EpexPredictor/<branch>/proxmox/switch-branch.sh)" _ <branch> [--timestral]
#
# Running it again updates to the latest commit of the branch. "main" switches back.

set -euo pipefail

BRANCH="${1:-}"
TIMESTRAL="${2:-}"
DIR="${EPEX_DIR:-/opt/epexpredictor}"
MODEL="${EPEX_TIMESTRAL_MODEL:-mistralai/Timestral-123M}"

if [[ -z "$BRANCH" ]]; then
  echo "usage: switch-branch.sh <branch> [--timestral]" >&2
  exit 1
fi
if [[ -n "$TIMESTRAL" && "$TIMESTRAL" != "--timestral" ]]; then
  echo "unknown option: $TIMESTRAL" >&2
  exit 1
fi
if [[ ! -d "$DIR/.git" ]]; then
  echo "no EpexPredictor installation found in $DIR" >&2
  exit 1
fi
UV="$(command -v uv || true)"
if [[ -z "$UV" ]]; then
  echo "uv not found" >&2
  exit 1
fi

cd "$DIR"
echo "Fetching $BRANCH"
git fetch origin "+refs/heads/$BRANCH:refs/remotes/origin/$BRANCH"
git checkout -q -B "$BRANCH" "origin/$BRANCH"
git branch -q --set-upstream-to="origin/$BRANCH" "$BRANCH"
echo "Now at $(git log -1 --format='%h %s')"

echo "Installing dependencies"
"$UV" pip install -q -r requirements.txt

touch .env
if [[ "$TIMESTRAL" == "--timestral" ]]; then
  if [[ ! -f requirements-timestral.txt ]]; then
    echo "this branch has no Timestral support" >&2
    exit 1
  fi
  free_mb=$(df -Pm "$DIR" | awk 'NR == 2 {print $4}')
  if (( free_mb < 2500 )); then
    echo "only ${free_mb} MB free disk space, PyTorch needs about 2 GB - grow the disk first (on the host: pct resize <id> rootfs +4G)" >&2
    exit 1
  fi
  echo "Installing Timestral with the CPU build of PyTorch"
  "$UV" pip install -q --torch-backend cpu -r requirements-timestral.txt
  if grep -q '^EPEXPREDICTOR_TIMESTRAL_MODEL=' .env; then
    sed -i "s|^EPEXPREDICTOR_TIMESTRAL_MODEL=.*|EPEXPREDICTOR_TIMESTRAL_MODEL=$MODEL|" .env
  else
    echo "EPEXPREDICTOR_TIMESTRAL_MODEL=$MODEL" >>.env
  fi
  mem_mb=$(awk '/^MemTotal:/ {print int($2 / 1024)}' /proc/meminfo)
  if (( mem_mb < 5500 )); then
    echo "Note: ${mem_mb} MB memory. With Timestral the service peaked at 2.7 GB with 8 regions, more with an ENTSO-E key;"
    echo "      consider 6 GB (on the host: pct set <id> --memory 6144)."
  fi
else
  sed -i 's|^EPEXPREDICTOR_TIMESTRAL_MODEL=|#EPEXPREDICTOR_TIMESTRAL_MODEL=|' .env
fi

if command -v systemctl >/dev/null && systemctl list-unit-files epexpredictor.service >/dev/null 2>&1; then
  systemctl restart epexpredictor
  echo "Restarted. The first start on a new branch may download data for several minutes; follow it with:"
  echo "  journalctl -u epexpredictor -f"
else
  echo "No epexpredictor service found - start the app yourself"
fi
