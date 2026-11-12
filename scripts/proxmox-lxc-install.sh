#!/usr/bin/env bash
#
# Creates an LXC container on a Proxmox VE host and installs EpexPredictor in it as a systemd service.
#
# Run as root on the Proxmox host:
#
#     bash proxmox-lxc-install.sh
#
# Every setting below can be overridden with an environment variable, for example:
#
#     CTID=120 STORAGE=local-zfs ENTSOE_API_KEY=... bash proxmox-lxc-install.sh
#
# Inside the container:
#     /opt/epexpredictor/app                the source code (git checkout)
#     /opt/epexpredictor/venv               the Python environment
#     /var/lib/epexpredictor                cached weather, price and gas data
#     /etc/epexpredictor.env                settings of the service (ENTSO-E API key, ...)
#     /usr/local/bin/epexpredictor-update   pulls the latest code and restarts the service (optional argument:
#                                           a branch to switch to)

set -euo pipefail

CTID="${CTID:-}"                          # container id (default: next free one)
CT_HOSTNAME="${CT_HOSTNAME:-epexpredictor}"
STORAGE="${STORAGE:-}"                    # storage for the container disk (default: first active one that allows containers)
TEMPLATE_STORAGE="${TEMPLATE_STORAGE:-}"  # storage for the OS template (default: first active one that allows templates)
# OS template: a file name as listed by "pveam available", or the start of one (e.g. "ubuntu-24.04-standard") to
# pick its newest version. EpexPredictor needs Python 3.12 or newer, so the default is Debian 13, or Ubuntu 24.04
# if that is not on offer.
TEMPLATE="${TEMPLATE:-}"
DISK_GB="${DISK_GB:-8}"
CORES="${CORES:-2}"
MEMORY_MB="${MEMORY_MB:-4096}"            # the service peaks at about 2 GB with the 8 regions available without ENTSO-E key
SWAP_MB="${SWAP_MB:-512}"
BRIDGE="${BRIDGE:-vmbr0}"
VLAN="${VLAN:-}"                          # optional VLAN tag
IP="${IP:-dhcp}"                          # "dhcp" or a static address in CIDR notation, e.g. 192.168.1.50/24
GATEWAY="${GATEWAY:-}"                    # only for a static address
PORT="${PORT:-8000}"                      # port the API listens on
REPO_URL="${REPO_URL:-https://github.com/jeremiahpslewis/EpexPredictor.git}"
REPO_REF="${REPO_REF:-main}"              # branch to install
ENTSOE_API_KEY="${ENTSOE_API_KEY:-}"      # optional, see README ("Self Hosting")
USE_NORDPOOL="${USE_NORDPOOL:-}"          # optional, "true" to enable (see README)
YES="${YES:-}"                            # set to 1 to skip the confirmation

msg() { echo -e "\n==> $*"; }
die() { echo "ERROR: $*" >&2; exit 1; }

WORKDIR=""
CREATED=""
on_exit() {
    local status=$?
    if [[ -n $WORKDIR ]]; then
        rm -rf "$WORKDIR"
    fi
    if [[ $status -ne 0 && -n $CREATED ]]; then
        echo >&2
        echo "The installation did not finish. Container $CTID was left as it is, so you can look into it" >&2
        echo "(pct enter $CTID). To remove it: pct stop $CTID; pct destroy $CTID" >&2
    fi
}
trap on_exit EXIT

# ---------------------------------------------------------------------------------------------- checks

[[ $EUID -eq 0 ]] || die "this script has to run as root on the Proxmox host"
for cmd in pct pveam pvesm pvesh; do
    command -v "$cmd" >/dev/null || die "'$cmd' not found - this script has to run on a Proxmox VE host"
done
if [[ ! $PORT =~ ^[1-9][0-9]*$ ]] || (( PORT > 65535 )); then
    die "PORT has to be a number between 1 and 65535"
fi
[[ -e /sys/class/net/$BRIDGE ]] || die "network bridge '$BRIDGE' does not exist on this host - set BRIDGE"

# first active storage that accepts the given content type
pick_storage() {
    pvesm status --content "$1" | awk 'NR > 1 && $3 == "active" && !found {print $1; found = 1}'
}

# newest template on offer whose file name starts with the given text
find_template() {
    local available
    available="$(pveam available --section system)"
    awk -v prefix="$1" 'index($2, prefix) == 1 && $2 ~ /_amd64\.tar\.(zst|xz|gz)$/ {print $2}' <<<"$available" | sort -V | tail -n 1
}

if [[ -z $CTID ]]; then
    CTID="$(pvesh get /cluster/nextid)"
fi
[[ $CTID =~ ^[0-9]+$ ]] || die "CTID has to be a number"
if pct status "$CTID" >/dev/null 2>&1 || { command -v qm >/dev/null && qm status "$CTID" >/dev/null 2>&1; }; then
    die "the id $CTID is already in use"
fi

if [[ -z $STORAGE ]]; then
    STORAGE="$(pick_storage rootdir)"
fi
[[ -n $STORAGE ]] || die "no active storage for containers found - set STORAGE"
if [[ -z $TEMPLATE_STORAGE ]]; then
    TEMPLATE_STORAGE="$(pick_storage vztmpl)"
fi
[[ -n $TEMPLATE_STORAGE ]] || die "no active storage for container templates found - set TEMPLATE_STORAGE"

if [[ ! $TEMPLATE =~ \.tar\.(zst|xz|gz)$ ]]; then
    msg "Looking for an OS template"
    pveam update >/dev/null || echo "WARNING: could not refresh the list of templates, using the one from the last refresh" >&2
    if [[ -n $TEMPLATE ]]; then
        wanted="$TEMPLATE"
        TEMPLATE="$(find_template "$wanted")"
        [[ -n $TEMPLATE ]] || die "no template starting with '$wanted' on offer (see: pveam available --section system)"
    else
        TEMPLATE="$(find_template debian-13-standard)"
        if [[ -z $TEMPLATE ]]; then
            TEMPLATE="$(find_template ubuntu-24.04-standard)"
        fi
        [[ -n $TEMPLATE ]] || die "neither a Debian 13 nor an Ubuntu 24.04 template is on offer - set TEMPLATE to one that has Python 3.12 or newer"
    fi
fi

NET0="name=eth0,bridge=${BRIDGE},ip=${IP}"
if [[ $IP != dhcp && -n $GATEWAY ]]; then
    NET0+=",gw=${GATEWAY}"
fi
if [[ -n $VLAN ]]; then
    NET0+=",tag=${VLAN}"
fi

cat <<EOF

About to create this container:
    id / hostname    $CTID / $CT_HOSTNAME
    template         $TEMPLATE_STORAGE:vztmpl/$TEMPLATE
    disk             ${DISK_GB} GB on $STORAGE
    cores / memory   $CORES / ${MEMORY_MB} MB (+ ${SWAP_MB} MB swap)
    network          $NET0
    EpexPredictor    $REPO_URL ($REPO_REF), port $PORT
    ENTSO-E API key  $([[ -n $ENTSOE_API_KEY ]] && echo "set" || echo "not set (fewer regions, see README)")

EOF
if [[ -z $YES ]]; then
    { exec 3</dev/tty; } 2>/dev/null || die "no terminal to ask for confirmation - set YES=1"
    read -r -p "Continue? [y/N] " answer <&3 || answer=""
    exec 3<&-
    [[ $answer =~ ^[yY] ]] || die "aborted"
fi

# ---------------------------------------------------------------------------------------------- container

downloaded="$(pveam list "$TEMPLATE_STORAGE")"
if ! grep -q -F -- "/${TEMPLATE}" <<<"$downloaded"; then
    msg "Downloading $TEMPLATE"
    pveam download "$TEMPLATE_STORAGE" "$TEMPLATE"
fi

msg "Creating container $CTID"
if ! pct create "$CTID" "${TEMPLATE_STORAGE}:vztmpl/${TEMPLATE}" \
    --hostname "$CT_HOSTNAME" \
    --cores "$CORES" \
    --memory "$MEMORY_MB" \
    --swap "$SWAP_MB" \
    --rootfs "${STORAGE}:${DISK_GB}" \
    --net0 "$NET0" \
    --unprivileged 1 \
    --features nesting=1 \
    --onboot 1 \
    --timezone host \
    --description "EpexPredictor - EPEX day-ahead price forecast API (port ${PORT})"
then
    die "the container could not be created, see the message above. If it says that the version of the OS is not" \
        "supported, this Proxmox VE is older than the template: update it, or use TEMPLATE=ubuntu-24.04-standard"
fi
CREATED=1

pct start "$CTID"

msg "Waiting for the network inside the container"
online=""
for _ in $(seq 1 30); do
    if pct exec "$CTID" -- getent hosts github.com >/dev/null 2>&1; then
        online=1
        break
    fi
    sleep 2
done
[[ -n $online ]] || die "the container cannot resolve host names - check its network settings (bridge, VLAN, IP, DNS)"

# ---------------------------------------------------------------------------------------------- installation

WORKDIR="$(mktemp -d)"

# Settings of the service. Written to a file, so that the API key never appears on a command line.
(
    umask 077
    {
        echo "EPEXPREDICTOR_DATADIR=/var/lib/epexpredictor"
        if [[ -n $ENTSOE_API_KEY ]]; then echo "EPEXPREDICTOR_ENTSOE_API_KEY=${ENTSOE_API_KEY}"; fi
        if [[ -n $USE_NORDPOOL ]]; then echo "EPEXPREDICTOR_USE_NORDPOOL=${USE_NORDPOOL}"; fi
    } > "$WORKDIR/epexpredictor.env"
)

# runs as root inside the container
cat > "$WORKDIR/setup.sh" <<'SETUP'
#!/usr/bin/env bash
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
export LANG=C.UTF-8 LC_ALL=C.UTF-8
export DEBIAN_FRONTEND=noninteractive

APP_USER=epexpredictor
APP_HOME=/opt/epexpredictor
DATA_DIR=/var/lib/epexpredictor

echo "--- installing system packages"
apt-get -qq -o Acquire::Retries=3 update
apt-get -qq -y -o Acquire::Retries=3 -o DPkg::Lock::Timeout=300 install --no-install-recommends \
    python3 python3-venv git ca-certificates libgomp1 > /dev/null

python3 - <<'PY'
import sys
if sys.version_info < (3, 12):
    sys.exit(f"EpexPredictor needs Python 3.12 or newer, but this OS template has {sys.version.split()[0]}")
PY

if ! id -u "$APP_USER" >/dev/null 2>&1; then
    useradd --system --user-group --home-dir "$APP_HOME" --shell /usr/sbin/nologin "$APP_USER"
fi
install -d -o "$APP_USER" -g "$APP_USER" "$APP_HOME" "$DATA_DIR"
cd "$APP_HOME"
as_app() { runuser -u "$APP_USER" -- env HOME="$APP_HOME" "$@"; }

echo "--- fetching EpexPredictor ($REPO_REF)"
as_app git clone --quiet --branch "$REPO_REF" "$REPO_URL" "$APP_HOME/app"

echo "--- installing Python packages (takes a few minutes)"
as_app python3 -m venv "$APP_HOME/venv"
as_app "$APP_HOME/venv/bin/pip" install --quiet --no-cache-dir -r "$APP_HOME/app/requirements.txt"

# a service that does not run as root needs an extra permission for ports below 1024
CAPABILITIES=""
if [ "$PORT" -lt 1024 ]; then
    CAPABILITIES="AmbientCapabilities=CAP_NET_BIND_SERVICE"
fi

cat > /etc/systemd/system/epexpredictor.service <<UNIT
[Unit]
Description=EpexPredictor - EPEX day-ahead price forecast API
After=network-online.target
Wants=network-online.target

[Service]
User=$APP_USER
Group=$APP_USER
WorkingDirectory=$APP_HOME/app
EnvironmentFile=/etc/epexpredictor.env
# A stored data file that was cut off while it was written (power loss, kill) would keep the app from starting.
# Remove it, the app downloads its content again.
ExecStartPre=/bin/sh -c 'for f in $DATA_DIR/*.json.gz; do gzip -t "\$\$f" 2>/dev/null || rm -f "\$\$f"; done'
ExecStart=$APP_HOME/venv/bin/uvicorn predictor.api.priceapi:app --host 0.0.0.0 --port $PORT
# The app downloads its data before it starts to listen. If a data source turns it away meanwhile (rate limit),
# the app exits. It is then started again a minute later and continues with what it already stored. As long as
# it keeps failing, the pause grows up to 10 minutes, so that a source which refuses is not asked every minute.
Restart=on-failure
RestartSec=60
RestartSteps=4
RestartMaxDelaySec=600
# While it downloads its data or trains, the app does not act on a request to stop. Do not wait long for it,
# stopping it the hard way does no harm (see ExecStartPre).
TimeoutStopSec=30
NoNewPrivileges=true
$CAPABILITIES

[Install]
WantedBy=multi-user.target
UNIT

cat > /usr/local/bin/epexpredictor-update <<UPDATE
#!/usr/bin/env bash
# Pulls the latest code, updates the Python packages and restarts the service.
# Without argument it stays on the installed branch, "epexpredictor-update <branch>" switches to another one.
set -euo pipefail
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
cd $APP_HOME/app
as_app() { runuser -u $APP_USER -- env HOME=$APP_HOME "\$@"; }
as_app git fetch --quiet --prune origin
if [ -n "\${1:-}" ]; then
    as_app git checkout --quiet "\$1"
fi
as_app git pull --ff-only
as_app $APP_HOME/venv/bin/pip install --quiet --no-cache-dir -r requirements.txt
systemctl restart epexpredictor
echo "EpexPredictor was updated and restarted. It answers again once it has loaded its data and trained its models."
UPDATE
chmod 755 /usr/local/bin/epexpredictor-update

systemctl daemon-reload
systemctl enable --now epexpredictor

# a broken installation fails within a few seconds (e.g. on import), the download of the data takes much longer
sleep 10
if ! systemctl is-active --quiet epexpredictor; then
    journalctl --no-pager --lines 40 --unit epexpredictor || true
    echo "The service did not stay up, see its log above" >&2
    exit 1
fi
SETUP

msg "Installing EpexPredictor inside the container"
pct push "$CTID" "$WORKDIR/epexpredictor.env" /etc/epexpredictor.env --perms 600
pct push "$CTID" "$WORKDIR/setup.sh" /root/epexpredictor-setup.sh --perms 700
pct exec "$CTID" -- env REPO_URL="$REPO_URL" REPO_REF="$REPO_REF" PORT="$PORT" bash /root/epexpredictor-setup.sh
pct exec "$CTID" -- rm -f /root/epexpredictor-setup.sh

# ---------------------------------------------------------------------------------------------- done

CT_IP="$(pct exec "$CTID" -- hostname -I 2>/dev/null | awk '{print $1}' || true)"
URL="http://${CT_IP:-<address of the container>}:${PORT}"
cat <<EOF

EpexPredictor is installed in container $CTID and its service is running.

    Prices     $URL/prices?region=DE
    API docs   $URL/docs
    Logs       pct exec $CTID -- journalctl -u epexpredictor -f
    Update     pct exec $CTID -- /usr/local/bin/epexpredictor-update
    Shell      pct enter $CTID
    Settings   /etc/epexpredictor.env in the container, then: pct exec $CTID -- systemctl restart epexpredictor

The API does not answer yet (connection refused): the service first downloads about 18 months of price and weather
history for every region and trains its models, and only then starts to listen. That takes a quarter of an hour at
least (the price source is only asked twice a minute), and longer if the weather source limits the requests.
Later restarts use the stored data and are much faster. Follow the log to see the progress.
EOF
