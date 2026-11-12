#!/usr/bin/env bash

# License: BSD-3-Clause | https://github.com/jeremiahpslewis/EpexPredictor/raw/main/LICENSE
# Source: https://github.com/jeremiahpslewis/EpexPredictor

source /dev/stdin <<<"$FUNCTIONS_FILE_PATH"
color
verb_ip6
catch_errors
setting_up_container
network_check
update_os

msg_info "Installing Dependencies"
$STD apt install -y git libgomp1 tzdata
msg_ok "Installed Dependencies"

PYTHON_VERSION="3.13" setup_uv

msg_info "Installing EpexPredictor"
$STD git clone https://github.com/jeremiahpslewis/EpexPredictor /opt/epexpredictor
cd /opt/epexpredictor
$STD uv venv
$STD uv pip install -r requirements.txt
mkdir -p /opt/epexpredictor/data
cat <<EOF >/opt/epexpredictor/.env
EPEXPREDICTOR_DATADIR=/opt/epexpredictor/data
#EPEXPREDICTOR_ENTSOE_API_KEY=
EOF
msg_ok "Installed EpexPredictor"

msg_info "Creating Service"
cat <<EOF >/etc/systemd/system/epexpredictor.service
[Unit]
Description=EpexPredictor
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
WorkingDirectory=/opt/epexpredictor
EnvironmentFile=/opt/epexpredictor/.env
ExecStart=/opt/epexpredictor/.venv/bin/uvicorn predictor.api.priceapi:app --host 0.0.0.0 --port 8000
Restart=on-failure
RestartSec=60

[Install]
WantedBy=multi-user.target
EOF
systemctl enable -q --now epexpredictor
msg_ok "Created Service"

motd_ssh
customize
cleanup_lxc
