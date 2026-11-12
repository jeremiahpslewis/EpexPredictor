#!/usr/bin/env bash
_CS_DEFAULT_URL="${_CS_DEFAULT_URL:-https://raw.githubusercontent.com/jeremiahpslewis/EpexPredictor/main/proxmox}"
source <(curl -fsSL "${COMMUNITY_SCRIPTS_CORE_URL:-https://raw.githubusercontent.com/community-scripts/core/main}/core/build.func")
# License: BSD-3-Clause | https://github.com/jeremiahpslewis/EpexPredictor/raw/main/LICENSE
# Source: https://github.com/jeremiahpslewis/EpexPredictor

APP="EpexPredictor"
var_tags="${var_tags:-energy}"
var_cpu="${var_cpu:-2}"
var_ram="${var_ram:-4096}"
var_disk="${var_disk:-6}"
var_os="${var_os:-debian}"
var_version="${var_version:-13}"
var_unprivileged="${var_unprivileged:-1}"

header_info "$APP"
variables
color
catch_errors

function update_script() {
  header_info
  check_container_storage
  check_container_resources
  if [[ ! -d /opt/epexpredictor ]]; then
    msg_error "No ${APP} Installation Found!"
    exit
  fi

  msg_info "Updating ${APP}"
  cd /opt/epexpredictor
  output=$(git pull --no-rebase)
  if echo "$output" | grep -q "Already up to date."; then
    msg_ok "${APP} is already up to date."
    exit
  fi
  $STD uv pip install -r requirements.txt
  systemctl restart epexpredictor
  msg_ok "Updated successfully!"
  exit
}

start
build_container
description

msg_ok "Completed successfully!\n"
echo -e "${CREATING}${GN}${APP} setup has been successfully initialized!${CL}"
echo -e "${INFO}${YW}Access it using the following URL:${CL}"
echo -e "${GATEWAY}${BGN}http://${IP}:8000${CL}"
echo -e "${INFO}${YW}It answers once it has downloaded its data, which takes a quarter of an hour or more.${CL}"
