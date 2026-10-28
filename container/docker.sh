#!/usr/bin/env bash
# Builds the tiago_mpc container image, creates or starts the container and attaches a shell.
# Run it on the machine that runs the container (the desktop or the Jetson).
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CHECKOUT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WORKSPACE_HOST="$(cd "${CHECKOUT}/.." && pwd)"   # impl/: sibling worktrees and ../GATO resolve
WORKDIR="/workspace/$(basename "${CHECKOUT}")"
LOCAL_UID="$(id -u)"
LOCAL_GID="$(id -g)"
LOCAL_USER="$(id -un)"

target=desktop
rebuild_image=0 recreate_container=0 restart_container=0 attach_shell=1
while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            if [[ $# -lt 2 ]]; then echo '--target requires desktop or jetson' >&2; exit 1; fi
            target="$2"; shift 2 ;;
        --rebuild-image) rebuild_image=1; shift ;;
        --recreate-container) recreate_container=1; shift ;;
        --restart) restart_container=1; shift ;;
        --no-attach) attach_shell=0; shift ;;
        -h|--help)
            cat <<HELP
Usage: $0 [--target desktop|jetson] [--rebuild-image] [--recreate-container] [--restart] [--no-attach]

  --target              Hardware target (default: desktop; the Jetson uses L4T r36.4.0).
  --rebuild-image       Rebuild the image, then recreate the container.
  --recreate-container  Recreate the container without rebuilding the image.
  --restart             Stop and start the container.
  --no-attach           Leave the container running without opening a shell.

The shell starts in this checkout, inside the mounted parent directory. Select the
ROS profile in it: source container/ros_simulation.sh or container/ros_tiago.sh.
HELP
            exit 0 ;;
        *) printf 'Unknown option: %s\n' "$1" >&2; exit 1 ;;
    esac
done

image=tiago-mpc
container=tiago-mpc-container
gpu_args=(--gpus all)
case "${target}" in
    desktop) base_image=nvidia/cuda:12.9.1-devel-ubuntu22.04 ;;
    jetson)
        base_image=nvcr.io/nvidia/l4t-jetpack:r36.4.0
        image="${image}:jetson-r36.4.0"
        container="${container}-jetson-r36.4.0"
        gpu_args=(--runtime=nvidia -e NVIDIA_VISIBLE_DEVICES=all)
        ;;
    *) printf 'Unknown target: %s\n' "${target}" >&2; exit 1 ;;
esac

if [[ "${rebuild_image}" -eq 1 ]] || ! docker image inspect "${image}" >/dev/null 2>&1; then
    docker build \
        --build-arg "BASE_IMAGE=${base_image}" \
        --build-arg "LOCAL_UID=${LOCAL_UID}" \
        --build-arg "LOCAL_GID=${LOCAL_GID}" \
        --build-arg "LOCAL_USER=${LOCAL_USER}" \
        -t "${image}" -f "${SCRIPT_DIR}/Dockerfile" "${SCRIPT_DIR}"
fi

container_exists="$(docker ps -aq -f "name=^/${container}$")"
if [[ -n "${container_exists}" && ( "${rebuild_image}" -eq 1 || "${recreate_container}" -eq 1 ) ]]; then
    docker rm -f "${container}" >/dev/null
    container_exists=""
fi
if [[ -z "${container_exists}" ]]; then
    # label=disable: the workspace is also mounted by other containers, so no SELinux relabeling.
    docker create -it "${gpu_args[@]}" --network=host --ipc=host --security-opt label=disable \
        -v "${WORKSPACE_HOST}:/workspace" -w "${WORKDIR}" \
        --name "${container}" "${image}" >/dev/null
elif [[ "${restart_container}" -eq 1 ]]; then
    docker stop "${container}" >/dev/null
fi
docker start "${container}" >/dev/null
if [[ "${attach_shell}" -eq 1 ]]; then
    docker exec -it -w "${WORKDIR}" "${container}" /bin/bash
else
    printf "Container '%s' is ready.\n" "${container}"
fi
