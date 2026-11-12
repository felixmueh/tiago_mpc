#!/usr/bin/env bash
set -euo pipefail

study_root=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
study_mode=${1:-all}
study_output=${2:-_expr_data/jetson-studies}
case "$study_mode" in
  all|histograms|matrix|describe) ;;
  *) printf 'Usage: ./run_jetson_studies.sh [all|histograms|matrix|describe] [output-root]\n' >&2; exit 2 ;;
esac
cd "$study_root"
mkdir -p /workspace/.locks
exec 9>/workspace/.locks/heavy.lock
flock 9
trap 'exit 130' INT
trap 'exit 143' TERM
export UV_PROJECT_ENVIRONMENT=.venv-container
export CMAKE_BUILD_PARALLEL_LEVEL=1 MAX_JOBS=1 GIT_TERMINAL_PROMPT=0

prepare_dependency() {
  local name=$1 url=$2 base=$3 expected_tree=$4 patch=$5
  local target="$study_root/_expr_data/jetson-deps/$name"
  local head tree
  if [[ ! -d "$target/.git" ]]; then
    if [[ -e "$target" ]]; then
      printf 'Refusing to replace existing dependency path: %s\n' "$target" >&2
      return 1
    fi
    mkdir -p "$(dirname -- "$target")"
    git init --quiet "$target"
    git -C "$target" remote add origin "$url"
    git -C "$target" config core.autocrlf false
  fi
  if ! head=$(git -C "$target" rev-parse --verify HEAD 2>/dev/null); then
    if [[ $(git -C "$target" remote get-url origin) != "$url" ]]; then
      printf 'Incomplete dependency has an unexpected origin: %s\n' "$target" >&2
      return 1
    fi
    git -C "$target" fetch --depth 1 origin "$base"
    GIT_LFS_SKIP_SMUDGE=1 git -C "$target" -c advice.detachedHead=false checkout --detach FETCH_HEAD
    head=$(git -C "$target" rev-parse HEAD)
  fi
  if [[ "$head" != "$base" ]]; then
    printf 'Dependency %s must stay at base %s, found %s; leaving it unchanged.\n' "$name" "$base" "$head" >&2
    return 1
  fi
  if ! git -C "$target" diff --quiet --ignore-submodules=all; then
    printf 'Dependency %s has unstaged changes; leaving it unchanged.\n' "$name" >&2
    return 1
  fi
  if [[ -n $(git -C "$target" ls-files --others --exclude-standard) ]]; then
    printf 'Dependency %s has untracked files; leaving it unchanged.\n' "$name" >&2
    return 1
  fi
  tree=$(git -C "$target" write-tree)
  if [[ "$tree" == $(git -C "$target" rev-parse HEAD^{tree}) ]]; then
    git -C "$target" apply --index --check "$patch"
    git -C "$target" apply --index "$patch"
    tree=$(git -C "$target" write-tree)
  fi
  if [[ "$tree" != "$expected_tree" ]]; then
    printf 'Dependency %s differs from the pinned patched tree; leaving it unchanged.\n' "$name" >&2
    return 1
  fi
  printf '%s source tree verified: %s\n' "$name" "$tree"
}

prepare_dependency GATO https://github.com/A2R-Lab/GATO.git \
  c9da6a58c7fec4c688f673b79de1c8267359098c a9ccd156c9beefedc14a4941b90b601186f71c10 \
  "$study_root/tools/jetson/gato.patch"
prepare_dependency GRiD https://github.com/A2R-Lab/GRiD.git \
  712df1c6d91e0c07ff5e482df3c5f703df364fb6 cf7a44d6ad528c54295b47772403aa40c8e16912 \
  "$study_root/tools/jetson/grid.patch"
git -C "$study_root/_expr_data/jetson-deps/GRiD" submodule foreach --recursive '
  if [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    printf "Dependency submodule has local changes; leaving it unchanged: %s\n" "$displaypath" >&2
    exit 1
  fi
'
git -C "$study_root/_expr_data/jetson-deps/GRiD" submodule update --init --recursive --depth 1 -- \
  external/GLASS external/RBDReference external/URDFParser
git -C "$study_root/_expr_data/jetson-deps/GRiD" submodule foreach --recursive '
  if [ "$(git rev-parse HEAD)" != "$sha1" ] || [ -n "$(git status --porcelain --untracked-files=normal)" ]; then
    printf "Dependency submodule is not clean at its pinned gitlink: %s\n" "$displaypath" >&2
    exit 1
  fi
'
uv sync --locked --no-dev

if [[ "$study_mode" == describe ]]; then
  uv run --locked --no-dev python -m experiments.pcg_bdsv_histograms --describe
  uv run --locked --no-dev python -m experiments.jetson_nb_matrix --describe
  exit 0
fi

mkdir -p "$study_output"
study_status=0
if [[ "$study_mode" == all || "$study_mode" == histograms ]]; then
  if uv run --locked --no-dev python -m experiments.pcg_bdsv_histograms \
    --output "$study_output/histograms" --knots 16 32 64 128 256 512 --repeats 3; then
    histogram_status=0
  else
    histogram_status=$?
  fi
  printf 'histograms exit=%s\n' "$histogram_status" | tee -a "$study_output/launcher-status.log"
  case "$histogram_status" in 130|143) exit "$histogram_status" ;; esac
  if [[ "$histogram_status" != 0 ]]; then study_status=1; fi
fi
if [[ "$study_mode" == all || "$study_mode" == matrix ]]; then
  if uv run --locked --no-dev python -m experiments.jetson_nb_matrix \
    --output "$study_output/nb-matrix"; then
    matrix_status=0
  else
    matrix_status=$?
  fi
  printf 'matrix exit=%s\n' "$matrix_status" | tee -a "$study_output/launcher-status.log"
  case "$matrix_status" in 130|143) exit "$matrix_status" ;; esac
  if [[ "$matrix_status" != 0 ]]; then study_status=1; fi
fi
exit "$study_status"
