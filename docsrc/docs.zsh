#!/usr/bin/env zsh
#
# Build, preview, and deploy the x4c documentation site (Jupyter Book 2 / MyST).
#
#   ./docsrc/docs.zsh preview      # regenerate assets, build, serve on localhost
#   ./docsrc/docs.zsh build        # regenerate assets and build only
#   ./docsrc/docs.zsh deploy       # build for GitHub Pages and push to gh-pages
#   ./docsrc/docs.zsh stop         # stop a running preview server
#   ./docsrc/docs.zsh clean        # remove build output
#
# Options:
#   -p, --port N     preview port (default 4200)
#   -y, --yes        skip the deploy confirmation prompt
#
# To view the preview from a laptop, tunnel the port over SSH:
#   ssh -N -L 4200:localhost:4200 $USER@$(hostname -s).hpc.ucar.edu
# then open http://localhost:4200

emulate -L zsh
set -euo pipefail

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
SCRIPT=${0:A}                         # this script (inside a function $0 is the fn name)
DOCSRC=${SCRIPT:h}                    # directory holding this script
REPO=${DOCSRC:h}                      # repository root
BUILD_DIR="$DOCSRC/_build/html"
CONDA_ENV=${X4C_DOCS_ENV:-x4c-py313}
PORT=4200
ASSUME_YES=0

# Project sites are served from https://<org>.github.io/<repo>/, so MyST needs to
# know the subpath or every asset URL resolves against the domain root and 404s.
# Preview is served from / instead, so BASE_URL must be EMPTY there.
DEPLOY_BASE_URL=${X4C_DOCS_BASE_URL:-/x4c}
DEPLOY_BRANCH=gh-pages

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
autoload -U colors && colors
info()  { print -P "%F{cyan}==>%f $*" }
ok()    { print -P "%F{green}✓%f $*" }
warn()  { print -P "%F{yellow}!%f $*" }
die()   { print -P "%F{red}✗%f $*" >&2; exit 1 }

usage() { sed -n '3,17p' "$SCRIPT" | sed 's/^# \?//'; exit ${1:-0} }

# Make `conda activate` usable from a non-interactive shell. `module` is a shell
# function, so it has to be sourced here rather than inherited.
activate_env() {
  if ! command -v conda >/dev/null 2>&1; then
    if ! typeset -f module >/dev/null 2>&1; then
      [[ -f /etc/profile.d/z00_modules.sh ]] && source /etc/profile.d/z00_modules.sh
    fi
    typeset -f module >/dev/null 2>&1 || die "no 'conda' and no 'module' command available"
    module load conda
  fi
  eval "$(conda shell.zsh hook)"
  conda activate "$CONDA_ENV" \
    || die "could not activate conda env '$CONDA_ENV' (override with X4C_DOCS_ENV)"

  # jupyter-book only works with the env active: the conda package relies on
  # activation scripts to set QUARTO/NODE-style paths.
  command -v jupyter-book >/dev/null 2>&1 \
    || die "jupyter-book not found in '$CONDA_ENV' -- conda install -c conda-forge jupyter-book"
}

# Kill whatever holds the preview port. Match by PORT, never by command pattern:
# a pattern like "jupyter-book start" also matches this script and kills it.
stop_server() {
  local pids
  # `|| true` matters: with `set -o pipefail`, grep exits 1 when the port is
  # already free, which under `set -e` would abort the script instead of
  # reporting "nothing listening". Empty elements are dropped so the count is real.
  pids=(${(f)"$(ss -ltnp 2>/dev/null | grep -E "127\.0\.0\.1:${PORT}\b" \
                 | grep -oE 'pid=[0-9]+' | cut -d= -f2 | sort -u || true)"})
  pids=(${pids:#})
  if (( ${#pids} )); then
    for p in $pids; do kill "$p" 2>/dev/null && info "stopped pid $p on port $PORT"; done
    sleep 1
    ok "port $PORT released"
  else
    info "nothing listening on port $PORT"
  fi
}

# --------------------------------------------------------------------------
# Actions
# --------------------------------------------------------------------------
build_site() {
  local base_url="${1:-}"

  activate_env
  cd "$DOCSRC"

  info "regenerating gallery assets"
  python "$DOCSRC/scripts/extract_figs.py" \
    || die "figure extraction failed -- have the notebooks been executed?"

  # MyST does not clear stale output between builds, so hashed assets from
  # previous runs pile up (37 MB after three builds vs 13 MB clean). Removing
  # only _build/html keeps the 127 MB template cache in _build/templates, so
  # this costs nothing in download time.
  rm -rf "$BUILD_DIR"

  if [[ -n "$base_url" ]]; then
    info "building with BASE_URL=$base_url"
    BASE_URL="$base_url" jupyter-book build --html
  else
    info "building for local preview (no BASE_URL)"
    BASE_URL= jupyter-book build --html
  fi

  [[ -f "$BUILD_DIR/index.html" ]] || die "build produced no index.html"

  # `jupyter-book build --html` returns before it has finished writing its JS
  # bundles. Watch total BYTES, not file count: MyST creates every file first and
  # fills them afterwards, so the count settles while ~35 MB is still being
  # written (this reported 1.7 MB for a 37 MB build before the fix).
  local prev=-1 now=0 settle=0
  repeat 60 {
    now=$(du -sk "$BUILD_DIR" 2>/dev/null | cut -f1)
    if (( now == prev && now > 0 )); then
      (( ++settle >= 3 )) && break
    else
      settle=0
    fi
    prev=$now
    sleep 1
  }

  # GitHub Pages runs Jekyll by default, which skips paths beginning with an
  # underscore. Harmless to add even when nothing currently starts with one.
  touch "$BUILD_DIR/.nojekyll"

  local n_pages n_assets size
  n_pages=$(find "$BUILD_DIR" -name '*.html' | wc -l | tr -d ' ')
  n_assets=$(ls "$BUILD_DIR/assets" 2>/dev/null | wc -l | tr -d ' ')
  size=$(du -sh "$BUILD_DIR" | cut -f1)

  (( n_assets > 0 )) || die "no assets in $BUILD_DIR/assets -- check static_files in myst.yml"
  ok "built $n_pages pages, $n_assets assets ($size)"
}

do_preview() {
  build_site ""
  stop_server

  info "serving $BUILD_DIR on 127.0.0.1:$PORT"
  ( cd "$BUILD_DIR" && setsid nohup python -m http.server "$PORT" --bind 127.0.0.1 \
      >/dev/null 2>&1 </dev/null & )

  # wait for it rather than assuming
  local tries=0
  until curl -sf -o /dev/null "http://127.0.0.1:$PORT/" 2>/dev/null; do
    (( ++tries > 20 )) && die "server did not come up on port $PORT"
    sleep 1
  done

  ok "preview running"
  print
  # Braces are required: zsh reads $PORT:l as the :l (lowercase) modifier and
  # silently eats the "l" of localhost.
  print "  local:   http://localhost:${PORT}"
  print "  tunnel:  ssh -N -L ${PORT}:localhost:${PORT} ${USER}@$(hostname -s).hpc.ucar.edu"
  print
  print "  stop it with:  $SCRIPT stop"
}

do_deploy() {
  cd "$REPO"

  # Refuse to publish from a dirty tree -- the deployed site should correspond to
  # a known commit, otherwise it is impossible to say what is live.
  if [[ -n "$(git status --porcelain)" ]]; then
    warn "working tree has uncommitted changes:"
    git status --short | head -15
    print
    if (( ! ASSUME_YES )); then
      read -q "REPLY?Deploy anyway? [y/N] " || { print; die "aborted" }
      print
    fi
  fi

  build_site "$DEPLOY_BASE_URL"

  local remote_url
  remote_url=$(git remote get-url origin)

  print
  info "about to publish"
  print "  from:    $BUILD_DIR"
  print "  to:      $remote_url ($DEPLOY_BRANCH)"
  print "  BASE_URL: $DEPLOY_BASE_URL"
  print

  if (( ! ASSUME_YES )); then
    read -q "REPLY?Push to $DEPLOY_BRANCH? [y/N] " || { print; die "aborted" }
    print
  fi

  command -v ghp-import >/dev/null 2>&1 \
    || die "ghp-import not found -- conda install -c conda-forge ghp-import"

  # -n .nojekyll  -p push  -f force  -b branch
  ghp-import -n -p -f -b "$DEPLOY_BRANCH" "$BUILD_DIR" \
    || die "ghp-import failed"

  ok "deployed to $DEPLOY_BRANCH"
  print
  print "  Pages must be serving from the '$DEPLOY_BRANCH' branch for this to appear."
  print "  If Pages is set to 'GitHub Actions' instead, this push will not publish."
}

do_clean() {
  info "removing $DOCSRC/_build"
  rm -rf "$DOCSRC/_build"
  ok "clean"
}

# --------------------------------------------------------------------------
# Argument parsing
# --------------------------------------------------------------------------
ACTION=""
while (( $# )); do
  case "$1" in
    preview|build|deploy|stop|clean) ACTION="$1"; shift ;;
    -p|--port)  PORT="${2:?--port needs a value}"; shift 2 ;;
    -y|--yes)   ASSUME_YES=1; shift ;;
    -h|--help)  usage 0 ;;
    *)          print "unknown argument: $1" >&2; usage 1 ;;
  esac
done

case "$ACTION" in
  preview) do_preview ;;
  build)   build_site "" ;;
  deploy)  do_deploy ;;
  stop)    stop_server ;;
  clean)   do_clean ;;
  "")      usage 1 ;;
esac
