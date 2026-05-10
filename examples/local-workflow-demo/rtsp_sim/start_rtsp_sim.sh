#!/usr/bin/env bash
# Spins up a local RTSP server (MediaMTX) and pushes the traffic CCTV video
# to it on infinite loop. URL: rtsp://localhost:8554/traffic
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
DEMO_DIR="$(cd "$HERE/.." && pwd)"
VIDEO="${1:-$DEMO_DIR/sample_video/traffic_cctv.mp4}"
RTSP_PATH="${2:-traffic}"
PORT="${3:-8554}"

if [[ ! -f "$VIDEO" ]]; then
  echo "video not found: $VIDEO" >&2
  exit 1
fi

# Bootstrap: download MediaMTX binary on first run (not checked into the repo)
MTX_VERSION="v1.9.3"
ARCH="$(uname -m)"
case "$ARCH" in
  x86_64)  MTX_ARCH="amd64"  ;;
  aarch64) MTX_ARCH="arm64v8" ;;
  *) echo "unsupported arch: $ARCH" >&2; exit 1 ;;
esac
if [[ ! -x "$HERE/mediamtx" ]]; then
  URL="https://github.com/bluenviron/mediamtx/releases/download/${MTX_VERSION}/mediamtx_${MTX_VERSION}_linux_${MTX_ARCH}.tar.gz"
  echo "fetching mediamtx ${MTX_VERSION} from ${URL} ..."
  wget -q --timeout=60 "$URL" -O "$HERE/mediamtx.tar.gz" || {
    echo "failed to download mediamtx" >&2; exit 1; }
  tar -xzf "$HERE/mediamtx.tar.gz" -C "$HERE" mediamtx
  chmod +x "$HERE/mediamtx"
  rm -f "$HERE/mediamtx.tar.gz"
fi

# Kill anything already on the port (idempotent restart)
if ss -tlnp 2>/dev/null | grep -q ":${PORT} "; then
  pid=$(ss -tlnp | grep ":${PORT} " | grep -oP 'pid=\K\d+' || true)
  [[ -n "$pid" ]] && kill "$pid" 2>/dev/null || true
  sleep 1
fi
pkill -f "ffmpeg.*rtsp://localhost:${PORT}/${RTSP_PATH}" 2>/dev/null || true

# Start the RTSP server in the background, log to file
"$HERE/mediamtx" "$HERE/mediamtx.yml" >"$HERE/mediamtx.log" 2>&1 &
MTX_PID=$!
echo "mediamtx PID=$MTX_PID (logs: $HERE/mediamtx.log)"

# Wait for port to be open
for i in {1..20}; do
  ss -tlnp 2>/dev/null | grep -q ":${PORT} " && break
  sleep 0.25
done

# Push the video on infinite loop, h264-encoded, TCP transport.
# -re => respect source frame rate (real-time pacing)
# -stream_loop -1 => loop forever
ffmpeg -hide_banner -loglevel warning \
       -re -stream_loop -1 -i "$VIDEO" \
       -c:v libx264 -preset veryfast -tune zerolatency -bf 0 \
       -an -f rtsp -rtsp_transport tcp \
       "rtsp://localhost:${PORT}/${RTSP_PATH}" \
       >"$HERE/ffmpeg.log" 2>&1 &
FF_PID=$!
echo "ffmpeg PID=$FF_PID (logs: $HERE/ffmpeg.log)"

cat <<EOF

RTSP simulation running.
  Server : rtsp://localhost:${PORT}/${RTSP_PATH}
  LAN    : rtsp://$(hostname -I | awk '{print $1}'):${PORT}/${RTSP_PATH}
  Source : $VIDEO  (looping)

Stop:    kill $MTX_PID $FF_PID
EOF
