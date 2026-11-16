#!/usr/bin/env bash
# Build the app and install + launch it on the iPhone connected by cable (no Xcode window needed).
# Builds outside the iCloud-synced repo: iCloud adds file metadata that code signing rejects
# ("resource fork, Finder information, or similar detritus not allowed").
# Needs: Xcode signed in to your Apple ID (Settings > Accounts), iPhone in Developer Mode.
set -euo pipefail
export DEVELOPER_DIR="${DEVELOPER_DIR:-/Applications/Xcode.app/Contents/Developer}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="$HOME/Library/Developer/Xcode/DerivedData/SynapSysDemo-cli"
xcodebuild -project "$HERE/SynapSysDemo.xcodeproj" -target SynapSysDemo -sdk iphoneos \
  -configuration Debug -allowProvisioningUpdates SYMROOT="$OUT" OBJROOT="$OUT/obj" build -quiet
DEVICE="$(xcrun devicectl list devices 2>/dev/null | grep -E 'iPhone.*connected' \
  | grep -oE '[0-9A-F]{8}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{4}-[0-9A-F]{12}' | head -1)"
if [[ -z "$DEVICE" ]]; then echo "no iPhone connected by cable" >&2; exit 1; fi
xcrun devicectl device install app --device "$DEVICE" "$OUT/Debug-iphoneos/SynapSysDemo.app" >/dev/null
xcrun devicectl device process launch --device "$DEVICE" com.synapsys.livedemo >/dev/null
echo "installed and launched on $DEVICE"
