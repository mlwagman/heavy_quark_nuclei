#!/usr/bin/env bash
# vendor_three.sh — build vendor/three/three.iife.js: three.js 0.186.1 + the bloom addons as ONE classic script
# (global THREE), so the World opens from file:// in Chrome too (Chrome blocks ES modules there; ticket 03).
# Offline: the tarball comes from the npm cache, pinned by integrity; esbuild is the copy already on disk in DnD.
#   bash talk/world/tools/vendor_three.sh        (re-running gives a byte-identical file)
set -euo pipefail
VER=0.186.1
INTEGRITY='sha512-blFeqb49wRCSG'   # prefix of npm's integrity for three-0.186.1.tgz; checked below
ESBUILD=${ESBUILD:-/home/gofer/program/DnD/ui/node_modules/.bin/esbuild}
W=$(cd "$(dirname "$0")/.." && pwd)
OUT="$W/vendor/three"
TMP=$(mktemp -u /tmp/vendor_three.XXXXXX); mkdir -p "$TMP"; trap 'rm -rf "$TMP"' EXIT
cd "$TMP"
PACK=$(npm pack "three@$VER" --offline --json 2>/dev/null)
got=$(printf '%s' "$PACK" | jq -r '.[0].integrity')
[ "${got#"$INTEGRITY"}" != "$got" ] || { echo "VENDOR FAIL integrity $got" >&2; exit 1; }
mkdir -p node_modules/three && tar -xzf "three-$VER.tgz" -C node_modules/three --strip-components=1
cat > entry.js <<'EOF'
export * from 'three';
export { EffectComposer } from 'three/addons/postprocessing/EffectComposer.js';
export { RenderPass } from 'three/addons/postprocessing/RenderPass.js';
export { UnrealBloomPass } from 'three/addons/postprocessing/UnrealBloomPass.js';
export { OutputPass } from 'three/addons/postprocessing/OutputPass.js';
EOF
mkdir -p "$OUT"
"$ESBUILD" entry.js --bundle --format=iife --global-name=THREE --legal-comments=inline --log-level=warning \
  --banner:js="/* three.js $VER + EffectComposer/RenderPass/UnrealBloomPass/OutputPass as a classic script (global THREE). npm integrity ${got:0:24}…; esbuild $("$ESBUILD" --version); built by talk/world/tools/vendor_three.sh. MIT, see LICENSE. */" \
  --outfile="$OUT/three.iife.js"
cp node_modules/three/LICENSE "$OUT/LICENSE"
echo "VENDOR OK $(sha256sum "$OUT/three.iife.js" | cut -c1-16) $(wc -c < "$OUT/three.iife.js") bytes"
