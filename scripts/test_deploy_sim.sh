#!/usr/bin/env bash
# Mirror the deploy-frontend prepare-site step from .github/workflows/collect-news.yml
# and re-run the static check against the resulting site/ tree. This catches
# anything that breaks only after the deploy-time sed pass or the cp -r flatten.
#
# Invocation: `bash scripts/test_deploy_sim.sh`
# Exits non-zero on first failure.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# GNU-sed on Linux supports `-i ''`-less in-place; macOS BSD sed needs a backup arg.
# Use perl for portability since both runners (local macOS, Linux CI) have it.
sed_inplace() {
    local pattern="$1"; shift
    perl -i -pe "$pattern" "$@"
}

sed_delete_block() {
    # Match on the workflow's `sed -i '/pattern/,/pattern/d'` — delete from
    # first marker to second, inclusive. `$found` flips state across lines.
    local marker="$1"; shift
    perl -i -ne 'BEGIN{$f=0} if(/\Q'"$marker"'\E/){ $f = !$f; next } print unless $f || /\Q'"$marker"'\E/' "$@"
}

FAILURES=0
fail() { echo "  FAIL  $*"; FAILURES=$((FAILURES + 1)); }
okmsg() { echo "  ok    $*"; }

TMPDIR="$(mktemp -d)"
trap 'rm -rf "$TMPDIR"' EXIT

run_scenario() {
    local scenario="$1"
    local ga_id="$2"
    local site="$TMPDIR/$scenario"
    mkdir -p "$site"
    cp -r src/frontend/* "$site/"
    rm -rf "$site/prototypes"

    if [[ -n "$ga_id" ]]; then
        # inject path — mirror `sed -i "s|__GOOGLE_ANALYTICS__|$ID|g" site/index.html`
        perl -i -pe "s|__GOOGLE_ANALYTICS__|$ga_id|g" "$site/index.html"
    else
        # remove-block path — mirror `sed -i '/<!-- __GOOGLE_ANALYTICS__ -->/,/<!-- __GOOGLE_ANALYTICS__ -->/d'`
        sed_delete_block '<!-- __GOOGLE_ANALYTICS__ -->' "$site/index.html"
    fi
    echo "$site"
}

echo "== scenario: GA injected =="
SITE_INJECT="$(run_scenario inject G-TEST12345)"
[[ -f "$SITE_INJECT/index.html" ]] || fail "index.html missing after prepare-site"
if grep -q 'G-TEST12345' "$SITE_INJECT/index.html"; then
    okmsg "GA id substituted into index.html"
else
    fail "GA id not injected into index.html"
fi
if grep -q '__GOOGLE_ANALYTICS__' "$SITE_INJECT/index.html"; then
    fail "__GOOGLE_ANALYTICS__ placeholder still present after injection"
else
    okmsg "no __GOOGLE_ANALYTICS__ placeholders remain"
fi

echo "== scenario: GA removed =="
SITE_REMOVE="$(run_scenario remove '')"
[[ -f "$SITE_REMOVE/index.html" ]] || fail "index.html missing after prepare-site"
if grep -q '__GOOGLE_ANALYTICS__' "$SITE_REMOVE/index.html"; then
    fail "__GOOGLE_ANALYTICS__ tokens remain after removal"
else
    okmsg "GA block cleanly removed"
fi
# gtag script block is deleted, but the Clarity script must still be present.
if grep -q 'clarity.ms/tag' "$SITE_REMOVE/index.html"; then
    okmsg "MS Clarity script preserved across GA removal"
else
    fail "Clarity script was collateral damage of GA block removal"
fi

echo "== prototypes excluded from deploy =="
if [[ -e "$SITE_INJECT/prototypes" ]]; then
    fail "prototypes/ present in prepared site (design mockup would ship to production)"
else
    okmsg "prototypes/ absent from prepared site"
fi

echo "== API content =="
if [[ -f "$SITE_INJECT/api/latest.json" ]]; then
    COUNT="$(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(len(d.get("articles", [])))' "$SITE_INJECT/api/latest.json")"
    if [[ "$COUNT" -gt 0 ]]; then
        okmsg "latest.json has $COUNT articles"
    else
        fail "latest.json has zero articles"
    fi
else
    fail "site/api/latest.json missing"
fi

echo "== re-run static check on prepared site =="
if SKIP_GA_MARKERS=1 node scripts/test_frontend_static.mjs "$SITE_INJECT" > "$TMPDIR/static.log" 2>&1; then
    okmsg "static check passes against site/"
else
    fail "static check failed against site/ (see below)"
    sed 's/^/    /' "$TMPDIR/static.log"
fi

echo "== all script/link refs resolve inside site/ =="
# Extract href/src values from index.html, skip absolute/protocol-relative/data/fragment.
python3 - "$SITE_INJECT" <<'PY'
import os, re, sys
site = sys.argv[1]
html = open(os.path.join(site, 'index.html')).read()
refs = re.findall(r'''(?:href|src)=["']([^"']+)["']''', html)
missing = []
for r in refs:
    if r.startswith(('http://', 'https://', '//', 'data:', '#')):
        continue
    p = r[1:] if r.startswith('/') else r
    if not os.path.exists(os.path.join(site, p)):
        missing.append(r)
if missing:
    print('FAIL: refs missing in site/:', missing)
    sys.exit(1)
print(f'  ok    {len(refs)} refs resolve inside site/')
PY
PY_EXIT=$?
if [[ $PY_EXIT -ne 0 ]]; then
    fail "index.html refs missing in prepared site (see python output)"
fi

echo
if [[ "$FAILURES" -gt 0 ]]; then
    echo "FAIL — $FAILURES issue(s) in deploy simulation"
    exit 1
fi
echo "PASS — deploy simulation clean"
