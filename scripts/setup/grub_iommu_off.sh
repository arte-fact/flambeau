#!/usr/bin/env bash
# grub_iommu_off.sh — minimal patch: replace `iommu=pt` with `iommu=off`
# on the kernel cmdline, nothing else.
#
# Why: at the start of this session the rig was running with `iommu=pt` in
# the cmdline but `/sys/kernel/iommu_groups/` was empty — i.e. AMD-Vi was
# present but not registering devices, effectively off. Yesterday's 37 t/s
# intra-die TP2 baseline was measured in that state. After reverting BIOS
# IOMMU back to Enabled, AMD-Vi is now active in passthrough mode and adds
# ~10% overhead to BAR1 P2P AR traffic — the source of the 33 t/s ceiling.
# `iommu=off` suppresses AMD-Vi at the kernel level regardless of the BIOS
# setting, restoring the original measurement state without needing another
# BIOS trip.
#
# This script ONLY touches `iommu=`. It does NOT add `pcie_aspm=off` or
# `pci=pcie_bus_perf` — those are unrelated tweaks for cross-die P2P
# stability and have a separate script (grub_p2p_tweaks.sh).
#
# Default mode is DRY-RUN. Re-run with --apply to actually edit + update-grub.

set -euo pipefail

APPLY=0
case "${1:-}" in
  --apply) APPLY=1 ;;
  ""|--dry-run) APPLY=0 ;;
  -h|--help)
    sed -n '1,/^set -euo/p' "$0" | sed '$d'
    exit 0
    ;;
  *) echo "unknown arg: $1 (try --dry-run or --apply)" >&2; exit 2 ;;
esac

GRUB_FILE=/etc/default/grub
BACKUP=/etc/default/grub.bak.iommu_off.$(date +%Y%m%d-%H%M%S)

[ -f "$GRUB_FILE" ] || { echo "no $GRUB_FILE — not an Ubuntu/Debian-style GRUB host" >&2; exit 1; }

# Per-variable rewrite: drop any iommu=* token, then `GRUB_CMDLINE_LINUX`
# gets `iommu=off` appended. `GRUB_CMDLINE_LINUX_DEFAULT` just gets the
# token dropped (no add).
rewrite_var() {
  local var="$1" add_iommu_off="$2"
  local current_line current_val
  current_line=$(grep "^${var}=" "$GRUB_FILE" | head -1 || true)
  [ -n "$current_line" ] || return 0
  current_val=$(printf '%s\n' "$current_line" \
    | sed -E "s/^${var}=\"?(.*)\"?\$/\1/" \
    | sed -E 's/"$//')
  local new_tokens=()
  local tok
  for tok in $current_val; do
    case "$tok" in
      iommu=*) ;;
      "")      ;;
      *) new_tokens+=("$tok") ;;
    esac
  done
  if [ "$add_iommu_off" -eq 1 ]; then
    new_tokens+=("iommu=off")
  fi
  printf '%s="%s"\n' "$var" "${new_tokens[*]}"
}

# Ubuntu convention: `iommu=*` goes on GRUB_CMDLINE_LINUX (the always-applied
# line). Strip the token from _DEFAULT defensively too, in case the previous
# tweak script left a residue.
PROPOSED_LINUX=$(rewrite_var GRUB_CMDLINE_LINUX         1)
PROPOSED_DEFAULT=$(rewrite_var GRUB_CMDLINE_LINUX_DEFAULT 0)
CURRENT_LINUX=$(grep "^GRUB_CMDLINE_LINUX="         "$GRUB_FILE" | head -1 || true)
CURRENT_DEFAULT=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" "$GRUB_FILE" | head -1 || true)

show_diff() {
  local label="$1" current="$2" proposed="$3"
  if [ -z "$current" ]; then
    echo "=== $label: not present (skipped)"
    return
  fi
  echo "=== $label"
  echo "  current:  $current"
  echo "  proposed: $proposed"
  if [ "$current" = "$proposed" ]; then echo "  (unchanged)"; fi
}
show_diff "GRUB_CMDLINE_LINUX"         "$CURRENT_LINUX"   "$PROPOSED_LINUX"
echo
show_diff "GRUB_CMDLINE_LINUX_DEFAULT" "$CURRENT_DEFAULT" "$PROPOSED_DEFAULT"
echo
echo "Effect on /proc/cmdline after reboot:"
echo "  any iommu=* token replaced with iommu=off (kernel suppresses AMD-Vi)"
echo

NEED_LINUX=0
NEED_DEFAULT=0
[ -n "$CURRENT_LINUX"   ] && [ "$CURRENT_LINUX"   != "$PROPOSED_LINUX"   ] && NEED_LINUX=1
[ -n "$CURRENT_DEFAULT" ] && [ "$CURRENT_DEFAULT" != "$PROPOSED_DEFAULT" ] && NEED_DEFAULT=1

if [ "$NEED_LINUX" -eq 0 ] && [ "$NEED_DEFAULT" -eq 0 ]; then
  echo "Already in target state — nothing to do."
  exit 0
fi

if [ "$APPLY" -ne 1 ]; then
  echo "DRY RUN — nothing written. Re-run with: sudo $0 --apply"
  exit 0
fi

if [ "$EUID" -ne 0 ]; then
  echo "--apply requires root. Re-run with sudo." >&2; exit 1
fi

echo "Backing up $GRUB_FILE → $BACKUP"
cp -p "$GRUB_FILE" "$BACKUP"

TMP=$(mktemp)
awk \
  -v new_linux="$PROPOSED_LINUX" \
  -v new_default="$PROPOSED_DEFAULT" '
  /^GRUB_CMDLINE_LINUX=/         { if (new_linux   != "") { print new_linux;   next } }
  /^GRUB_CMDLINE_LINUX_DEFAULT=/ { if (new_default != "") { print new_default; next } }
  { print }
' "$GRUB_FILE" > "$TMP"
chmod --reference="$GRUB_FILE" "$TMP"
mv "$TMP" "$GRUB_FILE"

echo "=== /etc/default/grub now contains:"
grep -E '^GRUB_CMDLINE_LINUX(_DEFAULT)?=' "$GRUB_FILE"
echo
echo "=== Running update-grub:"
update-grub
echo
echo "=== Done. Reboot for the new cmdline to take effect."
echo "    After reboot: cat /proc/cmdline | grep iommu  → should show iommu=off"
echo "                   ls /sys/kernel/iommu_groups/    → should be empty"
echo "    Rollback: sudo cp $BACKUP $GRUB_FILE && sudo update-grub"
