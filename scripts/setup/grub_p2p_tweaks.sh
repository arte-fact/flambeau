#!/usr/bin/env bash
# grub_p2p_tweaks.sh — add kernel cmdline params for cross-die GPU P2P stability
# on the 4-MI50 Threadripper rig.
#
# Edits BOTH `GRUB_CMDLINE_LINUX` and `GRUB_CMDLINE_LINUX_DEFAULT` in
# /etc/default/grub so the kernel cmdline contains:
#   - iommu=off            (set; replaces any iommu=* including iommu=pt)
#   - pcie_aspm=off        (force PCIe links to stay up; no ASPM gating)
#   - pci=pcie_bus_perf    (force max common MPS across topology)
# Both variables are processed because:
#   - GRUB_CMDLINE_LINUX         is applied to ALL boot entries (normal + recovery)
#   - GRUB_CMDLINE_LINUX_DEFAULT is applied only to non-recovery entries
# `iommu=pt` is typically set in the former on Ubuntu hosts that enable
# IOMMU at install time; editing only `_DEFAULT` left `iommu=pt` in
# /proc/cmdline alongside the new `iommu=off` (kernel last-wins parsing
# meant `iommu=off` still applied, but the duplication is ugly and brittle).
#
# Why each param:
#   iommu=off          — empty /sys/kernel/iommu_groups/ confirmed AMD-Vi is
#                        not registering devices anyway; fully off removes any
#                        residual amd_iommu codepath inspecting DMA paths.
#   pcie_aspm=off      — disables PCIe link power management. ASPM idles links
#                        during sub-µs gaps; under bursty 4-way AR traffic the
#                        wake-from-L1 latency causes jitter on cross-die hops.
#   pci=pcie_bus_perf  — picks the largest MaxPayloadSize common to every device
#                        in each PCI hierarchy. Bigger MPS = fewer TLPs per AR
#                        payload = less per-TLP overhead on the Infinity Fabric
#                        crossing.
#
# Default mode is DRY-RUN. Re-run with --apply to actually edit + update-grub.
# A reboot is required for the changes to take effect; the script does NOT
# reboot for you.

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
BACKUP=/etc/default/grub.bak.$(date +%Y%m%d-%H%M%S)

# --- Sanity checks -----------------------------------------------------------
[ -f "$GRUB_FILE" ] || { echo "no $GRUB_FILE — not an Ubuntu/Debian-style GRUB host" >&2; exit 1; }

# --- Per-variable token rewrite ---------------------------------------------
# rewrite_var <varname>
# Reads the current `<varname>="..."` line from $GRUB_FILE, drops any
# iommu=*, pcie_aspm=*, pci=pcie_bus_* tokens, appends the canonical set,
# and prints the proposed replacement line. If the variable is absent
# from the file, prints nothing (caller treats as no-op).
rewrite_var() {
  local var="$1"
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
      iommu=*)          ;;
      pcie_aspm=*)      ;;
      pci=pcie_bus_*)   ;;
      "")               ;;
      *) new_tokens+=("$tok") ;;
    esac
  done
  new_tokens+=("iommu=off")
  new_tokens+=("pcie_aspm=off")
  new_tokens+=("pci=pcie_bus_perf")
  printf '%s="%s"\n' "$var" "${new_tokens[*]}"
}

# Compute proposed lines for both variables.
PROPOSED_DEFAULT=$(rewrite_var GRUB_CMDLINE_LINUX_DEFAULT)
PROPOSED_LINUX=$(rewrite_var GRUB_CMDLINE_LINUX)
CURRENT_DEFAULT=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" "$GRUB_FILE" | head -1 || true)
CURRENT_LINUX=$(grep "^GRUB_CMDLINE_LINUX=" "$GRUB_FILE" | head -1 || true)

# --- Show the change --------------------------------------------------------
show_diff() {
  local label="$1" current="$2" proposed="$3"
  if [ -z "$current" ]; then
    echo "=== $label: not present in $GRUB_FILE (skipped)"
    return
  fi
  echo "=== $label"
  echo "  current:  $current"
  echo "  proposed: $proposed"
  if [ "$current" = "$proposed" ]; then
    echo "  (unchanged)"
  fi
}
show_diff "GRUB_CMDLINE_LINUX"         "$CURRENT_LINUX"   "$PROPOSED_LINUX"
echo
show_diff "GRUB_CMDLINE_LINUX_DEFAULT" "$CURRENT_DEFAULT" "$PROPOSED_DEFAULT"
echo
echo "=== Canonical tokens appended/normalised on each line:"
echo "  iommu=off, pcie_aspm=off, pci=pcie_bus_perf"
echo "  (any pre-existing iommu=*, pcie_aspm=*, pci=pcie_bus_* dropped)"
echo

NEED_LINUX_EDIT=0
NEED_DEFAULT_EDIT=0
[ -n "$CURRENT_LINUX"   ] && [ "$CURRENT_LINUX"   != "$PROPOSED_LINUX"   ] && NEED_LINUX_EDIT=1
[ -n "$CURRENT_DEFAULT" ] && [ "$CURRENT_DEFAULT" != "$PROPOSED_DEFAULT" ] && NEED_DEFAULT_EDIT=1

if [ "$NEED_LINUX_EDIT" -eq 0 ] && [ "$NEED_DEFAULT_EDIT" -eq 0 ]; then
  echo "Already up to date — no changes to apply."
  exit 0
fi

if [ "$APPLY" -ne 1 ]; then
  echo "DRY RUN — nothing written. Re-run with: sudo $0 --apply"
  exit 0
fi

# --- Apply ------------------------------------------------------------------
if [ "$EUID" -ne 0 ]; then
  echo "--apply requires root. Re-run with sudo." >&2; exit 1
fi

echo "Backing up $GRUB_FILE → $BACKUP"
cp -p "$GRUB_FILE" "$BACKUP"

# Rewrite both lines atomically: build a temp file with awk substituting
# each line iff a proposed replacement exists, then mv into place.
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
echo "=== After reboot, verify with: cat /proc/cmdline"
echo "    (Expect: iommu=off pcie_aspm=off pci=pcie_bus_perf; NO iommu=pt)"
echo "=== Rollback (if needed): sudo cp $BACKUP $GRUB_FILE && sudo update-grub"
