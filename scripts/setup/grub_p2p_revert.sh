#!/usr/bin/env bash
# grub_p2p_revert.sh — undo grub_p2p_tweaks.sh and restore the high-decode
# intra-die TP2 config that we measured at ~37 tok/s on Qwen3.6-27B-Q4_0
# before the cross-die-P2P tuning session.
#
# Two actions:
#   1) Edit /etc/default/grub to remove `iommu=off`, `pcie_aspm=off`,
#      `pci=pcie_bus_perf` from BOTH `GRUB_CMDLINE_LINUX` and
#      `GRUB_CMDLINE_LINUX_DEFAULT`, and restore `iommu=pt` (the original
#      Ubuntu default) on `GRUB_CMDLINE_LINUX`. Reboot required.
#   2) Restore the per-MI50 power cap to 250 W (`power1_cap_max`) via
#      hwmon sysfs — runtime, no reboot needed. Yesterday's power-cap
#      sweep showed 250 W gave the best intra-die TP2 decode (37.25 t/s).
#
# What this does NOT touch:
#   - The peer-access constant fix in crates/backend-hip/src/sys.rs (705 → 704):
#     keep it. That is a real bug fix, not a TP4-stability hack — reverting it
#     would silently re-disable BAR1 P2P AR on every multi-GPU run.
#   - The HipKernel::launch / launch_raw kernel-name-in-error-message patch:
#     pure diagnostic, no perf impact.
#   - The `tp4_sum_f32_correctness_sweep_and_latency` test (#[ignore]'d):
#     same — documentary, doesn't affect anything when ignored.
#
# What it CANNOT touch (must be done manually in BIOS):
#   - Memory Interleaving — set back to its original value (likely "Auto"
#     or "Channel", not "Die"). Yesterday's intra-die TP2 numbers were
#     measured with default channel interleaving.
#   - IOMMU — restore to the BIOS default (likely Enabled).
#   - Any other BIOS toggles touched during this session.
#
# Default mode is DRY-RUN. Re-run with --apply to actually edit.

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
BACKUP=/etc/default/grub.bak.revert.$(date +%Y%m%d-%H%M%S)

[ -f "$GRUB_FILE" ] || { echo "no $GRUB_FILE — not an Ubuntu/Debian-style GRUB host" >&2; exit 1; }

# --- Per-variable revert ----------------------------------------------------
# revert_var <varname> <restore_iommu_pt:0|1>
# Reads current `<varname>="..."` line, drops iommu=*, pcie_aspm=*,
# pci=pcie_bus_* tokens, optionally appends `iommu=pt`. Prints the
# proposed replacement line. Empty output = variable not present.
revert_var() {
  local var="$1" restore_pt="$2"
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
  if [ "$restore_pt" -eq 1 ]; then
    new_tokens+=("iommu=pt")
  fi
  printf '%s="%s"\n' "$var" "${new_tokens[*]}"
}

# Ubuntu convention: GRUB_CMDLINE_LINUX is where IOMMU / boot-critical tokens
# go (applied to every entry incl. recovery); GRUB_CMDLINE_LINUX_DEFAULT is
# for non-essential normal-boot tokens (`quiet splash`). So `iommu=pt` is
# restored only on GRUB_CMDLINE_LINUX.
PROPOSED_LINUX=$(revert_var GRUB_CMDLINE_LINUX 1)
PROPOSED_DEFAULT=$(revert_var GRUB_CMDLINE_LINUX_DEFAULT 0)
CURRENT_LINUX=$(grep "^GRUB_CMDLINE_LINUX=" "$GRUB_FILE" | head -1 || true)
CURRENT_DEFAULT=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT=" "$GRUB_FILE" | head -1 || true)

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
echo "=== Action summary:"
echo "  drop:    iommu=*, pcie_aspm=*, pci=pcie_bus_* from both lines"
echo "  restore: iommu=pt on GRUB_CMDLINE_LINUX"
echo "  power:   set every MI50's power1_cap to 250 W via hwmon (runtime)"
echo

NEED_LINUX_EDIT=0
NEED_DEFAULT_EDIT=0
[ -n "$CURRENT_LINUX"   ] && [ "$CURRENT_LINUX"   != "$PROPOSED_LINUX"   ] && NEED_LINUX_EDIT=1
[ -n "$CURRENT_DEFAULT" ] && [ "$CURRENT_DEFAULT" != "$PROPOSED_DEFAULT" ] && NEED_DEFAULT_EDIT=1

# --- Show current MI50 power caps for context ------------------------------
echo "=== Current per-MI50 power caps:"
for bdf in $(lspci -D -d 1002:66a1 2>/dev/null | awk '{print $1}'); do
  for hwmon in /sys/bus/pci/devices/$bdf/hwmon/hwmon*; do
    cap=$(cat $hwmon/power1_cap 2>/dev/null)
    max=$(cat $hwmon/power1_cap_max 2>/dev/null)
    [ -n "$cap" ] && echo "  $bdf cap=$((cap/1000000))W max=$((max/1000000))W"
  done
done
echo

if [ "$NEED_LINUX_EDIT" -eq 0 ] && [ "$NEED_DEFAULT_EDIT" -eq 0 ]; then
  echo "GRUB already at the reverted state — only power-cap restoration would run."
fi

if [ "$APPLY" -ne 1 ]; then
  echo
  echo "DRY RUN — nothing written. Re-run with: sudo $0 --apply"
  exit 0
fi

# --- Apply ------------------------------------------------------------------
if [ "$EUID" -ne 0 ]; then
  echo "--apply requires root. Re-run with sudo." >&2; exit 1
fi

# 1) GRUB
if [ "$NEED_LINUX_EDIT" -eq 1 ] || [ "$NEED_DEFAULT_EDIT" -eq 1 ]; then
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
else
  echo "(GRUB unchanged — skipping update-grub)"
fi

# 2) Power cap → 250 W. Best-effort: skip any card that doesn't support it.
echo "=== Setting per-MI50 power cap to 250 W:"
for bdf in $(lspci -D -d 1002:66a1 2>/dev/null | awk '{print $1}'); do
  for hwmon in /sys/bus/pci/devices/$bdf/hwmon/hwmon*; do
    cap_file="$hwmon/power1_cap"
    max_file="$hwmon/power1_cap_max"
    [ -w "$cap_file" ] || continue
    max=$(cat "$max_file" 2>/dev/null)
    target=250000000
    if [ -n "$max" ] && [ "$max" -lt "$target" ]; then
      target="$max"
    fi
    if echo "$target" > "$cap_file" 2>/dev/null; then
      new=$(cat "$cap_file" 2>/dev/null)
      echo "  $bdf → $((new/1000000))W"
    else
      echo "  $bdf  failed to set power cap (driver may not allow this card)"
    fi
  done
done
echo
echo "=== Done."
echo "    GRUB:   reboot required for the cmdline change to take effect."
echo "    Power:  applied immediately; resets to driver default on reboot."
echo "    BIOS:   restore Memory Interleaving and IOMMU manually if changed."
echo "    Code:   peer-access constant fix (sys.rs:25 = 704) and kernel-name"
echo "            patch in module.rs are KEPT — they are real fixes, not tweaks."
echo "    Rollback: sudo cp $BACKUP $GRUB_FILE && sudo update-grub"
