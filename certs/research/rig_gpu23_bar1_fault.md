# RIG-EXPLORE — MI50 BAR1 {2,3} fault: precedents and workarounds

## Symptom signature recap

On a 4× MI50 PCIe 3.0 x16 rig (no xGMI), `hipDeviceCanAccessPeer` and
`hipDeviceEnablePeerAccess` succeed for every off-diagonal pair, but a
kernel-launched cross-rank BAR1 read+sum hits HIP error 719 ("kfd device
kernel invalid") only when GPUs 2 and 3 are both in the world. After such
a failure, further AR stress on `{2,3}` corrupts kfd state until the host
is rebooted (`hipGetDeviceCount` returns 0 on all cards). Identical
behaviour across ROCm 6.x and 7.x.

## Matched reports

- [MI50 32GB p2p not working — ROCm/ROCm#4793 — 2025](https://github.com/ROCm/ROCm/issues/4793):
  Dual MI50-32 on Supermicro X11SRA-F. `rocm-bandwidth-test` shows
  inter-device access flagged 1 yet bidirectional bandwidth is N/A —
  exact pattern of "canAccessPeer says yes, real BAR1 traffic dies".
  Reproduces on ROCm 5.7.1 and 6.4.0; AMD has it under investigation.
- [GPU Peer-to-Peer crashes machine — ROCm/ROCm#1495 — 2021](https://github.com/ROCm/ROCm/issues/1495):
  RX Vega 64 + Radeon VII (gfx906 sibling). `rocm-bandwidth-test` peer
  run produces SDMA `SRBM_WRITE non-privilege buffer` page faults, GPU
  resets, then "Failed to initialize parser -125" cascades — the same
  shape as our kfd-corrupt-after-failure cascade.
- [Poor bandwidth on dual Radeon Pro VII — ROCm/rocm_smi_lib#110 — 2021](https://github.com/ROCm/rocm_smi_lib/issues/110):
  Two Pro VIIs, topology says XGMI/1-hop/weight 15 and `link accessible`,
  yet RCCL collapses to 9.86 GB/s vs 285 GB/s single-GPU. Same
  "topology lies" symptom; AMD labelled "Under Investigation" with no
  resolution. Direct match for the secondary `peer_access_full=false`
  reordering quirk.
- [llama.cpp segfault on multi-AMD-GPU — ggml-org/llama.cpp#17583 — 2025](https://github.com/ggml-org/llama.cpp/issues/17583):
  4× Radeon Pro VII (gfx906) on AsRock H510 PRO BTC+. `llama-server`
  segfaults the moment a second GPU is engaged, even when the model
  fits on one card. Survives BIOS bumps, ROCm 6.4.4 → 7.1.x, OS
  reinstall. Matches our "harmless when sharded onto one card,
  catastrophic when peer DMA crosses ranks" shape on a consumer board.
- [llama.cpp `--split-mode row` requires P2P — ahmadosman.com — 2024](https://www.ahmadosman.com/blog/do-not-use-llama-cpp-or-ollama-on-multi-gpus-setups-use-vllm-or-exllamav2/):
  Independent confirmation that on PCIe-only AMD rigs, kernel-launched
  P2P access is the hot landmine; vLLM avoids it via RCCL-over-shmem.
  Same class of failure as ours; the workaround (route the AR through
  RCCL/host-staged) is the structural escape valve.
- [ROCm BAR Memory troubleshooting — official docs](https://rocm.docs.amd.com/en/latest/how-to/Bar-Memory.html):
  AMD's own guidance: P2P DMA "only works when one device can directly
  access the local BAR memory of another. If the memory address of a
  BAR exceeds the physical addressing limit of a device, the device
  will not be able to access that BAR." This is the exact mechanism
  for a per-pair failure: only the pair whose BARs land outside each
  other's reachable window dies.
- [16× MI50 Qwen3.5-397B setup guide — ai-infos](https://github.com/ai-infos/guidances-setup-16-mi50-qwen35-397b/tree/main):
  The most battle-tested multi-MI50 cmdline in the wild:
  `pcie_ports=native pci=realloc=on pciehp.pciehp_force=1
  pci=assign-busses pci=hpmmioprefsize=128G pci=hpmemsize=128G
  pci=hpmmiosize=16G pci=hpiosize=4M pci=hpbussize=16` plus
  `iommu=pt`. They also pin BIOS PCIe to Gen3 explicitly — relevant
  because flaky autonegotiation on slot 3 is a known way to silently
  brown-out one pair.
- [MI50 best-AI-card-for-beginners — Willy Tarreau, 2025](http://wtarreau.blogspot.com/2025/12/amd-radeon-instinct-mi50-32gb-best-ai.html):
  Confirms Above-4G Decoding as a non-negotiable for MI50. Author had
  to swap motherboards entirely on rigs whose BIOS lacked the toggle —
  symptom there is "no boot", but the underlying constraint is the
  same MMIO-aperture issue that makes a single pair fail when the
  toggle is on but the high-MMIO window is misplaced.

## Root-cause theories ranked by evidence

1. **Bad PCIe link or addressing window on the slot hosting GPU 3 (or
   GPU 2) — most likely.** AMD's BAR doc says peer DMA fails when
   either device's BAR lands outside the *other* device's reachable
   range. On a consumer board with 4 slots sharing one CPU root
   complex, the BIOS may place GPU 3's 32 GiB BAR above GPU 2's
   reachable window even though BARs 0/1 (the small windows used by
   `canAccessPeer` probing) are fine. Issues #4793 and #110 both show
   topology APIs reporting healthy while real DMA dies — same
   signature. Counter-evidence: would normally affect any pair
   involving the misplaced card, not specifically `{2,3}`. The
   "kernel-launched read+sum" shape (vs SDMA copy) explains the
   asymmetry: the kernel does not go through SDMA's BAR-range
   sanity-check, so it faults at first cache-line miss rather than at
   `enablePeerAccess` time.
2. **Above-4G / MMIO-high aperture misplaced.** Even with the toggle
   on, BIOSes commonly pick a high-MMIO base above 2^44, which gfx906
   cannot address (BAR doc explicitly cites a 44-bit example). The
   IFB-bridge dual-Pro-VII case (#110) lines up cleanly with this.
3. **Faulty riser/cable on slot 3.** A re-driver or marginal-trace
   riser produces exactly this shape — `lspci` link-up, training at
   x16, but CRC errors under sustained DMA. Issue #17583's reporter
   tried 4 GPUs on a mining board (notorious for Gen3 redrivers) with
   the same blanket-multi-GPU-segfault. Counter-evidence for our rig:
   only `{2,3}` fails; risers usually take the whole card down.
4. **Motherboard PCIe lane allocation / shared root complex.** If
   slots 2 and 3 share a PCIe switch downstream of the root, the
   switch may not advertise full peer-DMA routing even though the
   root does. The 16× MI50 guide's explicit `pcie1 = x4x4x4x4` and
   ROMED8-2T BIOS recipe exists precisely to defuse this on
   consumer-adjacent boards. Counter-evidence: would normally affect
   any pair through the switch, not just `{2,3}`.
5. **VBIOS mismatch between GPUs 2 and 3.** A search hit flagged this
   for mixed MI50/Radeon VII setups; less likely if all four cards
   are nominally identical, but worth a `rocm-smi --showvbios` cross-
   check before swapping silicon.
6. **Genuine card-level fault on GPU 2 or 3 (BAR-aperture decoder
   damaged).** Last on the list because we have no per-card stress
   evidence yet — but slot-swap test will collapse this and theory 3
   into a single answer.

## Workarounds tried in the wild

| Workaround | Source | Worked? | Notes |
|---|---|---|---|
| `iommu=pt` on GRUB cmdline | ROCm docs, MultiAMDGPU guide, 16× MI50 guide | Yes for hangs in RCCL collectives | First thing to try; cheap, reversible |
| `pci=realloc=on` + `pci=assign-busses` + `pci=hpmmiosize=16G` etc. | 16× MI50 guide | Yes for boards that mis-place BARs at boot | Forces kernel to re-assign BAR windows; exactly the lever for theory 1/2 |
| Above-4G Decoding ON in BIOS | Tarreau, ROCm BAR doc | Yes (mandatory) | Without it the card won't even POST; with it, the *placement* still matters |
| MMIO High Base/Size pinned below 2^44 | ROCm BAR doc | Yes when BIOS exposes the knob | Very board-specific; consumer BIOSes often don't surface it |
| `rmmod amdgpu && modprobe amdgpu` | MultiAMDGPU guide | Sometimes (recovers from reset cascades, not from kfd-process death) | After our `{2,3}` corruption the kfd process itself is wedged — this often *won't* recover, reboot is needed |
| ROCm version pin (5.7.1 / 6.4.4) | Tarreau, mixa3607/ML-gfx906 | Sometimes | gfx906 is in maintenance mode since ROCm 6.0; 7.x is unofficially supported by copying tensile files. Behaviour identical for #4793 across 5.7.1↔6.4.0, so version-pinning is unlikely to fix our specific BAR1 fault |
| Slot swap (move suspect card to a known-good slot) | implicit in #1495, #17583 follow-ups | Diagnostic, not a fix | Decisive for separating card-fault from slot-fault |
| RCCL over shared memory / host-bounce instead of kernel P2P | vLLM, ahmadosman blog | Yes — structural workaround | This is what flambeau already does on PP hand-off; the production decode path therefore stays alive even if `{2,3}` BAR1 P2P never works |
| ACS override patch | not seen in any matched gfx906 P2P thread | Unknown for this symptom | Irrelevant unless IOMMU groups are blocking peer access, which is not our reported failure mode |

## Recommended next test on this rig

**Physically swap the GPUs in slots 2 and 3 with two known-good cards
(currently in slots 0 and 1), keep BIOS and cmdline identical, and
re-run the bracket sweep.**

Rationale: the matched evidence puts theory 1 ("bad slot/riser/BAR
window for the slot 3 position") and theory 6 ("bad card") at the top,
and they are the only two remaining hypotheses that the existing
software-only diagnostics (the bracket sweep, the link-probe, ROCm
version flips) cannot separate. ROCm/ROCm#4793 and rocm_smi_lib#110
both show the topology APIs lying about a hardware-level fault, which
is exactly what we are seeing — the answer comes from moving silicon,
not from another software run. If the failure follows the cards into
slots 0/1, it is a card-level BAR-decoder fault on GPU 2 or 3; if it
stays on slots 2/3 with healthy cards in them, it is a board/riser
fault localised to that slot pair. Either result tells us whether to
RMA cards or reroute the rig topology, and costs only a power-down +
two cable swaps. Before swapping, capture
`sudo lspci -vv -s <bdf>` for all four cards and a full
`dmesg | grep -E 'amdgpu|BAR|pci'` for the baseline so the post-swap
delta is unambiguous. Concurrent with the swap, set
`pci=realloc=on iommu=pt` on the GRUB cmdline (16× MI50 guide
recipe) — these are zero-risk and remove theory 2/4 from the hypothesis
list whichever way the swap goes.
