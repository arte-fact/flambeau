# X399 Threadripper — slot-2↔3 P2P fault: platform-topology analysis

**Companion to** `rig_gpu23_bar1_fault.md` (card-vs-slot triage, closed 2026-04-27 by physical swap — fault stayed with slots, not silicon).

**Question:** *given the fault is slot-bound, what about the X399 platform makes specifically the {slot 2, slot 3} pair fail when {0,1}, {0,2}, {1,3} all work?*

This document collects what's publicly known about X399 PCIe topology and ranks platform-level hypotheses, each with a concrete on-rig diagnostic.

---

## 1. X399 reference topology (what every consumer X399 board has in common)

- **CPU side (Zen 1 / Zen+ Threadripper):** 60 PCIe 3.0 lanes from the SoC, organised as **4 root complexes of x16**, one per Zeppelin die quadrant. The two-die MCM means **lanes are physically wired per-die: 32 lanes from die 0, 32 lanes from die 1** ([anandtech]).
- **Chipset (Promontory / X399 PCH):** 4 PCIe 3.0 lanes uplink to CPU; downstream a small fan-out of 8 PCIe 2.0 lanes plus USB/SATA/etc. The chipset owns at most one of the visible x16-physical slots on most boards (and often *zero* — chipset lanes go to the small x1/x4 slots and M.2 only).
- **Hard-wired slot config on every standard X399 board:** `x16 / x8 / x16 / x8` ([tweaktown buyers guide]). The four "x16 mechanical" slots map onto two die boundaries:
  - Slots 1 + 2 (top-half) → one Zeppelin die's 32-lane budget (24 to slots, 8 to M.2/U.2).
  - Slots 3 + 4 (bottom-half) → the other die's 32-lane budget (24 to slots, 8 to chipset/M.2).
- **Inter-die path:** all cross-die PCIe peer traffic goes over **Infinity Fabric On-Package (IFOP)** at ~50 GB/s nominal ([wikichip IF, hot chips 2017]). On Zen 1/Zen+ this is the same fabric that carries cross-die memory traffic, and is well known to be the chief NUMA penalty on this CPU.

### Why this matters for peer DMA

`hipDeviceEnablePeerAccess` succeeding does not mean the path is reliable. The kernel-launched BAR1 read+sum does not go through SDMA (which has its own routing sanity-check) — it issues PCIe TLPs that are routed by the **PCIe root that owns the *initiator* BAR window**. If the target BAR is on a different root complex (different die), the TLP must:

1. Climb to the initiator die's root complex,
2. Cross the **die-to-die IF coherent link**,
3. Descend into the target die's root complex,
4. Reach the target's BAR.

This path is documented to work — `{0,2}` and `{1,3}` do work on this rig — but it is also documented to be the most fragile path on Threadripper Gen 1/Gen 2 (`Threadripper & PCIe Bus Errors` thread on Level1Techs ran for 50+ pages without a clean fix; partial mitigations were *Gen2 link-speed downgrade* or *ASPM disable*, neither acceptable for our throughput targets — [level1techs PCIe-bus-errors]).

---

## 2. Why specifically `{2,3}` could be the failing pair

Three structural explanations, all consistent with `{0,1}` `{0,2}` `{1,3}` healthy.

### Hypothesis A — **slot 2 and slot 3 sit on opposite sides of the die boundary, *and* one of them is downstream of a PCIe quick-switch / bridge that drops peer-DMA TLPs**. ★ leading

X399 boards routinely deploy ASMedia / Pericom **PCIe quick switches** to multiplex bottom-half x8 lanes between a slot and the U.2 connector ([tweaktown buyers guide]: *"the only two reasons quick switches are seen on an X399 motherboard are for switching bandwidth to and from a U.2 port, or next to a clock generator"*). The switch is transparent for normal upstream traffic but, on at least some implementations, **does not advertise PCIe ACS `EgressControl` / `DirectTranslated` capabilities** required for non-root peer routing.

Effect: enabling peer access succeeds (root sees the BAR), but a peer TLP that has to go *through* the switch in the lateral (downstream-to-downstream-via-bridge) direction is silently dropped or NAK'd, surfacing as `unspecified launch failure (719)` once the kernel actually touches the remote line. Pairs that route through the root *without* the switch in the lateral path (e.g. `{1,3}` where each card is upstream of its own die's root) work fine because the TLP path is `slot → root → IF → root → slot` and never re-enters a downstream switch.

This is the same mechanism that makes `pcie_acs_override=downstream` "work" for some passthrough cases ([heiko-sieger ACS]), and it is the mechanism that the official ROCm guidance warns about: *"ACS forces P2P transactions through the PCIe root complex … the disable-ACS script should be run prior to any workloads"* ([rocm-radeon mGPU]).

**Diagnostic on rig:**
```bash
sudo lspci -tv                       # Tree view: is one of {slot2, slot3} downstream of an extra bridge?
sudo lspci -vvv -s <bdf-gpu-2> | grep -E 'ACSCap|ACSCtl|LnkCap|LnkSta'
sudo lspci -vvv -s <bdf-gpu-3> | grep -E 'ACSCap|ACSCtl|LnkCap|LnkSta'
# Look for: ACSCap with SrcValid+TransBlk+ReqRedir+CmpltRedir set, ACSCtl matching.
# Asymmetry between cards 2 and 3 in ACSCap is the smoking gun.
```
If the lspci tree shows GPU 3 (or 2) downstream of an ASMedia / Pericom bridge that the others don't share, that bridge is the suspect. Test the official ROCm `disable-ACS` script (linked from [rocm-radeon mGPU]) and re-run the `{2,3}` bracket. Or set `pcie_acs_override=downstream,multifunction` in GRUB as a *diagnostic only* — it's a security regression and should not ship.

### Hypothesis B — **`{2,3}` is the only pair where the lateral peer path crosses *both* the die boundary AND the chipset uplink, because slot 3 is a chipset-fed slot on this board**.

Most consumer X399 boards put all four x16-mechanical slots on CPU lanes, but a minority of OEM / "test" boards have surfaced where the lowest x16 slot is actually a chipset x4-in-x16 slot. The Promontory X399 chipset has a 4-lane PCIe 3.0 uplink to one of the dies and **does not validate cross-chipset peer-DMA**: peer TLPs going from a CPU-attached slot to a chipset-attached slot have to climb to the CPU root, cross the chipset uplink, traverse the chipset's internal PCIe switch, and reach the target — a path that is functionally never tested by AMD or the chipset vendor for sustained GPU peer-DMA. ROCm explicitly says: *"only PCIe slots connected by the CPU should be used, avoiding PCIe slots connected via chipset"* ([rocm-radeon mGPU]). A chipset-attached slot that gets `canAccessPeer = 1` is not officially supported and can fail exactly like this.

**Diagnostic on rig:**
```bash
sudo lspci -t                        # See which root each GPU sits under.
sudo lspci -nn | grep -E '1022:14|1022:43'   # 1022:43xx are X399 chipset bridges.
# If one of GPU 2/3 sits on a 1022:43xx downstream port, hypothesis B is alive.
```

### Hypothesis C — **board-level PCB fault localised to a specific lane group / repeater on the slot-2↔slot-3 trace**.

Cross-die peer traffic between physical-adjacent slots routes the most signal length on most X399 PCBs (longest trace, often through an external Gen3 redriver IC). A single bad redriver, cracked solder joint on a CLK_REQ pin, or a damaged decoupling cap on the slot-3 lanes can pass POST-time link training (which uses few lanes at low speed) but corrupt sustained x16 traffic. Issue [llamacpp#17583] on a Pro VII rig surveyed this exact failure mode after multi-card brought down peer DMA on a mining-style board.

**Diagnostic on rig:**
```bash
# Peer load test on the failing pair while watching AER:
sudo dmesg --follow | grep -iE 'aer|pcie|amdgpu' &
# In another shell, run the bracket-sweep w=2{2,3} again.
# Look for "Bad TLP" / "Bad DLLP" / "Replay timer" / "Non-Fatal" AER prints
# strictly during the run, not at idle.
sudo lspci -vv -s <bdf-gpu-3> | grep -E 'CESta|UESta'   # Correctable / Uncorrectable counts
```
If correctable-error counts on slot 3 climb monotonically during a `{2,3}` run but stay flat for `{0,2}` or `{1,3}`, the trace / redriver is failing under sustained DMA on that specific pair.

### Hypothesis D — **BIOS placed slot-2's BAR1 outside slot-3's reachable address window (or vice-versa), but only for that pair**.

Already covered as theory 1 in `rig_gpu23_bar1_fault.md`. The slot-swap result *partially* survives this: even with cards swapped, BIOS is likely to assign BARs *by slot* (deterministic on most BIOSes), so the same physical card now in slot 3 still gets a BAR window above slot 2's reachable range. ROCm's BAR doc spells this out: gfx9 needs the target's BAR `< 2^44` to be peer-readable ([rocm BAR doc]). If the BIOS places one of the bottom-half slots' 32 GB BAR above 2^44 because high-MMIO base is misplaced, *that* slot's card is the one nobody else can read — and `{2,3}` is the only pair where *both* cards live in the bottom half.

**Diagnostic on rig:**
```bash
sudo lspci -vv -s <bdf-gpu-2> | grep 'Region 0\|Memory at'
sudo lspci -vv -s <bdf-gpu-3> | grep 'Region 0\|Memory at'
# Look at the upper 64-bit BAR (the "32G/64G prefetchable" line).
# Compare the assigned base address against 2^44 = 0x100000000000.
# If GPU 3's BAR base is ≥ 2^44 (0x100000000000), other gfx9 cards cannot read it.
```
Cheap fix to attempt: GRUB `pci=realloc=on pci=hpmmiosize=16G pci=hpmemsize=128G` (the 16× MI50 recipe — see prior doc), forces kernel to re-pick BAR placement under 2^44. If `{2,3}` recovers after a `pci=realloc` boot, hypothesis D is confirmed.

---

## 3. The "obscure AMD test board" angle

The user describes the board as an obscure AMD X399 *test board*. There is no widely-documented "AMD reference X399" board sold at retail; possibilities:

- An **AMD-internal QA / validation board** issued to ISVs, occasionally surfacing on second-hand markets.
- A **server / SP3r2 sibling** board (TR4 socket but routed for 4S workstation rather than gaming).
- An OEM board (e.g. **Tyan TR4**, **Supermicro M11SDV**-style) marketed primarily to embedded / engineering customers.

Each of these is more likely than retail consumer boards to have **non-standard slot routing** — specifically, a chipset-attached x16 slot, an unbuffered PCB run that relies on signal integrity that consumer boards spec around with redrivers, or an unusual BIOS that doesn't expose the *Above-4G Decoding* / *MMIO High Base* knobs. All of these enlarge hypotheses B and D.

**Concrete asks for the user:**
1. Run `sudo dmidecode -t baseboard` and `sudo dmidecode -t bios` and paste the manufacturer / product / BIOS vendor / BIOS version. The board model is the single biggest unblocker for narrowing hypotheses.
2. Run `sudo lspci -tvvv` and grep the GPU BDFs — knowing the topology tree (which roots each GPU sits under, whether any sit below a chipset bridge or extra PCIe switch) cuts the hypothesis space immediately.
3. Photograph the BIOS *PCIe Configuration* page (or describe the available toggles): *Above-4G Decoding*, *MMIO High Base*, *Re-Size BAR Support*, *PCIe Link Speed override*, *ACS / SR-IOV*. The presence/absence of these and their current values tell us whether D is reachable as a software fix.

---

## 4. Ranked hypothesis summary

| # | Hypothesis                                                                              | Evidence weight | Fix path                                                                  |
|---|-----------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------|
| A | Lateral peer-routing through an ACS-incomplete quick-switch / bridge on slot 2 or 3     | High            | Run ROCm `disable-acs`; if it recovers, ship with that script in setup    |
| D | BIOS BAR placement above 2^44 for the bottom-half slot pair                             | High            | `pci=realloc=on pci=hpmmiosize=...` on GRUB; or fix in BIOS                |
| B | One of the slots is chipset-attached, not CPU-attached                                  | Medium          | Move both cards to top-half slots (1 + 2), avoid bottom-half pairing      |
| C | PCB / redriver fault on the slot-3 lane group only under sustained DMA                  | Medium          | AER counter monitoring during stress; if confirmed, hardware-side (board) |
| — | Cross-die IF reliability under sustained peer-DMA (Threadripper-1/2 platform-wide flaw) | Background      | Already covered by flambeau's host-bounce PP; never put `{2,3}` in TP world |

---

## 5. Pragmatic posture for flambeau given current rig

Independent of root cause, flambeau already has the right architectural escape: PP hand-off uses **pinned-host bounce** rather than kernel-launched peer DMA, which is why decode on `Mesh<4>` works at all on this rig despite the `{2,3}` peer link being broken. The two things to ship:

1. **Topology guard in `runtime::Mesh<N>` build:** when a 4-rank mesh is requested, run a *short* peer-DMA smoke on every off-diagonal pair *before* loading the model. If any pair fails, refuse to build the world-4 mesh and fall back to world-2 on `{0,1}`, `{0,2}`, or `{1,3}`. (This avoids the kfd-corruption cascade after a real run trips the bad link.)
2. **Honour `FLAMBEAU_MESH_PIN={0,1}` or `{0,2}` or `{1,3}`** for cert / perf runs on this rig until the link is fixed at the hardware level. Already in the project memory as the recommended posture.

The hardware-side resolution requires either (i) confirming and fixing one of A/B/D via BIOS/GRUB knobs, or (ii) sourcing a standard X399 board with a documented PCIe layout and rerouting cards. Until then `{2,3}` should be considered a permanent excluded pairing.

---

## References

- [AMD ThreadRipper: X399, 16C/32T, 64 PCIe lanes and more — TweakTown](https://www.tweaktown.com/news/57819/amd-threadripper-x399-16c-32t-64-pcie-lanes-more/index.html)
- [An AMD Threadripper X399 Motherboard Overview — AnandTech](https://www.anandtech.com/show/11685/amd-threadripper-x399-motherboards)
- [AMD X399 TR4 Threadripper Motherboard Buyer's Guide — TweakTown](https://www.tweaktown.com/guides/8342/amd-x399-tr4-threadripper-motherboard-buyers-guide/index.html) — confirms x16/x8/x16/x8 hard-wiring and quick-switch usage
- [Threadripper & PCIe Bus Errors — Level1Techs](https://forum.level1techs.com/t/threadripper-pcie-bus-errors/118977) — multi-page documentation of unresolved cross-die PCIe issues, attempted ASPM-disable / Gen2-downgrade workarounds
- [Infinity Fabric — WikiChip](https://en.wikichip.org/wiki/amd/infinity_fabric) — die-to-die IFOP fabric structure
- [Hot Chips 2017: AMD Outlines Threadripper And EPYC's MCM — Tom's Hardware](https://www.tomshardware.com/news/amd-threadripper-epyc-mcm-cost,35306.html) — original disclosure of two-die MCM and per-die PCIe controllers
- [mGPU setup and configuration — ROCm on Radeon](https://rocm.docs.amd.com/projects/radeon/en/latest/docs/install/native_linux/mgpu.html) — *"only PCIe slots connected by the CPU"*, ACS guidance, large BAR requirement
- [How ROCm uses PCIe Atomics — ROCm docs](https://rocm.docs.amd.com/en/latest/understand/More-about-how-ROCm-uses-PCIe-Atomics.html) — gfx9 BAR < 2^44 requirement
- [IOMMU Groups — What You Need to Consider — Heiko Sieger](https://www.heiko-sieger.info/iommu-groups-what-you-need-to-consider/) — ACS forces P2P through root, override risks
- [MI50 32GB p2p not working — ROCm/ROCm#4793](https://github.com/ROCm/ROCm/issues/4793) — direct symptom match (canAccessPeer=1, BW=N/A) on Intel X11SRA-F, unresolved
- [P2P support — ROCm/ROCm#787](https://github.com/ROCm/ROCm/issues/787) — historical P2P slot-specificity issue thread
- [Threadripper / Vega Reset Bug — Level1Techs](https://forum.level1techs.com/t/threadripper-vega-reset-bug/128058) — Vega-on-X399 multi-card issues, related class of fault
- [llama.cpp segfault on multi-AMD-GPU — ggml-org/llama.cpp#17583](https://github.com/ggml-org/llama.cpp/issues/17583) — multi-card peer DMA failure on consumer board
- [ROCm BAR Memory troubleshooting — official docs](https://rocm.docs.amd.com/en/latest/how-to/Bar-Memory.html) — peer-DMA-fails-when-BAR-out-of-range mechanism

---

## Addendum (2026-04-27, post-topology-dump) — hypothesis space collapsed

User ran `dmidecode -t baseboard` and `lspci -tvvv`. Decisive new information.

### Board confirmed: AMD-internal Whitehaven validation board

```
Manufacturer: AMD
Product Name: Whitehaven OPS rev B
Serial Number: 105-D10200-00A
```

*Whitehaven* is AMD's internal codename for the Zen-1 Threadripper platform; *OPS rev B* + the `105-D10200` part-number prefix identify this as an AMD operations / validation reference board, not a retail SKU. This is a literal AMD-internal X399 test board — vanishingly few are in the wild, no consumer BIOS support, no review coverage. Implication: there is no community BIOS / vendor support channel; whatever AMD's internal validation team pinned in BIOS *is* the BIOS.

### Topology, decoded

The lspci tree shows **two NUMA root complexes**, `[0000:00]` and `[0000:40]` — one per Zeppelin die, NPS2 enabled. The four MI50s land like this:

| logical | BDF       | sits under root | which die |
|---------|-----------|-----------------|-----------|
| GPU 0   | 0a:00.0   | `[0000:00]/01.3` | die 0     |
| GPU 1   | 0d:00.0   | `[0000:00]/03.1` | die 0     |
| GPU 2   | 44:00.0   | `[0000:40]/01.3` | die 1     |
| GPU 3   | 47:00.0   | `[0000:40]/03.1` | die 1     |

Other notes from the tree:
- All four GPUs sit behind the standard Zeppelin three-bridge chain (`01.3-[08-0a]----00.0-[09-0a]----00.0-[0a]`). No ASMedia / Pericom / PLX quick-switch anywhere. **Hypothesis A (lateral-routing-through-quick-switch) is killed.**
- All four GPUs are on **CPU lanes**, not chipset lanes. The X370/X399 PCH is on `[0000:00]/01.1` and only fans out to USB / SATA / two NICs and the audio codec. **Hypothesis B (chipset-attached slot) is killed.**

### The actual pattern, and why it's surprising

Re-reading the bracket sweep against the new die assignment:

| pair  | die mapping        | result |
|-------|--------------------|--------|
| {0,1} | die 0 + die 0      | works  |
| {0,2} | die 0 + die 1      | works  |
| {1,3} | die 0 + die 1      | works  |
| {2,3} | die 1 + die 1      | **fails** |

The failing pair is **the only intra-die-1 pair**. Both intra-die-0 (`{0,1}`) and both cross-die pairs (`{0,2}`, `{1,3}`) survive sustained peer DMA. The fault is *not* the cross-die Infinity Fabric link (that path is clean); it's the **intra-die peer-routing on die 1 specifically**.

This is a meaningful narrowing because cross-die IF was the natural Threadripper-platform suspect (Level1Techs PCIe-bus-errors thread, etc.). Cross-die IF *is not the problem here*. The problem is localised to whatever path connects the two PCIe controllers (`01.3` and `03.1`) **inside die 1's local Data Fabric**, when both endpoints under those controllers try to peer with each other.

### Revised hypothesis ranking

| # | Hypothesis                                                                                                                                                                                                                                                                                                                | Weight | Why                                                                                                                                                                                  |
|---|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|--------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| E | **Die-1 intra-die peer-DMA path defective** at silicon (Data Fabric) or PCB level (clock / VRM / trace shared by both die-1 PCIe lane groups). The same `01.3 ↔ 03.1` topology works on die 0, so it's die-1-localised — not generic to the silicon design.                                                              | High   | Symmetry of evidence: only intra-die-1 fails. Both die-0 controllers and both cross-die paths are clean. Pattern is unique to die 1's local fabric.                                  |
| D | **BIOS placed both die-1 GPU BARs in a window where they cannot read each other (e.g. one above 2^44, gfx9 ceiling).** Die-0 BARs land in a different (peer-readable) range, hence `{0,1}` works; cross-die pairs work because at least one side is in the readable die-0 range.                                          | Medium | The Whitehaven BIOS is internal AMD code with unknown PCIe-BAR allocation policy. `lspci -vv` BAR dump on all four cards (vs 0x100000000000 = 2^44) is a one-command test.            |
| C | **PCB-level fault on a clock / power / reference signal shared by die-1's two slot pairs only**, manifesting under sustained peer DMA. Less likely than E for a board that survived AMD validation, more likely than usual because Rev B → there were earlier revs and quirks may have shipped.                          | Medium | Survived card-swap (rules out silicon-on-card fault). AER counter monitoring during a `{2,3}` stress is the test (instructions in §2 above).                                          |
| —  | A, B, ACS-quick-switch, chipset-attached slot                                                                                                                                                                                                                                                                            | Killed | Topology dump shows neither structure exists on this board.                                                                                                                          |

### Recommended next diagnostics in priority order

1. **BAR placement check (cheapest, kills/confirms D):**
   ```bash
   for bdf in 0a:00.0 0d:00.0 44:00.0 47:00.0; do
     echo "=== $bdf ==="
     sudo lspci -vv -s $bdf | grep -E '^[[:space:]]+(Region|Memory at).*(prefetchable|64-bit)' | head -4
   done
   ```
   Compare the upper 64-bit BARs (the 32 GB prefetchable region) against `2^44 = 0x100000000000`. If GPU 2 or GPU 3's BAR sits above that threshold, the other one cannot peer-read it. Try `pci=realloc=on pci=hpmmiosize=16G pci=hpmemsize=128G iommu=pt` on GRUB cmdline and re-bracket. If `{2,3}` recovers, D is the answer and shipping a GRUB cmdline closes it permanently.

2. **AER counter delta during a `{2,3}` stress (kills/confirms C):**
   ```bash
   for bdf in 0a:00.0 0d:00.0 44:00.0 47:00.0; do
     echo -n "$bdf CESta:  "; sudo lspci -vv -s $bdf | grep -E 'CESta'
   done
   # Run the failing bracket-sweep w=2{2,3} test (subprocess-isolated per the link-probe-safety memory).
   # Re-read CESta on all four cards.
   ```
   If `44:00.0` or `47:00.0` shows monotonically growing correctable-error counts during the run while the others stay flat, hypothesis C is confirmed and the next step is hardware-side (board RMA, but for a Whitehaven OPS rev B that means whoever provisioned the rig — there is no AMD consumer-RMA path).

3. **If both above are clean → hypothesis E is the residual.** No software fix exists for a Data-Fabric-localised peer-routing defect. Mitigations:
   - Pin `Mesh<2>` to `{0,1}` (intra-die-0, matched-die memory-bandwidth, lowest-latency peer DMA).
   - Or `{0,2}` / `{1,3}` if a cross-die config is needed for any other reason. Avoid `{2,3}`.
   - For `Mesh<4>`, route all peer-DMA via host-bounce (already the flambeau PP default). Add a topology-guard at `Mesh<4>` build that smoke-tests every off-diagonal pair before model load and refuses to build the world if any pair fails.

### Why this is consistent with the board being a Whitehaven OPS rev B

The OPS-rev-B identifier strongly suggests this board went through *engineering* validation, not the *consumer* validation pass that retail X399 boards (Asus Zenith, Gigabyte Designare, MSI MEG, ASRock Taichi) all underwent. Two consequences:

- BIOS PCIe-BAR allocation policies on internal validation boards are routinely *less conservative* than retail BIOSes — the team was likely chasing performance, not Vega-class large-BAR portability — making D more plausible than on a retail board.
- A defective intra-die peer path is *exactly* the kind of fault an OPS validation board would surface, get noted in errata, and *not* get a board respin for. Rev-B already implies an earlier rev existed; whether the underlying issue was ever judged worth a respin is unknown without internal AMD docs.

### Bottom line

The previous report's leading hypotheses (A: ACS-incomplete switch; B: chipset-attached slot) are both refuted by the topology dump. The new leading hypothesis is **E (die-1 intra-fabric defect)** with **D (BAR-placement above 2^44 for die-1 slots)** as a cheap-to-test fallback that, if true, is fixable from the GRUB cmdline. Run the BAR check first; it's a 30-second one-liner and either fixes the rig outright or eliminates the only software-tractable hypothesis still standing.

---

## Addendum 2 (2026-04-27, BAR dump) — hypothesis D also killed; only C and E remain

User ran the BAR check. Result:

| GPU | die | Region 0 (BAR0) base   | size | base in decimal | vs 2^44 ceiling (`0x100000000000`, 16 TB) |
|-----|-----|------------------------|------|-----------------|-------------------------------------------|
| 0   | 0   | `0xf800000000`         | 16 G | ~992 GiB        | far below (94×)                           |
| 1   | 0   | `0xf000000000`         | 16 G | ~960 GiB        | far below (97×)                           |
| 2   | 1   | `0x7c00000000`         | 16 G | ~496 GiB        | far below (188×)                          |
| 3   | 1   | `0x7400000000`         | 16 G | ~464 GiB        | far below (200×)                          |

All four BARs are at least two orders of magnitude below the gfx9 ceiling. Peer-readability across the 64-bit prefetchable windows is unobstructed in the address-decode sense. **Hypothesis D is killed.** No GRUB / BIOS knob will rescue this rig.

Two side notes from the dump:

- **BARs split cleanly by die** (die 0 around the 1 TB mark, die 1 around 0.5 TB). The kernel allocated each die's PCIe windows from its own NUMA node memory map — confirms NPS2 mode is active. Not a fault by itself, just nails the topology.
- **All BAR0 sizes are 16 G** — full HBM directly mapped. The cards on this rig are the **MI50 16 GB SKU** (the original release; the 32 GB SKU came later); the 16 GiB BAR0 exposes the entire HBM with no windowing. No ReBAR concerns, no upper-half access penalty. This rules out one possible asymmetric-cost mechanism cleanly.

### Final hypothesis state

| # | Hypothesis                                                                                                          | State        | Test left? |
|---|---------------------------------------------------------------------------------------------------------------------|--------------|------------|
| A | ACS-incomplete quick-switch / bridge                                                                                | Killed by tree dump | — |
| B | Chipset-attached slot                                                                                                | Killed by tree dump | — |
| D | BIOS placed a BAR above 2^44                                                                                         | Killed by BAR dump  | — |
| C | PCB-level fault on die-1 lane group (clock / VRM / trace shared by both die-1 slots)                                 | **Live**     | AER counter delta during `{2,3}` stress |
| E | Die-1 intra-fabric peer-routing defect at silicon (Data Fabric crossbar in die 1)                                    | **Live**     | Residual after C is ruled out |

The two remaining hypotheses are **both hardware-side**. There is no software / BIOS / GRUB lever left that can fix this. The next informative test is the AER counter delta — if correctable-error counts on `44:00.0` and/or `47:00.0` climb during a `{2,3}` stress and stay flat at idle, hypothesis C is confirmed (PCB-localised, in principle reworkable). If AER stays clean and the launch still fails, hypothesis E is the residual (silicon-level Data-Fabric defect, no fix).

**For both C and E the operational mitigation is identical:** treat `{2,3}` as a permanent excluded pairing. Pin `Mesh<2>` to `{0,1}` (best — intra-die-0, lowest peer-DMA latency, no cross-die hop), `{0,2}`, or `{1,3}`. Run `Mesh<4>` only via host-bounce PP hand-off (already the flambeau default), with a topology-guard that smoke-tests every off-diagonal pair before model load.

Given this is an AMD-internal Whitehaven OPS rev B board with no consumer support channel, even confirming C via AER does not lead to a tractable repair — the *practical* posture is to ship the topology guard and move on.

### One last diagnostic worth running (for the record)

If the user wants to definitively close the C-vs-E question:

```bash
# Snapshot AER counters before:
for bdf in 0a:00.0 0d:00.0 44:00.0 47:00.0 \
           0000:40:01.3 0000:40:03.1 0000:40:00.0 \
           0000:00:01.3 0000:00:03.1 0000:00:00.0; do
  echo -n "$bdf  "
  sudo lspci -vv -s $bdf 2>/dev/null | grep -E 'CESta:|UESta:' | tr '\n' ' '
  echo
done > /tmp/aer-before.txt

# Run the failing bracket:
cargo test -p qwen3-moe --release tp_w2_pair_2_3_smoke -- --nocapture
# (or the equivalent harness that fires the {2,3} bracket; per
# feedback_link_probe_safety, run it subprocess-isolated with bounded iters
# so a hang doesn't wedge kfd)

# Snapshot AER counters after (same command, redirect to /tmp/aer-after.txt).
diff /tmp/aer-before.txt /tmp/aer-after.txt
```

The bridges to watch most carefully are `[0000:40]/01.3` and `[0000:40]/03.1` — the two die-1 GPU root ports. CESta / UESta growth on either of these *during* the test (and not at idle) confirms PCB-level signal-integrity fault on die-1 slot routing. Flat AER + persistent crash points at silicon.

Either way, the answer doesn't change what flambeau ships — but it would close the file on this rig.

---

## Addendum 3 (2026-04-27) — NVMe co-load detail reframes the hypothesis

User reports: **system crashes happen when GPUs *and* NVMe simultaneously have high activity**, not just on `{2,3}` peer DMA in isolation.

The lspci tree shows the Samsung NVMe sits on **die 1's root complex**, sibling controller to GPU 2:

```
[0000:40]   ← die 1
  +-01.2-[41]   Samsung NVMe         ← die 1, controller 01.2
  +-01.3-[42-44] GPU 2               ← die 1, controller 01.3 (sibling)
  +-03.1-[45-47] GPU 3               ← die 1, controller 03.1
```

Three high-bandwidth devices all on die 1's fabric. A *deterministic silicon defect* (hypothesis E) would not be load-correlated — it would fail the same way every time the path is used. A *load-correlated crash* points at a shared resource saturating: voltage rail, clock domain, or I/O-hub buffer.

The most common shape of this on Threadripper Gen 1/2 is **SoC voltage / Infinity-Fabric clock instability**: undervoltage on the SoC rail under sustained fabric load corrupts TLPs, leading to bad-DLLP / replay-timer-expired / unspecified-launch-failure cascades. The fix is BIOS-side: bump SoC voltage or lower FCLK.

**Updated hypothesis ranking:**

| # | Hypothesis                                                         | Weight    | Tractable fix?                       |
|---|---------------------------------------------------------------------|-----------|--------------------------------------|
| F (new) | SoC voltage / FCLK undervolt under die-1 fabric load          | **High**  | BIOS — bump SoC voltage +25-50 mV, or lower FCLK one notch |
| C | Load-dependent fault on die-1 lanes (PCB SI / power)               | **High**  | BIOS — Gen 2 force on die-1 slots (perf hit but shippable) |
| E | Silicon-level intra-fabric defect on die 1                          | Medium    | None (board respin)                  |

**Whitehaven OPS BIOS hunt list (top 3 to try first):**

1. `Advanced > AMD CBS > NBIO Common Options > SoC Voltage` — bump +25 mV from default. Single most likely fix for load-correlated IF/PCIe crashes on Threadripper 1/2.
2. `Advanced > AMD CBS > NBIO Common Options > FCLK Frequency` (or `Infinity Fabric Frequency`) — lower one notch (1600 → 1467 → 1333). Reduces IF stability margin demand.
3. `Advanced > AMD CBS > DF Common Options > Memory Addressing > Memory Interleaving` — try `Channel` (UMA) instead of `Die` (NUMA). Coalesces traffic, can dodge die-1 overload.

Tier-2 options (if Tier 1 doesn't shift it): force PCIe Gen 2 per slot on the die-1 GPUs (confirmation test for C); disable ASPM; enable PCIe AER explicitly; raise slot power limits; disable LCLK DPM. Full table in the conversation handoff.

**OS-side mitigation to try in parallel** (cheap, reversible — set on GRUB cmdline):
```
iommu=pt pci=realloc=on pcie_aer=on
```

`iommu=pt` removes IOMMU translation overhead on DMA paths (lower latency, less buffer pressure). `pci=realloc=on` lets the kernel re-pick BAR placement if BIOS picks a marginal one. `pcie_aer=on` ensures the kernel logs AER events to dmesg so the `aer-watch.sh` script captures real signal integrity faults rather than just sticky bits the firmware may have eaten.

If hypothesis F is correct, a +25 mV SoC bump alone should make `{2,3}` stop crashing under co-load with the NVMe. That would be the cleanest possible result and would unblock `Mesh<4>` for production decode on this rig. Worth testing before accepting the permanent-exclusion posture.

---

## Addendum 4 (2026-04-27) — TP4 vs PP+NVMe are two distinct faults

User clarification: **the NVMe-correlated crash happens during PP inference; TP4 inference crashes deterministically and is unrelated to NVMe activity**.

This splits the problem cleanly:

### Mode A — TP4 (always crash, NVMe-independent)

- TP4 uses kernel-launched peer DMA across all 4 ranks, including the `{2,3}` leg.
- Deterministic failure → rules out load / voltage / contention as the *primary* cause for this mode.
- This is the pure `{2,3}` peer-routing fault in isolation: the `01.3 ↔ 03.1` intra-die-1 path is broken at the path level.
- **Hypothesis E (silicon-level intra-fabric defect on die 1) returns to top.** Hypothesis C (PCB SI on die-1 lanes) is also still viable but would more typically be load-correlated; deterministic-always failure is more consistent with a silicon path being unable to route at all.
- BIOS SoC/FCLK knobs are unlikely to help here — the path is structurally broken, not marginally unstable.
- **Only mitigation: exclude `{2,3}` from any TP world.** TP world=2 on `{0,1}`, `{0,2}`, `{1,3}` already works per the bracket sweep; TP4 is permanently unsupported on this rig.

### Mode B — PP + NVMe co-load (stochastic crash, load-correlated)

- PP uses pinned-host-bounce ping-pong (DtoH/HtoD via CPU memory). The `{2,3}` peer link is **not** on the PP hot path.
- Crashes only when NVMe is also active. NVMe sits on die-1 root `[0000:40]/01.2`, sibling to GPU 2's `01.3`.
- Two GPUs (PP hand-offs touching all 4 cards) + NVMe high-bandwidth I/O all sharing die 1's data fabric and memory controller is a classic Threadripper-1/2 saturation pattern.
- **Hypothesis F (SoC undervolt / FCLK margin under combined load) is the right target.** Tier-1 BIOS hunt list applies here.

### Two flambeau-side checks before touching BIOS (Mode B only)

1. **NUMA affinity of pinned bounce buffers.** If flambeau's pinned-host buffers all live on die 0 (or are unspecified, in which case Linux defaults vary), GPUs on die 1 cross-IF on every PP hand-off — *while* the NVMe on die 1 competes on die 1's memory controller. Test: run flambeau under `numactl --interleave=all <command>` and see if Mode B clears. If it does, the fix is per-rank pinned-buffer allocation on the GPU's local NUMA node.
2. **Move the NVMe to a die-0 M.2 slot** if the chassis exposes one. The lspci tree shows die-0 root ports (`[0000:00]/01.2` path family) that aren't currently used. Moving the NVMe off die 1 isolates disk I/O from PP-hand-off fabric entirely. Cheapest physical fix; reboot only.

### Updated mitigation posture

| Workload     | Status on this rig                                                               |
|--------------|----------------------------------------------------------------------------------|
| Mesh<1>      | Always works on any single GPU.                                                   |
| Mesh<2> {0,1} | Works (intra-die-0). Best for cert / perf runs.                                 |
| Mesh<2> {0,2} or {1,3} | Works (cross-die, healthy IF path).                                     |
| Mesh<2> {2,3} | **Permanently unsupported** (Mode A path defect).                                |
| Mesh<4> PP   | Works *if* NVMe is idle. To make it robust, fix Mode B (BIOS SoC/FCLK or NUMA pinning of bounce buffers, or move NVMe to die 0). |
| Mesh<4> TP   | **Permanently unsupported** (Mode A; uses the broken `{2,3}` peer leg).          |

Mode A is a card-of-deck dropped — accept it. Mode B is the tractable fight: it's the difference between "Mesh<4> PP works in idle conditions" and "Mesh<4> PP works while a model is loading from disk in parallel". Worth resolving for production serve workloads where model swap / KV checkpoint hits the NVMe concurrent with decode traffic.
