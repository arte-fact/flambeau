
## code, agent Agent 1 (µs per cycle; V2 ['v2a', 'v2b'] vs V3 ['v3a', 'v3b'])
| quantité | V2 | V3 | Δ |
|---|---|---|---|
| wall | 90401.9 | 93478.5 | +3076.6 |
| busy | 72345.3 | 74665.0 | +2319.7 |
| launch | 2188.2 | 2248.2 | +60.0 |
| idle (wall−busy) | 18056.7 | 18813.6 | |
| attention (lanc. 25.0→25.0) | 2009.0 | 2019.3 | +10.2 |
| comm (lanc. 137.4→137.4) | 389.2 | 373.7 | -15.5 |
| gdn (lanc. 144.4→144.4) | 2422.2 | 2364.1 | -58.1 |
| iq4_xs_mmq (lanc. 0.1→0.1) | 52.4 | 52.2 | -0.2 |
| iq4_xs_mmvq (lanc. 117.0→117.0) | 13422.8 | 13422.1 | -0.7 |
| other (lanc. 1026.2→1026.3) | 8930.2 | 8921.9 | -8.3 |
| q4_k_mmq (lanc. 0.1→100.7) | 76.4 | 15489.2 | +15412.9 |
| q4_k_mmvq (lanc. 103.0→2.4) | 13581.2 | 286.1 | -13295.1 |
| q5_k_mmq (lanc. 0.8→1.0) | 591.8 | 676.4 | +84.6 |
| q5_k_mmvq (lanc. 131.0→131.0) | 14271.4 | 14284.3 | +13.0 |
| q6_k_mmq (lanc. 0.2→0.2) | 105.1 | 109.0 | +3.9 |
| q6_k_mmvq (lanc. 59.8→59.8) | 13304.7 | 13272.9 | -31.8 |
| q8_0_mmq (lanc. 101.9→102.0) | 1738.0 | 1747.9 | +10.0 |
| q8_0_mmvq (lanc. 8.0→8.0) | 68.9 | 68.8 | -0.1 |
| q8_0_rowtile (lanc. 4.4→4.4) | 105.1 | 106.0 | +0.8 |
| quant_ds4 (lanc. 12.9→92.0) | 48.3 | 333.4 | +285.1 |
| quant_q8_1 (lanc. 316.0→296.5) | 1228.6 | 1137.7 | -90.9 |

### Q4_K et quantification par noyau/grille (lancements/cycle, médiane µs)
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('163840', '1', '256') : 42.98/cyc × 117.6 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('278528', '1', '256') : 27.35/cyc × 183.4 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('98304', '1', '256') : 23.44/cyc × 78.1 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('196608', '1', '256') : 5.86/cyc × 133.1 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 123.07/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 58.60/cyc × 4.8 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 46.88/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('163840', '1', '256') : 1.02/cyc × 108.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.26/cyc × 3.5 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('278528', '1', '256') : 0.65/cyc × 164.9 µs
- v2a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('8704', '4', '64') : 0.07/cyc × 947.4 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 15.98/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('16384', '1', '256') : 0.98/cyc × 48.0 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 11.93/cyc × 3.7 µs
- v2a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('5120', '4', '64') : 0.09/cyc × 468.2 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('98304', '1', '256') : 0.56/cyc × 69.8 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 7.95/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 7.81/cyc × 3.0 µs
- v2a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('3072', '4', '64') : 0.05/cyc × 456.0 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 5.44/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('196608', '1', '256') : 0.14/cyc × 119.8 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 3.98/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 3.91/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 2.42/cyc × 3.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('34816', '1', '32') : 1.40/cyc × 4.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 1.21/cyc × 3.8 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '46', '128') : 0.60/cyc × 6.4 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('12288', '1', '32') : 1.12/cyc × 3.3 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '46', '128') : 0.28/cyc × 9.4 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('16384', '1', '256') : 0.02/cyc × 44.5 µs
- v2a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '46', '128') : 0.19/cyc × 5.1 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('23552', '1', '32') : 0.19/cyc × 4.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.23/cyc × 3.5 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '4', '128') : 0.19/cyc × 3.1 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.14/cyc × 3.8 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.12/cyc × 4.3 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '46', '128') : 0.09/cyc × 5.0 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '4', '128') : 0.09/cyc × 4.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.09/cyc × 3.0 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.05/cyc × 3.8 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('5120', '4', '64') : 42.98/cyc × 144.6 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('8704', '4', '64') : 27.35/cyc × 161.4 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('3072', '4', '64') : 23.44/cyc × 135.5 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('6144', '4', '64') : 5.86/cyc × 144.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 118.19/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 72.28/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 48.84/cyc × 4.8 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 42.00/cyc × 3.5 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('512', '4', '64') : 0.98/cyc × 137.9 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('163840', '1', '256') : 1.02/cyc × 108.2 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.26/cyc × 3.5 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('278528', '1', '256') : 0.65/cyc × 164.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 15.98/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 13.67/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 11.93/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('98304', '1', '256') : 0.56/cyc × 69.9 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 7.95/cyc × 3.7 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('5120', '4', '64') : 0.05/cyc × 478.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 5.44/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('196608', '1', '256') : 0.14/cyc × 119.8 µs
- v3a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '5', '128') : 4.88/cyc × 3.4 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 3.98/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('3072', '4', '64') : 0.02/cyc × 457.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 2.42/cyc × 3.5 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('34816', '1', '32') : 1.40/cyc × 4.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 1.21/cyc × 3.8 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('12288', '1', '32') : 1.12/cyc × 3.5 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '46', '128') : 0.47/cyc × 6.3 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '46', '128') : 0.21/cyc × 9.4 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('16384', '1', '256') : 0.02/cyc × 44.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('23552', '1', '32') : 0.19/cyc × 4.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.23/cyc × 3.5 µs
- v3a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '46', '128') : 0.14/cyc × 5.1 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '4', '128') : 0.19/cyc × 3.5 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.12/cyc × 4.3 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '46', '128') : 0.09/cyc × 5.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.14/cyc × 3.3 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '4', '128') : 0.09/cyc × 4.4 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.09/cyc × 3.0 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.05/cyc × 3.7 µs

## code, agent Agent 2 (µs per cycle; V2 ['v2a', 'v2b'] vs V3 ['v3a', 'v3b'])
| quantité | V2 | V3 | Δ |
|---|---|---|---|
| wall | 90401.9 | 93478.5 | +3076.6 |
| busy | 72163.1 | 74517.4 | +2354.3 |
| launch | 2188.2 | 2248.2 | +60.0 |
| idle (wall−busy) | 18238.8 | 18961.1 | |
| attention (lanc. 25.0→25.0) | 2035.9 | 2028.4 | -7.6 |
| comm (lanc. 137.4→137.4) | 375.4 | 391.7 | +16.2 |
| gdn (lanc. 144.4→144.4) | 2384.4 | 2322.3 | -62.2 |
| iq4_xs_mmq (lanc. 0.1→0.1) | 52.3 | 52.0 | -0.3 |
| iq4_xs_mmvq (lanc. 117.0→117.0) | 13391.0 | 13392.0 | +1.0 |
| other (lanc. 1026.2→1026.3) | 8904.9 | 8904.2 | -0.7 |
| q4_k_mmq (lanc. 0.1→100.7) | 76.1 | 15466.7 | +15390.6 |
| q4_k_mmvq (lanc. 103.0→2.4) | 13529.5 | 286.0 | -13243.5 |
| q5_k_mmq (lanc. 0.8→1.0) | 591.1 | 675.2 | +84.2 |
| q5_k_mmvq (lanc. 131.0→131.0) | 14231.5 | 14227.7 | -3.8 |
| q6_k_mmq (lanc. 0.2→0.2) | 105.3 | 108.8 | +3.5 |
| q6_k_mmvq (lanc. 59.8→59.8) | 13318.2 | 13268.5 | -49.7 |
| q8_0_mmq (lanc. 101.9→102.0) | 1728.2 | 1756.7 | +28.5 |
| q8_0_mmvq (lanc. 8.0→8.0) | 69.9 | 69.4 | -0.5 |
| q8_0_rowtile (lanc. 4.4→4.4) | 105.4 | 105.4 | +0.0 |
| quant_ds4 (lanc. 12.9→92.0) | 49.1 | 329.4 | +280.3 |
| quant_q8_1 (lanc. 316.0→296.5) | 1214.9 | 1133.1 | -81.7 |

### Q4_K et quantification par noyau/grille (lancements/cycle, médiane µs)
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('163840', '1', '256') : 42.98/cyc × 116.8 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('278528', '1', '256') : 27.35/cyc × 182.6 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('98304', '1', '256') : 23.44/cyc × 77.8 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('196608', '1', '256') : 5.86/cyc × 133.1 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 123.07/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 58.60/cyc × 4.8 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 46.88/cyc × 3.5 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('163840', '1', '256') : 1.02/cyc × 107.6 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.26/cyc × 3.5 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('278528', '1', '256') : 0.65/cyc × 163.9 µs
- v2a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('8704', '4', '64') : 0.07/cyc × 945.8 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 15.98/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('16384', '1', '256') : 0.98/cyc × 47.2 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 11.93/cyc × 3.7 µs
- v2a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('5120', '4', '64') : 0.09/cyc × 467.3 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('98304', '1', '256') : 0.56/cyc × 70.2 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 7.95/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 7.81/cyc × 3.4 µs
- v2a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('3072', '4', '64') : 0.05/cyc × 454.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 5.44/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('196608', '1', '256') : 0.14/cyc × 119.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 3.98/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 3.91/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 2.42/cyc × 3.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('34816', '1', '32') : 1.40/cyc × 4.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 1.21/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '46', '128') : 0.60/cyc × 6.6 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('12288', '1', '32') : 1.12/cyc × 3.4 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '46', '128') : 0.28/cyc × 9.8 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('16384', '1', '256') : 0.02/cyc × 43.0 µs
- v2a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '46', '128') : 0.19/cyc × 5.1 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('23552', '1', '32') : 0.19/cyc × 4.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.23/cyc × 3.4 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '4', '128') : 0.19/cyc × 3.2 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.14/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.12/cyc × 4.3 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '46', '128') : 0.09/cyc × 5.3 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '4', '128') : 0.09/cyc × 4.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.09/cyc × 3.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.05/cyc × 3.7 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('5120', '4', '64') : 42.98/cyc × 144.3 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('8704', '4', '64') : 27.35/cyc × 160.8 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('3072', '4', '64') : 23.44/cyc × 135.2 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('6144', '4', '64') : 5.86/cyc × 144.2 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 118.19/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 72.28/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 48.84/cyc × 4.8 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 42.00/cyc × 3.4 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('512', '4', '64') : 0.98/cyc × 138.9 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.26/cyc × 3.5 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('163840', '1', '256') : 1.02/cyc × 107.4 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('278528', '1', '256') : 0.65/cyc × 164.9 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 15.98/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 13.67/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 11.93/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('98304', '1', '256') : 0.56/cyc × 70.1 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 7.95/cyc × 3.7 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('5120', '4', '64') : 0.05/cyc × 476.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 5.44/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('196608', '1', '256') : 0.14/cyc × 121.4 µs
- v3a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '5', '128') : 4.88/cyc × 3.4 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 3.98/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_q8_1 grid=('3072', '4', '64') : 0.02/cyc × 454.1 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 2.42/cyc × 3.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('34816', '1', '32') : 1.40/cyc × 4.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 1.21/cyc × 3.8 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('12288', '1', '32') : 1.12/cyc × 3.5 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '46', '128') : 0.47/cyc × 6.5 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '46', '128') : 0.21/cyc × 9.4 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n4 grid=('16384', '1', '256') : 0.02/cyc × 43.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('23552', '1', '32') : 0.19/cyc × 4.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.23/cyc × 3.5 µs
- v3a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '46', '128') : 0.14/cyc × 5.2 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '4', '128') : 0.19/cyc × 3.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.14/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.12/cyc × 4.3 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '46', '128') : 0.09/cyc × 5.2 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '4', '128') : 0.09/cyc × 4.4 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.09/cyc × 3.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.05/cyc × 3.7 µs

## math, agent Agent 1 (µs per cycle; V2 ['v2a', 'v2b'] vs V3 ['v3a', 'v3b'])
| quantité | V2 | V3 | Δ |
|---|---|---|---|
| wall | 89989.7 | 93636.7 | +3647.0 |
| busy | 72396.3 | 74719.0 | +2322.7 |
| launch | 2180.6 | 2240.3 | +59.7 |
| idle (wall−busy) | 17593.4 | 18917.8 | |
| attention (lanc. 25.0→24.9) | 3368.6 | 3384.5 | +15.9 |
| comm (lanc. 136.9→136.9) | 388.1 | 372.1 | -16.0 |
| gdn (lanc. 144.0→144.0) | 2375.2 | 2322.6 | -52.6 |
| iq4_xs_mmvq (lanc. 117.0→117.0) | 13404.6 | 13411.4 | +6.8 |
| other (lanc. 1022.7→1022.3) | 8839.3 | 8846.4 | +7.1 |
| q4_k_mmq (lanc. 0.0→101.9) | 0.0 | 15666.1 | +15666.1 |
| q4_k_mmvq (lanc. 103.0→1.1) | 13589.1 | 119.1 | -13470.0 |
| q5_k_mmvq (lanc. 131.0→131.0) | 14288.9 | 14330.2 | +41.2 |
| q6_k_mmq (lanc. 0.0→0.0) | 20.2 | 5.7 | -14.5 |
| q6_k_mmvq (lanc. 59.6→59.5) | 13202.3 | 13163.8 | -38.4 |
| q8_0_mmq (lanc. 102.9→102.9) | 1529.0 | 1517.8 | -11.2 |
| q8_0_mmvq (lanc. 8.1→8.1) | 70.1 | 69.8 | -0.3 |
| q8_0_rowtile (lanc. 2.8→2.8) | 69.8 | 69.4 | -0.3 |
| quant_ds4 (lanc. 12.0→92.2) | 42.1 | 326.0 | +284.0 |
| quant_q8_1 (lanc. 315.5→295.6) | 1209.1 | 1116.9 | -92.2 |

### Q4_K et quantification par noyau/grille (lancements/cycle, médiane µs)
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('163840', '1', '256') : 43.55/cyc × 117.3 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('278528', '1', '256') : 27.71/cyc × 183.2 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('98304', '1', '256') : 23.75/cyc × 77.9 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('196608', '1', '256') : 5.94/cyc × 133.0 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 124.70/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 59.38/cyc × 4.8 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 47.51/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.67/cyc × 3.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 17.32/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('163840', '1', '256') : 0.45/cyc × 105.5 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('16384', '1', '256') : 0.99/cyc × 47.8 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 12.06/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('278528', '1', '256') : 0.29/cyc × 149.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 8.08/cyc × 3.5 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 7.92/cyc × 3.2 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('98304', '1', '256') : 0.25/cyc × 63.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 4.04/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 3.96/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 1.89/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('196608', '1', '256') : 0.06/cyc × 112.0 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 1.26/cyc × 3.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 0.63/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('26112', '1', '32') : 0.62/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.59/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.45/cyc × 3.3 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('9216', '1', '32') : 0.49/cyc × 2.9 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.39/cyc × 3.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.23/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.20/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('16384', '1', '256') : 0.01/cyc × 36.2 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '67', '128') : 0.03/cyc × 8.3 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '3', '128') : 0.08/cyc × 3.0 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '3', '128') : 0.04/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('34304', '1', '32') : 0.02/cyc × 4.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '67', '128') : 0.01/cyc × 5.9 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('5120', '4', '64') : 43.55/cyc × 144.6 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('8704', '4', '64') : 27.71/cyc × 161.4 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('3072', '4', '64') : 23.75/cyc × 135.5 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('6144', '4', '64') : 5.94/cyc × 144.5 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 119.75/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 73.24/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 49.48/cyc × 5.0 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 42.56/cyc × 3.4 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('512', '4', '64') : 0.99/cyc × 138.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.67/cyc × 3.5 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 17.38/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 13.86/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('163840', '1', '256') : 0.45/cyc × 104.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 12.04/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('278528', '1', '256') : 0.29/cyc × 149.9 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 8.06/cyc × 3.5 µs
- v3a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '5', '128') : 4.95/cyc × 3.4 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('98304', '1', '256') : 0.25/cyc × 63.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 4.03/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('196608', '1', '256') : 0.06/cyc × 113.2 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 1.89/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 1.26/cyc × 3.4 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 0.63/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('26112', '1', '32') : 0.62/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.49/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.49/cyc × 3.5 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('9216', '1', '32') : 0.49/cyc × 2.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.33/cyc × 2.9 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.25/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.16/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('16384', '1', '256') : 0.01/cyc × 37.4 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '3', '128') : 0.08/cyc × 2.9 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '3', '128') : 0.04/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('34304', '1', '32') : 0.02/cyc × 4.6 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '67', '128') : 0.01/cyc × 6.1 µs

## math, agent Agent 2 (µs per cycle; V2 ['v2a', 'v2b'] vs V3 ['v3a', 'v3b'])
| quantité | V2 | V3 | Δ |
|---|---|---|---|
| wall | 89989.7 | 93636.7 | +3647.0 |
| busy | 72237.0 | 74602.1 | +2365.1 |
| launch | 2180.6 | 2240.3 | +59.7 |
| idle (wall−busy) | 17752.7 | 19034.6 | |
| attention (lanc. 25.0→24.9) | 3396.7 | 3394.7 | -2.0 |
| comm (lanc. 136.9→136.9) | 374.1 | 389.8 | +15.7 |
| gdn (lanc. 144.0→144.0) | 2373.5 | 2287.5 | -86.0 |
| iq4_xs_mmvq (lanc. 117.0→117.0) | 13371.5 | 13381.0 | +9.4 |
| other (lanc. 1022.7→1022.3) | 8827.7 | 8826.0 | -1.7 |
| q4_k_mmq (lanc. 0.0→101.9) | 0.0 | 15639.9 | +15639.9 |
| q4_k_mmvq (lanc. 103.0→1.1) | 13536.2 | 118.9 | -13417.3 |
| q5_k_mmvq (lanc. 131.0→131.0) | 14220.4 | 14296.1 | +75.8 |
| q6_k_mmq (lanc. 0.0→0.0) | 28.6 | 5.7 | -22.9 |
| q6_k_mmvq (lanc. 59.6→59.5) | 13197.9 | 13172.2 | -25.7 |
| q8_0_mmq (lanc. 102.9→102.9) | 1528.7 | 1519.8 | -9.0 |
| q8_0_mmvq (lanc. 8.1→8.1) | 70.7 | 70.2 | -0.5 |
| q8_0_rowtile (lanc. 2.8→2.8) | 69.6 | 69.4 | -0.2 |
| quant_ds4 (lanc. 12.0→92.2) | 43.1 | 321.6 | +278.6 |
| quant_q8_1 (lanc. 315.5→295.6) | 1198.3 | 1114.5 | -83.9 |

### Q4_K et quantification par noyau/grille (lancements/cycle, médiane µs)
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('163840', '1', '256') : 43.55/cyc × 116.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('278528', '1', '256') : 27.71/cyc × 182.4 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('98304', '1', '256') : 23.75/cyc × 77.8 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('196608', '1', '256') : 5.94/cyc × 132.6 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 124.70/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 59.38/cyc × 4.8 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 47.51/cyc × 3.2 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.67/cyc × 3.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 17.32/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('163840', '1', '256') : 0.45/cyc × 103.9 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n5 grid=('16384', '1', '256') : 0.99/cyc × 47.2 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 12.06/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('278528', '1', '256') : 0.29/cyc × 148.9 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 8.08/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 7.92/cyc × 3.4 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('98304', '1', '256') : 0.25/cyc × 63.2 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 4.04/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 3.96/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('196608', '1', '256') : 0.06/cyc × 112.5 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 1.89/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 1.26/cyc × 3.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 0.63/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('26112', '1', '32') : 0.62/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.59/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.45/cyc × 3.5 µs
- v2a quant_q8_1 flambeau_quantize_row_q8_1 grid=('9216', '1', '32') : 0.49/cyc × 2.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.39/cyc × 3.0 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.23/cyc × 3.7 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.20/cyc × 3.7 µs
- v2a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('16384', '1', '256') : 0.01/cyc × 38.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '3', '128') : 0.08/cyc × 3.1 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '67', '128') : 0.03/cyc × 8.2 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '3', '128') : 0.04/cyc × 3.7 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '67', '128') : 0.01/cyc × 13.3 µs
- v2a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('34304', '1', '32') : 0.02/cyc × 4.6 µs
- v2a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '67', '128') : 0.01/cyc × 6.4 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('5120', '4', '64') : 43.55/cyc × 144.3 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('8704', '4', '64') : 27.71/cyc × 161.0 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('3072', '4', '64') : 23.75/cyc × 135.2 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('6144', '4', '64') : 5.94/cyc × 144.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('25600', '1', '32') : 119.75/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '5', '128') : 73.24/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('43520', '1', '32') : 49.48/cyc × 4.8 µs
- v3a q4_k_mmq flambeau_mmq_q4_K_lds_tiled_y64_x8_q8_1 grid=('512', '4', '64') : 0.99/cyc × 139.3 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('15360', '1', '32') : 42.56/cyc × 3.2 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2560', '1', '32') : 31.67/cyc × 3.5 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('15360', '1', '32') : 17.38/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '5', '128') : 13.86/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('163840', '1', '256') : 0.45/cyc × 104.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('5120', '1', '32') : 12.04/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('278528', '1', '256') : 0.29/cyc × 149.0 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('512', '1', '32') : 8.06/cyc × 3.7 µs
- v3a quant_ds4 flambeau_quantize_q8_1_mmq grid=('3072', '5', '128') : 4.95/cyc × 3.4 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('98304', '1', '256') : 0.25/cyc × 63.3 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('3072', '1', '32') : 4.03/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('8704', '1', '32') : 3.98/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('196608', '1', '256') : 0.06/cyc × 112.6 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('20480', '1', '32') : 1.89/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('2048', '1', '32') : 1.26/cyc × 3.5 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('12288', '1', '32') : 0.63/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('26112', '1', '32') : 0.62/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('10240', '1', '32') : 0.49/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1536', '1', '32') : 0.49/cyc × 3.0 µs
- v3a quant_q8_1 flambeau_quantize_row_q8_1 grid=('9216', '1', '32') : 0.49/cyc × 2.9 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('1024', '1', '32') : 0.33/cyc × 2.9 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('9216', '1', '32') : 0.25/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_1 grid=('6144', '1', '32') : 0.16/cyc × 3.7 µs
- v3a q4_k_mmvq flambeau_mmvq_q4_k_dp4a_q8_1_batched_n3 grid=('16384', '1', '256') : 0.01/cyc × 37.8 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('5120', '3', '128') : 0.08/cyc × 3.2 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('8704', '3', '128') : 0.04/cyc × 3.7 µs
- v3a quant_q8_1 flambeau_quantize_row_f16_q8_0 grid=('34304', '1', '32') : 0.02/cyc × 4.7 µs
- v3a quant_ds4 flambeau_quantize_f16_q8_1_mmq grid=('3072', '67', '128') : 0.01/cyc × 6.2 µs
