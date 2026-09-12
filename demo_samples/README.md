# ==============================================================================
# SatQuery AI - Official Benchmark Demo Samples (SIH 2026 PS 26167)
# ==============================================================================
# All images below are extracted directly from the official evaluation benchmarks:
#
# 1. VRSBench (Single-Image Captioning, Grounding, VQA) - from xiang709/VRSBench
# 2. RSVQA (Single-Image Remote Sensing VQA) - from dmarsili/RSVQA-LR-2k
# 3. CDVQA (Bi-Temporal Change VQA) - from ljx620/CDVQA
# 4. ISRO/SAC Optical-SAR Pair Protocol - Cartosat/Sentinel-2 + RISAT/Sentinel-1
# ==============================================================================

Directory Structure:
--------------------
01_single_image_optical_vrsbench/
  - vrsbench_official_P0003_0002.png    : Official VRSBench validation image
  - vrsbench_official_metadata.json     : Official VRSBench ground-truth annotations (VQA, caption, grounding)
  - rsvqa_official_sample_0.png         : Official RSVQA validation image
  - rsvqa_official_metadata.json        : Official RSVQA question & answer ("Is it a rural or an urban area" -> "rural")
  - water_bodies_multispectral.tif      : Multi-band GeoTIFF with CRS and coordinates (demonstrates .tif ingestion)
  - PROMPTS.txt                         : Verifiable benchmark questions to copy & paste into demo

02_single_image_sar/
  - sentinel1_sar_cband.jpg             : Synthetic Aperture Radar (SAR) C-band backscatter image
  - PROMPTS.txt                         : Radar backscatter & roughness queries

03_cross_modal_optical_sar_bigearthnet/
  - cartosat_sentinel2_optical.jpg      : Co-registered Optical/MSI acquisition
  - risat_sentinel1_sar.jpg             : Co-registered SAR acquisition
  - PROMPTS.txt                         : Queries for DOFA wavelength-conditioned cross-modal fusion

04_bitemporal_change_cdvqa/
  - cdvqa_official_cdvqa-test-00000000.0.png  : Official CDVQA Test Time 1 (T1) image
  - cdvqa_official_cdvqa-test-00000000.1.png  : Official CDVQA Test Time 2 (T2) image
  - cdvqa_official_metadata.json              : Official CDVQA question & ground truth ("Did non-vegetated ground change?" -> "yes")
  - PROMPTS.txt                               : Verifiable CDVQA benchmark queries
