# ==============================================================================
# SatQuery AI - Demo Samples for SIH 2026 PS 26167
# ==============================================================================
# This directory contains ready-to-use sample imagery curated for your video demo,
# organized by the exact modalities and benchmarks required in the problem statement.
#
# Simply open your file picker in the SatQuery web UI and drag & drop these files!
# ==============================================================================

Directory Structure:
--------------------
01_single_image_optical_vrsbench/
  - urban_scene_optical.jpg          : High-resolution urban scene for VQA & captioning
  - sentinel2_optical_scene.jpg      : Multi-spectral agricultural / rural scene
  - water_bodies_multispectral.tif   : Real geospatial GeoTIFF with coordinate metadata
  - PROMPTS.txt                      : Recommended queries for single-image demonstration

02_single_image_sar/
  - sentinel1_sar_cband.jpg          : Synthetic Aperture Radar (SAR) C-band backscatter image
  - PROMPTS.txt                      : Recommended queries for single SAR image analysis

03_cross_modal_optical_sar_bigearthnet/
  - cartosat_sentinel2_optical.jpg   : Co-registered Optical/MSI acquisition
  - risat_sentinel1_sar.jpg          : Co-registered SAR acquisition of the same area
  - PROMPTS.txt                      : Recommended queries for cross-modal fusion (DOFA)

04_bitemporal_change_cdvqa/
  - t1_pre_change_optical.jpg        : Time 1 (T1) baseline optical observation
  - t2_post_change_optical.jpg       : Time 2 (T2) follow-up observation showing change
  - PROMPTS.txt                      : Recommended queries for multitemporal change detection


Do you need to download entire external benchmark datasets?
------------------------------------------------------------
NO. The full benchmark test sets (VRSBench: ~20 GB, RSVQA: ~25 GB, CDVQA: ~15 GB, ISRO/SAC: hidden)
are only required if you are executing batch evaluations across 100,000+ images.

For your demonstration video and judging presentation:
- The images in this folder are 100% genuine remote-sensing data (Sentinel-1 SAR, Sentinel-2 Optical, and GeoTIFFs).
- Uploading these pairs into the SatQuery UI tests all 4 mandatory pathways:
  1. Single-Image VQA & Captioning
  2. Single-Image Region Grounding
  3. Bi-Temporal Change Detection & Difference Mapping
  4. Cross-Modal Optical + SAR Feature Fusion

If you want to add additional samples from the official benchmarks:
- VRSBench: https://huggingface.co/datasets/L-A-R-S/VRSBench
- RSVQA: https://rsvqa.sylvainlobry.com/
- CDVQA: https://github.com/Chen-Z-H/CDVQA
Just drop any extra `.jpg`, `.png`, or `.tif` files into the appropriate subfolder above.
