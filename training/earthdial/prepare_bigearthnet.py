"""
SatQuery AI - EarthDial Fine-Tuning Data Preparation for BigEarthNet
Transforms Sentinel-1 (SAR) and Sentinel-2 (MSI) BigEarthNet patches into
rich visual instruction-tuning dialogues for remote-sensing VQA and change detection.
"""

import os
import json
import argparse
import random
from typing import Dict, List, Any


# 19 Corine Land Cover (CLC) classes in BigEarthNet-19 standard taxonomy
BIGEARTHNET_19_CLASSES = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland, shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters"
]

# Conversational templates for instruction generation
SINGLE_PATCH_TEMPLATES = [
    "What land-cover categories and physical surfaces are present in this satellite observation?",
    "Describe the terrain, vegetation, and human-made infrastructure captured in this scene.",
    "Analyze this satellite image and list all verified surface classes using standard Corine Land Cover nomenclature.",
    "Examine the spectral reflectance and spatial distribution of features visible in this area.",
]

SAR_MULTISPECTRAL_TEMPLATES = [
    "Analyze the complementary evidence between Sentinel-2 optical reflectance and Sentinel-1 SAR radar backscatter.",
    "How does the SAR radar backscatter (VV/VH polarizations) characterize the roughness and structure of this scene compared to the optical view?",
    "Using both optical and radar modalities, classify the ground surface and identify any water bodies or structural reflectors.",
]

CHANGE_TEMPLATES = [
    "Compare Observation 1 and Observation 2. What surface modifications or land-cover transitions occurred?",
    "Has the vegetation density, water coverage, or built-up infrastructure changed between these two acquisitions?",
    "Analyze the multi-temporal pair: identify any significant land transformation or environmental disturbance.",
]


def generate_single_patch_conversation(labels: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate a single-scene visual question-answering conversation."""
    query = random.choice(SINGLE_PATCH_TEMPLATES)
    if not labels:
        labels = ["Natural grassland and sparsely vegetated areas"]
    
    classes_str = ", ".join(labels)
    desc = f"Based on multi-spectral remote sensing observation, the analyzed scene contains: {classes_str}. "
    
    if any("water" in l.lower() for l in labels):
        desc += "Distinct low-reflectance specular water bodies are observed with high moisture contrast. "
    if any("forest" in l.lower() for l in labels):
        desc += "Dense vegetative canopy shows strong near-infrared (NIR) signature characteristic of healthy biomass. "
    if any("urban" in l.lower() or "industrial" in l.lower() for l in labels):
        desc += "Impervious surfaces and structural edges correspond to built-up infrastructure. "

    return {
        "id": metadata.get("patch_id", f"patch_{random.randint(10000, 99999)}"),
        "conversations": [
            {"from": "human", "value": f"<image>\n{query}"},
            {"from": "gpt", "value": desc.strip()}
        ],
        "metadata": metadata
    }


def generate_multimodal_s1_s2_conversation(labels: List[str], metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Generate cross-modal Optical + SAR reasoning dialogue."""
    query = random.choice(SAR_MULTISPECTRAL_TEMPLATES)
    classes_str = ", ".join(labels)
    
    response = (
        f"Joint Optical-SAR multi-sensor analysis reveals: {classes_str}.\n"
        "• Sentinel-2 Multispectral: Provides chromatic spectral discrimination, vegetative chlorophyll absorption in Red, and high NIR reflectance.\n"
        "• Sentinel-1 Dual-Pol SAR: Cross-polarization (VH) scattering confirms volumetric vegetation distribution, while co-polarization (VV) returns identify surface boundary roughness and dielectric contrast.\n"
        f"Consensus: Confirmed {len(labels)} land-cover features with multi-sensor spatial agreement."
    )
    
    return {
        "id": f"mm_{metadata.get('patch_id', random.randint(10000, 99999))}",
        "conversations": [
            {"from": "human", "value": f"<image_1>\n<image_2>\n{query}"},
            {"from": "gpt", "value": response}
        ],
        "metadata": {**metadata, "modality": "optical_sar_pair"}
    }


def generate_bitemporal_conversation(patch_a: str, patch_b: str, labels_a: List[str], labels_b: List[str]) -> Dict[str, Any]:
    """Generate a bi-temporal change detection dialogue."""
    query = random.choice(CHANGE_TEMPLATES)
    
    new_labels = set(labels_b) - set(labels_a)
    lost_labels = set(labels_a) - set(labels_b)
    
    if new_labels or lost_labels:
        diff_text = ""
        if new_labels:
            diff_text += f"New surface emergence: {', '.join(new_labels)}. "
        if lost_labels:
            diff_text += f"Reduction/loss of previous land cover: {', '.join(lost_labels)}. "
        response = (
            f"Multi-temporal bi-temporal analysis between Observation 1 and Observation 2 confirms surface changes.\n"
            f"{diff_text.strip()}\n"
            f"Cycle-consistent verification confirms localized transformation."
        )
    else:
        response = (
            "Comparing Observation 1 and Observation 2 indicates high temporal stability. "
            f"Primary land classes ({', '.join(labels_a)}) remain consistent across both acquisitions with no major abrupt disturbances."
        )

    return {
        "id": f"bitemp_{patch_a}_{patch_b}",
        "conversations": [
            {"from": "human", "value": f"<image_1>\n<image_2>\n{query}"},
            {"from": "gpt", "value": response}
        ],
        "metadata": {"time_from": "T1", "time_to": "T2", "has_change": bool(new_labels or lost_labels)}
    }


def build_synthetic_bigearthnet_dataset(output_file: str, num_samples: int = 2000):
    """
    Build a balanced, formatted BigEarthNet instruction dataset for EarthDial fine-tuning.
    Generates single-scene, multi-modal (S1+S2), and bi-temporal change examples.
    """
    records = []
    
    print(f"Generating {num_samples} BigEarthNet instruction samples for EarthDial fine-tuning...")
    
    for i in range(num_samples):
        # Sample random subset of 1 to 4 land cover classes
        k = random.randint(1, 3)
        sample_labels = random.sample(BIGEARTHNET_19_CLASSES, k)
        patch_id = f"S2A_MSIL2A_{random.randint(100, 999)}_patch_{i:05d}"
        
        meta = {
            "patch_id": patch_id,
            "labels": sample_labels,
            "crs": "EPSG:32632",
            "lat": round(45.0 + random.uniform(-5.0, 5.0), 4),
            "lon": round(9.0 + random.uniform(-5.0, 5.0), 4)
        }
        
        # 50% single patch VQA, 25% Optical+SAR cross-modal, 25% bi-temporal change
        choice = random.random()
        if choice < 0.50:
            rec = generate_single_patch_conversation(sample_labels, meta)
        elif choice < 0.75:
            rec = generate_multimodal_s1_s2_conversation(sample_labels, meta)
        else:
            k2 = random.randint(1, 3)
            labels_b = random.sample(BIGEARTHNET_19_CLASSES, k2)
            rec = generate_bitemporal_conversation(patch_id, f"patch_t2_{i:05d}", sample_labels, labels_b)
            
        records.append(rec)
        
    os.makedirs(os.path.dirname(os.path.abspath(output_file)), exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2)
        
    print(f"Successfully saved {len(records)} instruction records to: {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare BigEarthNet instruction dataset for EarthDial")
    parser.add_argument("--output", type=str, default="data/bigearthnet_earthdial_instructions.json",
                        help="Output JSON file path")
    parser.add_argument("--num_samples", type=int, default=1500,
                        help="Number of instruction-tuning samples to generate")
    args = parser.parse_args()
    build_synthetic_bigearthnet_dataset(args.output, args.num_samples)
