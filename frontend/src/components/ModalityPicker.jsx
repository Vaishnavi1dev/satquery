import React from 'react';
import { Eye, Radio, Clock, Sparkles } from 'lucide-react';

const MODALITIES = [
  {
    id: 'optical',
    name: 'Mode A: Single Optical / Multispectral / SAR',
    shortName: 'Single Scene VQA & Grounding',
    icon: Eye,
    badge: '1 Image (RGB / MSI / SAR)',
    model: 'EarthDial-4B Specialist',
    description: 'Single-scene analysis across Optical (RGB), Multispectral (MSI GeoTIFF / 4-13 bands), or SAR radar. Supports visual question answering, spectral indices (NDVI/NDWI), dense captioning, and text-guided region grounding.',
    cardClass: 'optical',
    slots: [
      { id: 'opt_primary', label: 'Primary Scene (Optical / MSI / SAR)', accept: 'RGB, Multispectral (GeoTIFF, TIFF), or SAR (PNG, JPG)', modality: 'optical' }
    ]
  },
  {
    id: 'cross_modal',
    name: 'Mode B: Cross-Modal (Optical + SAR + Multi-Spectral)',
    shortName: 'Optical + SAR + Multi-Spectral Fusion',
    icon: Radio,
    badge: '2 Images (Pair / Multisensor)',
    model: 'DOFA-Large / Fusion Engine',
    description: 'All-weather SAR radar backscatter (VV/VH) fused with optical and multi-spectral bands to penetrate clouds, smoke, and resolve spectral signatures.',
    cardClass: 'sar',
    slots: [
      { id: 'pair_optical', label: 'Optical / Multi-Spectral Scene', accept: 'Sentinel-2 / Landsat / High-Res Optical (RGB/MSI)', modality: 'optical' },
      { id: 'pair_sar', label: 'SAR Radar Scene (Co-registered)', accept: 'Sentinel-1 / RISAT VV/VH Radar', modality: 'sar' }
    ]
  },
  {
    id: 'bitemporal',
    name: 'Mode C: Bi-Temporal Change Detection',
    shortName: 'Pre & Post Event',
    icon: Clock,
    badge: '2 Images (Pair)',
    model: 'DeltaVLM Bi-Temporal',
    description: 'Co-registered pre-event (T0) and post-event (T1) scenes for disaster impact, urban expansion, and cycle-consistent change logging.',
    cardClass: 'temporal',
    slots: [
      { id: 'temp_t0', label: 'Pre-Event Scene (T0)', accept: 'Baseline / Pre-Disaster Scene', modality: 'optical' },
      { id: 'temp_t1', label: 'Post-Event Scene (T1)', accept: 'Post-Disaster / Recent Scene', modality: 'optical' }
    ]
  },
  {
    id: 'temporal_sequence',
    name: 'Mode D: Multi-Temporal Sequence (T1...TN)',
    shortName: 'Timeline Progression',
    icon: Sparkles,
    badge: '3+ Images (Timeline)',
    model: 'DeltaVLM-Sequence Specialist',
    description: 'Continuous chronological analysis across T1 → T2 → ... → TN epochs producing cumulative event trends and land evolution curves.',
    cardClass: 'temporal',
    slots: [
      { id: 'seq_t1', label: 'Epoch T1 (Baseline Acquisition)', accept: 'Baseline Observation T1', modality: 'optical' },
      { id: 'seq_t2', label: 'Epoch T2 (Intermediate Epoch)', accept: 'Observation T2', modality: 'optical' },
      { id: 'seq_t3', label: 'Epoch T3 (Recent Observation)', accept: 'Observation T3', modality: 'optical' }
    ]
  }
];

export { MODALITIES };

export default function ModalityPicker({ selectedModality, onSelectModality }) {
  return (
    <div>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
        <div>
          <h2 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Select Analysis Architecture</h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Choose the sensor modality configuration tailored for your remote sensing task.
          </p>
        </div>
      </div>

      <div className="modality-picker-grid">
        {MODALITIES.map((mod) => {
          const Icon = mod.icon;
          const isSelected = selectedModality === mod.id;
          return (
            <div
              key={mod.id}
              className={`modality-card ${mod.cardClass} ${isSelected ? 'selected' : ''}`}
              onClick={() => onSelectModality(mod.id)}
            >
              <div className="modality-header">
                <div className="modality-title">
                  <div className="modality-icon">
                    <Icon size={18} />
                  </div>
                  <span>{mod.name}</span>
                </div>
                <span className="modality-badge tag-pill">{mod.badge}</span>
              </div>

              <p className="modality-desc">{mod.description}</p>

              <div className="modality-model-meta">
                <span>Specialist:</span>
                <span className="mono" style={{ color: 'var(--text-primary)', fontWeight: 600 }}>
                  {mod.model}
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
