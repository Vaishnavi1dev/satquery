import React from 'react';
import { Cpu, Radio, Clock, Sparkles, Eye, ShieldCheck, ArrowRight, Layers } from 'lucide-react';

export default function AgentRoutingBadge({ imagesCount, imagesList, query }) {
  const q_lower = (query || '').toLowerCase().trim();

  // Autonomous Agent Architecture Inference
  let architectureName = 'Awaiting Telemetry Ingestion...';
  let specialistModel = 'Agent Auto-Router Standby';
  let badgeColor = 'var(--text-muted)';
  let rationale = 'Drop 1, 2, or more satellite scenes and type your query. The agent will autonomously inspect spectral bands, coordinate pairing, and question intent to dispatch the optimal specialist model.';
  let Icon = Sparkles;
  let modeCode = 'STANDBY';

  const modalities = imagesList.map((img) => img.modality);
  const hasOpt = modalities.some((m) => m in { optical: 1, multispectral: 1 });
  const hasSar = modalities.some((m) => m === 'sar');

  if (imagesCount === 0) {
    if (q_lower) {
      architectureName = 'Query Interpreted (Awaiting Imagery)';
      rationale = `Agent ready to analyze question: "${query.slice(0, 48)}...". Upload scene(s) to execute.`;
    }
  } else if (imagesCount >= 3) {
    modeCode = 'MODE D';
    architectureName = 'Multi-Temporal Sequence Analysis (T1...TN)';
    specialistModel = 'DeltaVLM-Sequence Specialist';
    badgeColor = 'var(--temporal-color)';
    Icon = Clock;
    rationale = `Agent identified a multi-epoch sequence of ${imagesCount} satellite scenes. Automatically orchestrating cumulative trend extraction and chronological timeline synthesis.`;
  } else if (imagesCount === 2) {
    if ((hasOpt && hasSar) || q_lower.includes('sar') || q_lower.includes('radar')) {
      modeCode = 'MODE B';
      architectureName = 'Optical + SAR + Multi-Spectral Cross-Modal Fusion';
      specialistModel = 'DOFA-Large ViT-B (Wavelength Conditioned)';
      badgeColor = 'var(--sar-color)';
      Icon = Radio;
      rationale = 'Agent verified complementary Optical reflectance and SAR radar backscatter. Automatically fusing both sensors to penetrate cloud cover and extract structural dielectric returns.';
    } else {
      modeCode = 'MODE C';
      architectureName = 'Bi-Temporal Change Detection & Change-VQA';
      specialistModel = 'DeltaVLM + Qwen3.5-2B';
      badgeColor = 'var(--temporal-color)';
      Icon = Clock;
      rationale = 'Agent verified 2 co-registered temporal observations. Automatically routing to bi-temporal difference modeling and cycle-consistent change verification.';
    }
  } else if (imagesCount === 1) {
    const isGrounding = ['highlight', 'locate', 'box', 'draw', 'bounding', 'where is', 'detect', 'pinpoint'].some((k) => q_lower.includes(k));
    const isCaption = ['describe', 'caption', 'summary', 'overview', 'land-cover'].some((k) => q_lower.includes(k));
    const imgModality = modalities[0] || 'optical';
    const imgBands = imagesList[0]?.bands || 3;

    modeCode = 'MODE A';
    if (imgModality === 'multispectral') {
      architectureName = 'Single-Image Multispectral (MSI) Reasoning';
      specialistModel = 'EarthDial-4B (Band-Fusion & Spectral Analysis)';
      badgeColor = 'var(--msi-color)';
      Icon = Layers;
      rationale = `Agent detected a ${imgBands}-band Multispectral (MSI) datacube. Routing to EarthDial-4B band-fusion module to analyze spectral reflectance, chlorophyll absorption (NDVI), water indices (NDWI), and terrain composition.`;
    } else if (imgModality === 'sar') {
      architectureName = 'Single-Image SAR Radar VQA / Grounding';
      specialistModel = 'EarthDial-4B (Radar Backscatter Adaptation)';
      badgeColor = 'var(--sar-color)';
      Icon = Radio;
      rationale = 'Single SAR observation uploaded. Agent routing to EarthDial-4B radar adapter for dielectric backscatter interpretation, surface roughness, and metallic feature identification.';
    } else if (isGrounding) {
      architectureName = 'Text-Guided Visual Grounding (Optical)';
      specialistModel = 'EarthDial-4B (Grounding Head)';
      badgeColor = 'var(--cyan-400)';
      Icon = Eye;
      rationale = 'Single optical scene uploaded with spatial localization intent. Agent routing query to EarthDial-4B high-resolution grounding head for bounding box prediction.';
    } else if (isCaption) {
      architectureName = 'Natural Language Scene Captioning (Optical)';
      specialistModel = 'EarthDial-4B (LLM Primary)';
      badgeColor = 'var(--cyan-400)';
      Icon = Layers;
      rationale = 'Single optical scene uploaded with descriptive request. Agent routing to EarthDial-4B for comprehensive terrain and land-cover description.';
    } else {
      architectureName = 'Single-Image Remote Sensing VQA (Optical)';
      specialistModel = 'EarthDial-4B (InternVL2 + Band-Fusion)';
      badgeColor = 'var(--optical-color)';
      Icon = Eye;
      rationale = 'Single optical scene uploaded. Agent routing to EarthDial-4B for question answering, feature counting, and domain reasoning.';
    }
  }

  return (
    <div
      className="glass-panel"
      style={{
        padding: '1.25rem 1.5rem',
        borderRadius: '14px',
        border: '1px solid rgba(6, 182, 212, 0.3)',
        background: 'rgba(10, 16, 30, 0.75)',
        backdropFilter: 'blur(16px)',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.65rem',
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
          <div
            style={{
              width: 32,
              height: 32,
              borderRadius: '8px',
              background: 'rgba(6, 182, 212, 0.12)',
              color: 'var(--cyan-400)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
            }}
          >
            <Icon size={18} />
          </div>
          <div>
            <div style={{ fontSize: '0.72rem', textTransform: 'uppercase', letterSpacing: '0.06em', color: 'var(--text-muted)', fontWeight: 600 }}>
              Autonomous Agent Architecture Selection
            </div>
            <div style={{ fontSize: '1.05rem', fontWeight: 700, color: 'var(--text-primary)' }}>
              {architectureName}
            </div>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span className="tag-pill mono" style={{ color: badgeColor, border: `1px solid ${badgeColor}`, background: 'rgba(0, 0, 0, 0.3)' }}>
            {modeCode}
          </span>
          <span className="tag-pill mono" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)' }}>
            <Cpu size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
            {specialistModel}
          </span>
        </div>
      </div>

      <div style={{ fontSize: '0.82rem', color: 'var(--text-secondary)', lineHeight: 1.45, borderTop: '1px solid rgba(255, 255, 255, 0.05)', paddingTop: '0.6rem' }}>
        <span style={{ color: 'var(--cyan-400)', fontWeight: 600 }}>Agent Routing Rationale: </span>
        {rationale}
      </div>
    </div>
  );
}
