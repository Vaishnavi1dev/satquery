import React, { useState } from 'react';
import { 
  Cpu, 
  Radio, 
  Layers, 
  Clock, 
  ShieldCheck, 
  AlertTriangle, 
  Satellite, 
  Play, 
  TrendingUp, 
  CheckCircle2, 
  BarChart3, 
  ExternalLink,
  ChevronRight,
  Sparkles,
  Info
} from 'lucide-react';

export default function IntelligencePage({ onLaunchDemo, systemStatus = 'ONLINE' }) {
  const [activeCaseTab, setActiveCaseTab] = useState('cross_modal');

  const specialistModels = [
    {
      id: 'optical',
      name: 'DeltaVLM / Qwen2-VL',
      role: 'Optical Grounding & VQA Specialist',
      modality: 'RGB / High-Res Optical',
      resolution: '0.3m – 10m GSD',
      capabilities: ['Fine-grained Bounding Box Grounding', 'Dense Contextual Captioning', 'Visual Question Answering'],
      status: 'ONLINE',
      vram: '4.2 GB',
      color: 'var(--optical-color)',
      bg: 'var(--optical-bg)',
      border: 'var(--optical-border)',
      icon: Layers,
    },
    {
      id: 'sar',
      name: 'SAR-Whisperer / Qwen2-SAR',
      role: 'Synthetic Aperture Radar Specialist',
      modality: 'C/L-Band SAR Backscatter (VV/VH)',
      resolution: '5m – 20m GSD',
      capabilities: ['All-Weather Cloud Penetration', 'Metallic Corner Reflector Detection', 'Dielectric Moisture Sensing'],
      status: 'ONLINE',
      vram: '3.8 GB',
      color: 'var(--sar-color)',
      bg: 'var(--sar-bg)',
      border: 'var(--sar-border)',
      icon: Radio,
    },
    {
      id: 'changeformer',
      name: 'ChangeFormer / DeltaVLM Diff',
      role: 'Bi-Temporal Change Engine',
      modality: 'Dual-Epoch Co-registered Pairs (T1, T2)',
      resolution: 'Multi-Sensor Compatible',
      capabilities: ['Structural Expansion Detection', 'Environmental Shift Segmentation', 'Mask IoU Quantification'],
      status: 'ONLINE',
      vram: '3.5 GB',
      color: 'var(--temporal-color)',
      bg: 'var(--temporal-bg)',
      border: 'var(--temporal-border)',
      icon: Clock,
    },
    {
      id: 'sequence',
      name: 'Temporal Sequence Reasoner',
      role: 'Multi-Epoch Sequence Engine',
      modality: 'T1 ... TN Multi-Temporal Telemetry',
      resolution: 'Temporal Cadence (Daily – Seasonal)',
      capabilities: ['Longitudinal Trend Regression', 'Chronological Event Sequencing', 'Rate of Change Estimation'],
      status: 'ONLINE',
      vram: '2.9 GB',
      color: '#34d399',
      bg: 'rgba(52, 211, 153, 0.12)',
      border: 'rgba(52, 211, 153, 0.3)',
      icon: TrendingUp,
    },
    {
      id: 'multispectral',
      name: 'Spectral Index Analyzer',
      role: 'Multispectral Indices Engine',
      modality: '13-Band Multispectral (Sentinel-2)',
      resolution: '10m – 60m GSD',
      capabilities: ['NDVI Vegetation Stress', 'NDWI Water Inundation', 'NBR Burn Severity Mapping'],
      status: 'ONLINE',
      vram: '1.8 GB',
      color: 'var(--cyan-400)',
      bg: 'rgba(6, 182, 212, 0.12)',
      border: 'rgba(6, 182, 212, 0.3)',
      icon: Cpu,
    },
    {
      id: 'arbiter',
      name: 'Calibrated Fusion Arbiter',
      role: 'Cross-Sensor Uncertainty Engine',
      modality: 'Ensemble Meta-Reasoner',
      resolution: 'N/A (Decision Layer)',
      capabilities: ['Temperature Scaling (T=1.15)', 'Sensor Conflict Discrepancy Flagging', 'Reliability-Weighted Aggregation'],
      status: 'ONLINE',
      vram: '1.2 GB',
      color: '#f59e0b',
      bg: 'rgba(245, 158, 11, 0.12)',
      border: 'rgba(245, 158, 11, 0.3)',
      icon: ShieldCheck,
    },
  ];

  const caseStudies = {
    cross_modal: {
      title: 'Cloud-Penetrating Coastal Airbase Reconnaissance',
      scenario: 'High-altitude cirrus clouds completely occluded the optical satellite pass over a critical coastal runway. Conventional optical VLM reported zero visible aircraft and high uncertainty.',
      solution: 'SatQuery autonomously invoked SAR-Whisperer to fuse Sentinel-1 C-band radar backscatter. High-dielectric radar double-bounce reflections confirmed 14 parked airframes and verified runway integrity despite complete optical obscurity.',
      fusionOutcome: 'Confidence upgraded from 28% to 92%. Explicit conflict logged: "Optical cloud obstruction resolved via SAR radar penetration."',
      presetId: 'cross_modal',
      stats: [
        { label: 'Optical Visibility', val: '12%' },
        { label: 'SAR Penetration', val: '100%' },
        { label: 'Targets Grounded', val: '14 BBoxes' },
        { label: 'Calibrated Confidence', val: '92.4%' },
      ],
    },
    change_detection: {
      title: 'Strategic Port Infrastructure & Logistics Expansion',
      scenario: 'Monitoring maritime expansion across dual-epoch satellite observations (T1: March 2024 vs. T2: March 2026). The user inquired whether container handling capacity had expanded.',
      solution: 'ChangeFormer ingested co-registered multi-sensor scenes and produced a spatial difference mask isolating newly poured concrete aprons, 4 heavy ship-to-shore gantry cranes, and extended breakwaters.',
      fusionOutcome: 'Structural expansion quantified at +28.4% footprint growth with 4 new operational berths demarcated.',
      presetId: 'change',
      stats: [
        { label: 'Temporal Baseline', val: '24 Months' },
        { label: 'Expansion Area', val: '+28.4%' },
        { label: 'Berths Verified', val: '4 Berths' },
        { label: 'IoU Mask Agreement', val: '89.7%' },
      ],
    },
    sequence_analysis: {
      title: 'Multi-Epoch Drought & Reservoir Surface Depletion',
      scenario: 'Multi-temporal monitoring of major municipal water reservoir across 5 successive dry-season intervals (T1 through T5).',
      solution: 'The Temporal Sequence Reasoner plotted continuous shoreline regression curves, calculated cumulative water surface loss (-41.8%), and identified the critical depletion threshold event at Epoch T3.',
      fusionOutcome: 'Chronological timeline produced with exact rate-of-shrinkage milestones and drought inflection dates.',
      presetId: 'sequence',
      stats: [
        { label: 'Sequence Epochs', val: '5 Epochs' },
        { label: 'Surface Shrinkage', val: '-41.8%' },
        { label: 'Critical Threshold', val: 'Epoch T3' },
        { label: 'Regression Fit R²', val: '0.96' },
      ],
    },
  };

  return (
    <div className="intelligence-page-container" style={{ position: 'relative', zIndex: 1, paddingBottom: '4rem' }}>
      
      {/* Page Header */}
      <section style={{ textAlign: 'center', padding: '3rem 1rem 2rem', maxWidth: '960px', margin: '0 auto' }}>
        <span className="badge" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)', marginBottom: '0.75rem' }}>
          EARTH OBSERVATION INTELLIGENCE HUB
        </span>
        <h1 style={{
          fontSize: 'clamp(2.2rem, 4vw, 3.2rem)',
          fontWeight: 900,
          letterSpacing: '-0.02em',
          color: '#ffffff',
          marginBottom: '1rem',
        }}>
          Specialist Ensemble & Mission Telemetry
        </h1>
        <p style={{ fontSize: '1rem', color: 'var(--text-secondary)', maxWidth: '700px', margin: '0 auto', lineHeight: 1.6 }}>
          Comprehensive operational status of SatQuery's multi-model vision foundation ensemble, 
          sensor synergy cross-references, and calibrated uncertainty benchmarks.
        </p>
      </section>

      {/* Ensemble Registry Cards */}
      <section style={{ maxWidth: '1120px', margin: '0 auto', padding: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <Cpu size={20} color="var(--cyan-400)" />
            <span>Active Model Ensemble Registry</span>
          </h2>
          <span className="tag-pill" style={{ color: 'var(--success)', background: 'var(--success-glow)' }}>
            ALL 6 ENGINES OPERATIONAL
          </span>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(330px, 1fr))',
          gap: '1.25rem',
        }}>
          {specialistModels.map((m) => {
            const IconComponent = m.icon;
            return (
              <div 
                key={m.id} 
                className="card" 
                style={{ 
                  padding: '1.5rem', 
                  background: 'rgba(13, 19, 34, 0.75)',
                  border: `1px solid ${m.border}`,
                  position: 'relative',
                  overflow: 'hidden',
                }}
              >
                <div style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  right: 0,
                  height: 3,
                  background: m.color,
                }} />

                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                  <div style={{
                    width: 38,
                    height: 38,
                    borderRadius: 8,
                    background: m.bg,
                    border: `1px solid ${m.border}`,
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center',
                  }}>
                    <IconComponent size={20} color={m.color} />
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                    <span className="tag-pill mono" style={{ fontSize: '0.72rem', background: 'rgba(255,255,255,0.05)' }}>
                      VRAM: {m.vram}
                    </span>
                    <span className="tag-pill" style={{ fontSize: '0.72rem', color: 'var(--success)', background: 'var(--success-glow)' }}>
                      {m.status}
                    </span>
                  </div>
                </div>

                <h3 style={{ fontSize: '1.15rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.25rem' }}>
                  {m.name}
                </h3>
                <div style={{ fontSize: '0.8rem', color: m.color, fontWeight: 600, marginBottom: '0.85rem' }}>
                  {m.role}
                </div>

                <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', marginBottom: '0.85rem', display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
                  <div><strong style={{ color: '#cbd5e1' }}>Modality:</strong> {m.modality}</div>
                  <div><strong style={{ color: '#cbd5e1' }}>Resolution:</strong> {m.resolution}</div>
                </div>

                <div style={{ borderTop: '1px solid rgba(255, 255, 255, 0.06)', paddingTop: '0.75rem' }}>
                  <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.4rem' }}>
                    Specialist Capabilities
                  </div>
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column', gap: '0.35rem' }}>
                    {m.capabilities.map((c, i) => (
                      <li key={i} style={{ fontSize: '0.8rem', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <CheckCircle2 size={13} color="var(--cyan-400)" style={{ flexShrink: 0 }} />
                        <span>{c}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* Interactive Mission Case Studies */}
      <section style={{ maxWidth: '1120px', margin: '2.5rem auto 0', padding: '1rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
          <span className="badge" style={{ color: '#c084fc', background: 'rgba(192, 132, 252, 0.12)' }}>
            TACTICAL CASE STUDIES
          </span>
          <h2 style={{ fontSize: '1.85rem', fontWeight: 800, marginTop: '0.5rem', color: '#ffffff' }}>
            Mission Scenarios & Ground Truth Benchmarks
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '600px', margin: '0.4rem auto 0' }}>
            Real-world Earth Observation challenges showcasing how multi-model evidence fusion solves problems single-sensor models cannot.
          </p>
        </div>

        {/* Case Selector Tabs */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '0.5rem', marginBottom: '1.5rem', flexWrap: 'wrap' }}>
          <button 
            className={`btn btn-sm ${activeCaseTab === 'cross_modal' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setActiveCaseTab('cross_modal')}
          >
            <Radio size={14} />
            <span>Cloud-Penetrating Airbase (Optical + SAR + Multi-Spectral)</span>
          </button>
          <button 
            className={`btn btn-sm ${activeCaseTab === 'change_detection' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setActiveCaseTab('change_detection')}
          >
            <Clock size={14} />
            <span>Port Infrastructure Expansion (T1 vs T2)</span>
          </button>
          <button 
            className={`btn btn-sm ${activeCaseTab === 'sequence_analysis' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setActiveCaseTab('sequence_analysis')}
          >
            <TrendingUp size={14} />
            <span>Reservoir Drought Progression (T1...T5)</span>
          </button>
        </div>

        {/* Active Case Card */}
        {(() => {
          const c = caseStudies[activeCaseTab];
          return (
            <div style={{
              background: 'linear-gradient(180deg, rgba(16, 24, 44, 0.85) 0%, rgba(10, 16, 30, 0.95) 100%)',
              borderRadius: '16px',
              border: '1px solid rgba(255, 255, 255, 0.1)',
              padding: '2rem',
              boxShadow: '0 12px 36px rgba(0, 0, 0, 0.5)',
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.25rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.4rem' }}>
                    {c.title}
                  </h3>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                    <span className="tag-pill" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)' }}>
                      VERIFIED AUDIT
                    </span>
                    <span className="tag-pill mono" style={{ fontSize: '0.72rem' }}>
                      MISSION ID: PS-26167-{activeCaseTab.toUpperCase()}
                    </span>
                  </div>
                </div>

                <button 
                  className="btn btn-primary btn-sm"
                  onClick={() => onLaunchDemo(c.presetId)}
                  style={{ display: 'inline-flex', alignItems: 'center', gap: '0.45rem', padding: '0.6rem 1.1rem' }}
                >
                  <Play size={14} />
                  <span>Run Scenario in Ask SatQuery</span>
                </button>
              </div>

              {/* Grid: Scenario & Solution */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(300px, 1fr))',
                gap: '1.25rem',
                marginBottom: '1.5rem',
              }}>
                <div style={{ padding: '1.25rem', borderRadius: 12, background: 'rgba(239, 68, 68, 0.05)', border: '1px solid rgba(239, 68, 68, 0.2)' }}>
                  <div style={{ fontSize: '0.8rem', color: '#fca5a5', fontWeight: 700, marginBottom: '0.4rem', textTransform: 'uppercase' }}>
                    Operational Challenge
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.55 }}>
                    {c.scenario}
                  </p>
                </div>

                <div style={{ padding: '1.25rem', borderRadius: 12, background: 'rgba(6, 182, 212, 0.05)', border: '1px solid rgba(6, 182, 212, 0.2)' }}>
                  <div style={{ fontSize: '0.8rem', color: 'var(--cyan-400)', fontWeight: 700, marginBottom: '0.4rem', textTransform: 'uppercase' }}>
                    SatQuery Agent Solution
                  </div>
                  <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.55 }}>
                    {c.solution}
                  </p>
                </div>
              </div>

              {/* Stats Bar */}
              <div style={{
                display: 'grid',
                gridTemplateColumns: 'repeat(auto-fit, minmax(140px, 1fr))',
                gap: '0.85rem',
                padding: '1rem',
                borderRadius: 12,
                background: 'rgba(255, 255, 255, 0.03)',
                border: '1px solid rgba(255, 255, 255, 0.06)',
                marginBottom: '1.25rem',
              }}>
                {c.stats.map((s, idx) => (
                  <div key={idx}>
                    <div style={{ fontSize: '1.25rem', fontWeight: 800, color: '#ffffff', fontFamily: 'var(--font-mono)' }}>
                      {s.val}
                    </div>
                    <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', textTransform: 'uppercase' }}>
                      {s.label}
                    </div>
                  </div>
                ))}
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <Info size={15} color="var(--cyan-400)" style={{ flexShrink: 0 }} />
                <span><strong>Outcome:</strong> {c.fusionOutcome}</span>
              </div>
            </div>
          );
        })()}
      </section>

      {/* Modality Synergy Matrix */}
      <section style={{ maxWidth: '1120px', margin: '2.5rem auto 0', padding: '1rem' }}>
        <div style={{
          background: 'rgba(13, 19, 34, 0.75)',
          borderRadius: '16px',
          border: '1px solid var(--border-subtle)',
          padding: '2rem',
        }}>
          <h3 style={{ fontSize: '1.3rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.5rem' }}>
            Sensor Modality Synergy Matrix
          </h3>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.88rem', marginBottom: '1.5rem' }}>
            How SatQuery determines when to invoke optical, radar, or temporal tools based on environmental conditions and tactical intent.
          </p>

          <div style={{ overflowX: 'auto' }}>
            <table style={{ width: '100%', borderCollapse: 'collapse', textAlign: 'left', fontSize: '0.85rem' }}>
              <thead>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.12)', color: 'var(--text-muted)' }}>
                  <th style={{ padding: '0.75rem 1rem' }}>Sensor Type</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Cloud Penetration</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Night Imaging</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Key Spectral Strength</th>
                  <th style={{ padding: '0.75rem 1rem' }}>Primary Intelligence Role</th>
                </tr>
              </thead>
              <tbody>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: 'var(--optical-color)' }}>
                    High-Res Optical (RGB)
                  </td>
                  <td style={{ padding: '0.85rem 1rem', color: '#fca5a5' }}>None (0%)</td>
                  <td style={{ padding: '0.85rem 1rem', color: '#fca5a5' }}>No</td>
                  <td style={{ padding: '0.85rem 1rem' }}>Visible color, textures, shape fidelity</td>
                  <td style={{ padding: '0.85rem 1rem' }}>Precision object detection, markings, fine layout</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: 'var(--sar-color)' }}>
                    SAR Radar (C/L-Band)
                  </td>
                  <td style={{ padding: '0.85rem 1rem', color: 'var(--success)' }}>Complete (100%)</td>
                  <td style={{ padding: '0.85rem 1rem', color: 'var(--success)' }}>Yes (24/7 All-Weather)</td>
                  <td style={{ padding: '0.85rem 1rem' }}>Surface roughness, dielectric, corner reflections</td>
                  <td style={{ padding: '0.85rem 1rem' }}>Vessel detection, runway structure, metallic hulls</td>
                </tr>
                <tr style={{ borderBottom: '1px solid rgba(255, 255, 255, 0.05)' }}>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: 'var(--cyan-400)' }}>
                    Multispectral (13-Band)
                  </td>
                  <td style={{ padding: '0.85rem 1rem', color: '#fde047' }}>Haze only</td>
                  <td style={{ padding: '0.85rem 1rem', color: '#fca5a5' }}>No</td>
                  <td style={{ padding: '0.85rem 1rem' }}>NIR / SWIR chlorophyll & moisture reflection</td>
                  <td style={{ padding: '0.85rem 1rem' }}>Agricultural health, flood zones, burn severity</td>
                </tr>
                <tr>
                  <td style={{ padding: '0.85rem 1rem', fontWeight: 600, color: 'var(--temporal-color)' }}>
                    Multi-Temporal (T1...TN)
                  </td>
                  <td style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Sensor Dependent</td>
                  <td style={{ padding: '0.85rem 1rem', color: 'var(--text-secondary)' }}>Sensor Dependent</td>
                  <td style={{ padding: '0.85rem 1rem' }}>Differential delta between co-registered epochs</td>
                  <td style={{ padding: '0.85rem 1rem' }}>Infrastructure growth, coastline erosion, trend regression</td>
                </tr>
              </tbody>
            </table>
          </div>
        </div>
      </section>

      {/* Uncertainty Calibration Box */}
      <section style={{ maxWidth: '1120px', margin: '2.5rem auto 0', padding: '1rem' }}>
        <div style={{
          padding: '1.75rem',
          borderRadius: 16,
          background: 'rgba(245, 158, 11, 0.05)',
          border: '1px solid rgba(245, 158, 11, 0.25)',
          display: 'flex',
          alignItems: 'flex-start',
          gap: '1.25rem',
        }}>
          <div style={{
            width: 44,
            height: 44,
            borderRadius: 10,
            background: 'rgba(245, 158, 11, 0.15)',
            border: '1px solid rgba(245, 158, 11, 0.3)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            flexShrink: 0,
          }}>
            <ShieldCheck size={24} color="#f59e0b" />
          </div>

          <div>
            <h4 style={{ fontSize: '1.15rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.4rem' }}>
              Calibrated Uncertainty & Anti-Hallucination Framework
            </h4>
            <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.55 }}>
              In critical defense and Earth Observation tasks, a false positive or hallucinated certainty can be catastrophic. 
              SatQuery applies post-hoc <strong>Temperature Scaling (<em>T</em> = 1.15)</strong> on model logit distributions and runs 
              cross-specialist consensus voting. When models contradict one another (e.g., optical reports 0 aircraft due to clouds while SAR reports 14), 
              the system triggers an explicit <strong>Conflict Flag</strong> rather than silently averaging the outputs.
            </p>
          </div>
        </div>
      </section>

    </div>
  );
}
