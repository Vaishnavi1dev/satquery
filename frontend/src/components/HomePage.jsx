import React, { useState } from 'react';
import {
  Satellite,
  Layers,
  Radio,
  Clock,
  ShieldCheck,
  Cpu,
  ArrowRight,
  CheckCircle2,
  TrendingUp,
  BarChart3,
  Play,
  Info,
  Sparkles
} from 'lucide-react';

export default function HomePage({ onNavigateToAsk, onLaunchDemo }) {
  const [activeCaseTab, setActiveCaseTab] = useState('cross_modal');

  const specialistModels = [
    {
      id: 'earthdial-vqa',
      name: 'EarthDial-4B',
      role: 'Single-Image Vision-Language Specialist',
      modality: 'Single Scene (Optical / MSI / SAR)',
      resolution: '512×512 VLM Input',
      capabilities: ['Visual Question Answering (Optical/MSI/SAR)', 'Land-Cover Captioning', 'Text-Guided Region Grounding'],
      status: 'ONLINE',
      vram: '8.4 GB',
      color: 'var(--optical-color)',
      bg: 'var(--optical-bg)',
      border: 'var(--optical-border)',
      icon: Layers,
    },
    {
      id: 'dofa',
      name: 'DOFA ViT-B + Cross-Attention Head',
      role: 'Cross-Modal Optical/MSI + SAR Fusion Specialist',
      modality: 'Wavelength-Conditioned Optical + SAR Fusion',
      resolution: '512×512 Co-Registered',
      capabilities: ['Wavelength-Conditioned Feature Extraction', 'Optical↔SAR Cross-Attention Fusion', 'Joint Land-Cover Analysis'],
      status: 'ONLINE',
      vram: '0.4 GB',
      color: 'var(--sar-color)',
      bg: 'var(--sar-bg)',
      border: 'var(--sar-border)',
      icon: Radio,
    },
    {
      id: 'earthdial-change',
      name: 'EarthDial-4B Multi-Image Engine',
      role: 'Bi-Temporal Change Specialist',
      modality: 'Bi-Temporal Image Pair (T1, T2)',
      resolution: '512×512 Co-Registered',
      capabilities: ['Change Description & Change-VQA', 'Difference-Mask Localization', 'WGS84 Geo-Referenced Change Zones'],
      status: 'ONLINE',
      vram: '8.4 GB',
      color: 'var(--temporal-color)',
      bg: 'var(--temporal-bg)',
      border: 'var(--temporal-border)',
      icon: Clock,
    },
    {
      id: 'earthdial-sequence',
      name: 'EarthDial-4B Multi-Temporal Engine',
      role: 'T1…TN Sequence Specialist',
      modality: 'T1…TN Multi-Temporal Sequence',
      resolution: '512×512 per Epoch',
      capabilities: ['Cumulative Trend Analysis', 'Chronological Event Sequencing', 'Rate-of-Change Estimation'],
      status: 'ONLINE',
      vram: '8.4 GB',
      color: '#34d399',
      bg: 'rgba(52, 211, 153, 0.12)',
      border: 'rgba(52, 211, 153, 0.3)',
      icon: TrendingUp,
    },
    {
      id: 'rs-ground',
      name: 'EarthDial-4B Grounding Head',
      role: 'Text-Guided Grounding Specialist',
      modality: 'Single Scene + Referring Expression',
      resolution: '512×512 Bounding-Box Output',
      capabilities: ['Bounding-Box Localization', 'Honest not_located Fallback', 'Referring-Expression Grounding'],
      status: 'ONLINE',
      vram: '8.4 GB',
      color: 'var(--cyan-400)',
      bg: 'rgba(6, 182, 212, 0.12)',
      border: 'rgba(6, 182, 212, 0.3)',
      icon: Cpu,
    },
    {
      id: 'arbiter',
      name: 'Calibrated Fusion Arbiter',
      role: 'Cross-Sensor Uncertainty & Evidence Synthesis (Platform Layer)',
      modality: 'Ensemble Meta-Reasoner',
      resolution: 'N/A (Decision Layer)',
      capabilities: ['Temperature Scaling (T=1.15)', 'Specialist Conflict Flagging', 'Reliability-Weighted Aggregation'],
      status: 'ONLINE',
      vram: '0.2 GB',
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
      solution: 'SatQuery autonomously invokes the DOFA cross-modal specialist to fuse Sentinel-1 C-band radar backscatter with the optical pass and jointly analyze built-up and water regions despite complete optical obscurity.',
      capability: 'Routes the optical + SAR pair to the DOFA ViT-B cross-attention fusion specialist to jointly analyze built-up and water regions.',
      presetId: 'cross_modal',
    },
    change_detection: {
      title: 'Strategic Port Infrastructure & Logistics Expansion',
      scenario: 'Monitoring maritime expansion across dual-epoch satellite observations (T1: March 2024 vs. T2: March 2026). The user inquired whether container handling capacity had expanded.',
      solution: 'The EarthDial-4B multi-image change engine ingests co-registered multi-sensor scenes and produces a spatial difference mask to localize newly poured concrete aprons, ship-to-shore gantry cranes, and extended breakwaters.',
      capability: 'Routes the two epochs to the EarthDial-4B multi-image engine for bi-temporal change analysis plus a pixel-difference mask.',
      presetId: 'change',
    },
    sequence_analysis: {
      title: 'Multi-Epoch Drought & Reservoir Surface Depletion',
      scenario: 'Multi-temporal monitoring of major municipal water reservoir across 5 successive dry-season intervals (T1 through T5).',
      solution: 'The EarthDial-4B multi-temporal engine routes the sequence for chronological change analysis and a timeline of reservoir surface progression.',
      capability: 'Routes the multi-epoch sequence to the multi-temporal engine for a chronological timeline.',
      presetId: 'sequence',
    },
  };

  return (
    <div className="home-page-container" style={{ position: 'relative', zIndex: 1, paddingBottom: '4rem' }}>

      {/* 1. Hero Section */}
      <section className="home-hero-section" style={{
        textAlign: 'center',
        padding: '3.5rem 1rem 2rem',
        maxWidth: '1040px',
        margin: '0 auto',
      }}>
        <div
          className="ask-satquery-capsule"
          onClick={() => onNavigateToAsk()}
          style={{ cursor: 'pointer', marginBottom: '1.25rem', display: 'inline-flex' }}
        >
          <span>AUTONOMOUS EARTH OBSERVATION AGENT</span>
          <Sparkles size={13} color="var(--cyan-400)" />
        </div>

        <h1 style={{
          fontSize: 'clamp(2.5rem, 5vw, 4rem)',
          fontWeight: 900,
          letterSpacing: '-0.03em',
          lineHeight: 1.1,
          color: '#ffffff',
          textShadow: '0 0 40px rgba(6, 182, 212, 0.4)',
          marginBottom: '1.25rem',
        }}>
          Intelligent Satellite Reasoning with <span style={{
            background: 'linear-gradient(135deg, var(--cyan-400), #60a5fa)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}>Observable Evidence</span>
        </h1>

        <p style={{
          fontSize: '1.1rem',
          color: 'var(--text-secondary)',
          maxWidth: '740px',
          margin: '0 auto 2.5rem',
          lineHeight: 1.6,
        }}>
          SatQuery is an autonomous AI agent that transforms raw orbital telemetry,
          optical scenes, multispectral datacubes (MSI), SAR radar backscatter, and multi-temporal sequences into calibrated, evidence-backed intelligence.
        </p>

        {/* Call to Actions */}
        <div style={{ display: 'flex', justifyContent: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <button
            className="btn btn-primary"
            onClick={() => onNavigateToAsk()}
            style={{
              padding: '0.85rem 1.75rem',
              fontSize: '1rem',
              fontWeight: 700,
              boxShadow: '0 0 25px rgba(6, 182, 212, 0.45)',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.6rem',
            }}
          >
            <span>Launch Ask SatQuery Workspace</span>
            <ArrowRight size={18} />
          </button>
        </div>

        {/* Stats Strip */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1.25rem',
          marginTop: '3rem',
          padding: '1.25rem 1.75rem',
          background: 'rgba(10, 16, 30, 0.65)',
          backdropFilter: 'blur(16px)',
          borderRadius: '16px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
        }}>
          <div style={{ flex: '1 1 auto', textAlign: 'center', minWidth: '140px' }}>
            <div style={{ fontSize: '1.65rem', fontWeight: 800, color: 'var(--cyan-400)', fontFamily: 'var(--font-mono)' }}>6 Tools</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Specialist AI Ensemble</div>
          </div>
          <div style={{ flex: '1.4 1 auto', textAlign: 'center', minWidth: '240px' }}>
            <div style={{ fontSize: '1.45rem', fontWeight: 800, color: 'var(--msi-color)', fontFamily: 'var(--font-mono)' }}>Optical • MSI • SAR</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Tri-Modal Sensor Ingest</div>
          </div>
          <div style={{ flex: '1 1 auto', textAlign: 'center', minWidth: '150px' }}>
            <div style={{ fontSize: '1.65rem', fontWeight: 800, color: '#fbbf24', fontFamily: 'var(--font-mono)' }}>T1 ... TN</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Multi-Temporal Sequences</div>
          </div>
          <div style={{ flex: '1.1 1 auto', textAlign: 'center', minWidth: '150px' }}>
            <div style={{ fontSize: '1.65rem', fontWeight: 800, color: '#10b981', fontFamily: 'var(--font-mono)' }}>6 Phases</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Auditable Execution Trace</div>
          </div>
        </div>
      </section>

      {/* 2. Example Workflows & Query Templates */}
      <section style={{ maxWidth: '1120px', margin: '2rem auto 0', padding: '1rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.75rem' }}>
          <span className="badge" style={{ color: '#c084fc', background: 'rgba(192, 132, 252, 0.12)' }}>
            ILLUSTRATIVE WORKFLOW TEMPLATES
          </span>
          <h2 style={{ fontSize: '1.85rem', fontWeight: 800, marginTop: '0.5rem', color: '#ffffff' }}>
            Example Workflows & Query Templates
          </h2>
          <p style={{ color: 'var(--text-secondary)', fontSize: '0.9rem', maxWidth: '640px', margin: '0.4rem auto 0', lineHeight: 1.5 }}>
            Illustrative example workflows showing how SatQuery's multi-model evidence fusion can be applied to common Earth Observation tasks. These scenarios are demonstrations, not records of real missions.
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
            <div className="card" style={{
              padding: '2rem',
              background: 'rgba(13, 19, 34, 0.85)',
              border: '1px solid rgba(255, 255, 255, 0.12)',
              borderRadius: 16,
              boxShadow: '0 8px 32px rgba(0, 0, 0, 0.35)',
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '1rem', marginBottom: '1.5rem' }}>
                <div>
                  <h3 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.35rem' }}>
                    {c.title}
                  </h3>
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <span className="tag-pill" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)' }}>
                      EXAMPLE WORKFLOW
                    </span>
                  </div>
                </div>

                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'flex-end', gap: '0.5rem', maxWidth: '320px' }}>
                  <button 
                    className="btn btn-primary btn-sm"
                    onClick={() => onLaunchDemo && onLaunchDemo(c.presetId)}
                    style={{ display: 'inline-flex', alignItems: 'center', gap: '0.4rem', padding: '0.5rem 1rem' }}
                  >
                    <Play size={14} />
                    <span>Run This Workflow</span>
                  </button>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)', lineHeight: 1.45, textAlign: 'right' }}>
                    Demo images: clicking Run loads bundled benchmark sample scenes for demonstration — they are not imagery of the scenario described above.
                  </div>
                </div>
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
                    Illustrative Scenario
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

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                <Info size={15} color="var(--cyan-400)" style={{ flexShrink: 0 }} />
                <span><strong>Agent Capability:</strong> {c.capability}</span>
              </div>
            </div>
          );
        })()}
      </section>

      {/* 3. Active Specialist Model Ensemble */}
      <section style={{ maxWidth: '1120px', margin: '3rem auto 0', padding: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <span className="badge" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)', marginBottom: '0.4rem' }}>
              AGENT ARCHITECTURE
            </span>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
              <Cpu size={20} color="var(--cyan-400)" />
              <span>Specialist Model Ensemble Registry</span>
            </h2>
          </div>
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
                    {m.capabilities.map((cap, i) => (
                      <li key={i} style={{ fontSize: '0.8rem', color: '#e2e8f0', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                        <CheckCircle2 size={13} color="var(--cyan-400)" style={{ flexShrink: 0 }} />
                        <span>{cap}</span>
                      </li>
                    ))}
                  </ul>
                </div>
              </div>
            );
          })}
        </div>
      </section>

      {/* 4. Modality Synergy Matrix */}
      <section style={{ maxWidth: '1120px', margin: '3rem auto 0', padding: '1rem' }}>
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

      {/* 6. Calibrated Uncertainty & Anti-Hallucination Framework */}
      <section style={{ maxWidth: '1120px', margin: '3rem auto 0', padding: '1rem' }}>
        <div style={{
          padding: '1.75rem',
          borderRadius: 16,
          background: 'rgba(13, 19, 34, 0.85)',
          backdropFilter: 'blur(16px)',
          WebkitBackdropFilter: 'blur(16px)',
          border: '1px solid rgba(245, 158, 11, 0.35)',
          boxShadow: '0 8px 32px rgba(0, 0, 0, 0.45), 0 0 25px rgba(245, 158, 11, 0.08)',
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
