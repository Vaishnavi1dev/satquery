import React from 'react';
import {
  Layers,
  Radio,
  Clock,
  ShieldCheck,
  Cpu,
  ArrowRight,
  CheckCircle2,
  TrendingUp,
  Sparkles
} from 'lucide-react';

export default function HomePage({ onNavigateToAsk }) {
  const specialistModels = [
    {
      id: 'earthdial-vqa',
      name: 'EarthDial-4B',
      role: 'Single-Scene Optical VQA / Captioning / Grounding',
      modality: 'Optical (Inputs Normalized to RGB)',
      resolution: '512x512 VLM Input',
      capabilities: ['Visual Question Answering (Optical)', 'Land-Cover Captioning', 'Text-Guided Region Grounding'],
      color: 'var(--optical-color)',
      bg: 'var(--optical-bg)',
      border: 'var(--optical-border)',
      icon: Layers,
    },
    {
      id: 'dofa',
      name: 'DOFA ViT-B + Cross-Attention Head',
      role: 'Cross-Modal Optical + SAR Fusion Specialist',
      modality: 'Size-Harmonized Optical + SAR Fusion',
      resolution: '512x512 Size-Harmonized',
      capabilities: ['Wavelength-Conditioned Feature Extraction', 'Optical <-> SAR Cross-Attention Fusion', 'Joint Land-Cover Analysis'],
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
      resolution: '512x512 Co-Registered',
      capabilities: ['Change Description & Change-VQA', 'Difference-Mask Localization', 'Pixel-Difference Change Regions (GeoJSON only when the input is georeferenced)'],
      color: 'var(--temporal-color)',
      bg: 'var(--temporal-bg)',
      border: 'var(--temporal-border)',
      icon: Clock,
    },
    {
      id: 'earthdial-sequence',
      name: 'EarthDial-4B Multi-Temporal Engine',
      role: 'T1...TN Sequence Specialist',
      modality: 'T1...TN Multi-Temporal Sequence',
      resolution: '512x512 per Epoch',
      capabilities: ['Per-Transition Measured Pixel Difference (T1 -> T2 -> ...)'],
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
      resolution: '512x512 Bounding-Box Output',
      capabilities: ['Bounding-Box Localization', 'Returns no box if it cannot localize', 'Referring-Expression Grounding'],
      color: 'var(--cyan-400)',
      bg: 'rgba(6, 182, 212, 0.12)',
      border: 'rgba(6, 182, 212, 0.3)',
      icon: Cpu,
    },
  ];

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
          optical scenes, optional multi-band inputs, and SAR radar backscatter, along with multi-temporal sequences,
          into evidence-backed answers with confidence and provenance.
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
            <div style={{ fontSize: '1.65rem', fontWeight: 800, color: 'var(--cyan-400)', fontFamily: 'var(--font-mono)' }}>6 Tool Adapters</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>3 Model Checkpoints</div>
          </div>
          <div style={{ flex: '1.4 1 auto', textAlign: 'center', minWidth: '240px' }}>
            <div style={{ fontSize: '1.45rem', fontWeight: 800, color: 'var(--msi-color)', fontFamily: 'var(--font-mono)' }}>Optical / SAR</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>Multi-band Inputs (RGB-Normalized)</div>
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

      {/* 2. Specialist Model Registry */}
      <section style={{ maxWidth: '1120px', margin: '3rem auto 0', padding: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1.25rem', flexWrap: 'wrap', gap: '0.5rem' }}>
          <div>
            <span className="badge" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)', marginBottom: '0.4rem' }}>
              AGENT ARCHITECTURE
            </span>
            <h2 style={{ fontSize: '1.4rem', fontWeight: 800, color: '#ffffff', display: 'flex', alignItems: 'center', gap: '0.5rem', marginTop: '0.2rem' }}>
              <Cpu size={20} color="var(--cyan-400)" />
              <span>Specialist Model Registry</span>
            </h2>
          </div>
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

                <div style={{ display: 'flex', alignItems: 'center', marginBottom: '1rem' }}>
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

      {/* 3. Uncertainty & Disagreement Reporting */}
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
              Uncertainty & Disagreement Reporting
            </h4>
            <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: 1.55 }}>
              For critical Earth Observation tasks, it matters that the system reports what it does and does not know.
              SatQuery applies post-hoc fixed-temperature logit rescaling (<em>T</em> = 1.15, not fitted), reports a
              <strong> cross-specialist divergence flag</strong> when specialist outputs contradict one another, and
              presents an <strong>averaged confidence</strong>.
            </p>
          </div>
        </div>
      </section>

    </div>
  );
}