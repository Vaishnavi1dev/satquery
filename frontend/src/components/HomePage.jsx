import React from 'react';
import {
  Satellite,
  Layers,
  Radio,
  Clock,
  ShieldCheck,
  Cpu,
  ArrowRight,
  Compass,
  CheckCircle2,
  Activity,
  Eye,
  Zap,
  Sparkles
} from 'lucide-react';

export default function HomePage({ onNavigateToAsk, onNavigateToIntelligence }) {
  return (
    <div className="home-page-container" style={{ position: 'relative', zIndex: 1, paddingBottom: '4rem' }}>

      {/* Hero Section */}
      <section className="home-hero-section" style={{
        textAlign: 'center',
        padding: '3.5rem 1rem 2.5rem',
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
          maxWidth: '720px',
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
            <span>Ask SatQuery</span>
            <ArrowRight size={18} />
          </button>


          <button
            className="btn btn-ghost"
            onClick={() => onNavigateToIntelligence()}
            style={{
              padding: '0.85rem 1.5rem',
              fontSize: '1rem',
              fontWeight: 600,
              background: 'rgba(255, 255, 255, 0.04)',
              border: '1px solid rgba(255, 255, 255, 0.15)',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.6rem',
            }}
          >
            <Compass size={18} color="var(--cyan-400)" />
            <span>Explore Intelligence Hub</span>
          </button>
        </div>

        {/* Stats Strip */}
        <div style={{
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          flexWrap: 'wrap',
          gap: '1.25rem',
          marginTop: '3.5rem',
          padding: '1.25rem 1.75rem',
          background: 'rgba(10, 16, 30, 0.65)',
          backdropFilter: 'blur(16px)',
          borderRadius: '16px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
        }}>
          <div style={{ flex: '1 1 auto', textAlign: 'center', minWidth: '140px' }}>
            <div style={{ fontSize: '1.65rem', fontWeight: 800, color: 'var(--cyan-400)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>6 Tools</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>Specialist AI Ensemble</div>
          </div>
          <div style={{ flex: '1.4 1 auto', textAlign: 'center', minWidth: '240px' }}>
            <div style={{ fontSize: '1.45rem', fontWeight: 800, color: 'var(--msi-color)', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap', letterSpacing: '-0.01em' }}>Optical • MSI • SAR</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>Tri-Modal Sensor Ingest</div>
          </div>
          <div style={{ flex: '1 1 auto', textAlign: 'center', minWidth: '150px' }}>
            <div style={{ fontSize: '1.65rem', fontWeight: 800, color: '#fbbf24', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>T1 ... TN</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>Multi-Temporal Sequences</div>
          </div>
          <div style={{ flex: '1.1 1 auto', textAlign: 'center', minWidth: '150px' }}>
            <div style={{ fontSize: '1.65rem', fontWeight: 800, color: '#10b981', fontFamily: 'var(--font-mono)', whiteSpace: 'nowrap' }}>6 Phases</div>
            <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textTransform: 'uppercase', letterSpacing: '0.05em', whiteSpace: 'nowrap' }}>Auditable Execution Trace</div>
          </div>
        </div>
      </section>

    </div>
  );
}
