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

      {/* Core Capabilities Section */}
      <section style={{ maxWidth: '1120px', margin: '0 auto', padding: '2rem 1rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '2.5rem' }}>
          <span className="badge" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)', marginBottom: '0.5rem' }}>
            CORE CAPABILITIES
          </span>
          <h2 style={{ fontSize: '2rem', fontWeight: 800, color: '#ffffff' }}>
            What Does SatQuery Do?
          </h2>
          <p style={{ color: 'var(--text-secondary)', maxWidth: '640px', margin: '0.5rem auto 0', fontSize: '0.95rem' }}>
            Traditional vision systems provide black-box text. SatQuery orchestrates specialized foundation models to produce auditable, grounded geospatial intelligence.
          </p>
        </div>

        <div style={{
          display: 'grid',
          gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))',
          gap: '1.25rem',
        }}>
          {/* Card 1 */}
          <div className="card" style={{ padding: '1.75rem', background: 'rgba(13, 19, 34, 0.75)' }}>
            <div style={{ width: 44, height: 44, borderRadius: 10, background: 'rgba(56, 189, 248, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem', border: '1px solid rgba(56, 189, 248, 0.3)' }}>
              <Eye size={22} color="var(--optical-color)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem', color: '#ffffff' }}>
              Evidence-Backed Answers
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.00rem', lineHeight: 1.55 }}>
              Ground responses with visual evidence-bounding boxes, annotated regions of interest, and category tags overlaid directly on satellite scenes.
            </p>
          </div>

          {/* Card 2 */}
          <div className="card" style={{ padding: '1.75rem', background: 'rgba(13, 19, 34, 0.75)' }}>
            <div style={{ width: 44, height: 44, borderRadius: 10, background: 'rgba(192, 132, 252, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem', border: '1px solid rgba(192, 132, 252, 0.3)' }}>
              <Radio size={22} color="var(--sar-color)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem', color: '#ffffff' }}>
              Optical + SAR + Multi-Spectral Cross-Modal Fusion
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.00rem', lineHeight: 1.55 }}>
              Fuse optical and multi-spectral imagery with all-weather SAR radar to penetrate clouds, smoke, and night while capturing structural dielectric returns.
            </p>
          </div>

          {/* Card 3 */}
          <div className="card" style={{ padding: '1.75rem', background: 'rgba(13, 19, 34, 0.75)' }}>
            <div style={{ width: 44, height: 44, borderRadius: 10, background: 'rgba(251, 191, 36, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem', border: '1px solid rgba(251, 191, 36, 0.3)' }}>
              <Clock size={22} color="var(--temporal-color)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem', color: '#ffffff' }}>
              Bi-Temporal & Sequence Analysis
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.00rem', lineHeight: 1.55 }}>
              Detect surface changes across dual epochs (T₁ → T₂) or track continuous environmental evolution across multi-temporal sequences (T₁ … T<sub>N</sub>).
            </p>
          </div>

          {/* Card 4 */}
          <div className="card" style={{ padding: '1.75rem', background: 'rgba(13, 19, 34, 0.75)' }}>
            <div style={{ width: 44, height: 44, borderRadius: 10, background: 'rgba(16, 185, 129, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem', border: '1px solid rgba(16, 185, 129, 0.3)' }}>
              <ShieldCheck size={22} color="var(--success)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem', color: '#ffffff' }}>
              Calibrated Confidence & Conflicts
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.00rem', lineHeight: 1.55 }}>
              Computes calibrated uncertainty scores and explicitly flags cross-sensor disagreements (e.g. optical cloud cover vs. SAR radar counts).
            </p>
          </div>

          {/* Card 5 */}
          <div className="card" style={{ padding: '1.75rem', background: 'rgba(13, 19, 34, 0.75)' }}>
            <div style={{ width: 44, height: 44, borderRadius: 10, background: 'rgba(6, 182, 212, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem', border: '1px solid rgba(6, 182, 212, 0.3)' }}>
              <Zap size={22} color="var(--cyan-400)" />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem', color: '#ffffff' }}>
              Autonomous Architecture Routing
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.00rem', lineHeight: 1.55 }}>
              Automatically inspects spectral bands, resolution, and question intent to orchestrate the ideal foundation model stack.
            </p>
          </div>

          {/* Card 6 */}
          <div className="card" style={{ padding: '1.75rem', background: 'rgba(13, 19, 34, 0.75)' }}>
            <div style={{ width: 44, height: 44, borderRadius: 10, background: 'rgba(148, 163, 184, 0.15)', display: 'flex', alignItems: 'center', justifyContent: 'center', marginBottom: '1.25rem', border: '1px solid rgba(148, 163, 184, 0.3)' }}>
              <Activity size={22} color="#94a3b8" />
            </div>
            <h3 style={{ fontSize: '1.2rem', fontWeight: 700, marginBottom: '0.5rem', color: '#ffffff' }}>
              6-Phase Explainable Trace
            </h3>
            <p style={{ color: 'var(--text-secondary)', fontSize: '1.00rem', lineHeight: 1.55 }}>
              Full auditable transparency across every phase-from input validation and model selection to evidence extraction and final synthesis.
            </p>
          </div>
        </div>
      </section>

      {/* How SatQuery Works: 4-Step Pipeline */}
      <section style={{ maxWidth: '1120px', margin: '2.5rem auto', padding: '2rem 1rem' }}>
        <div style={{
          background: 'linear-gradient(180deg, rgba(13, 19, 34, 0.85) 0%, rgba(8, 12, 22, 0.95) 100%)',
          borderRadius: '24px',
          border: '1px solid rgba(255, 255, 255, 0.08)',
          padding: '3.25rem 2.25rem',
        }}>
          <div style={{ textAlign: 'center', marginBottom: '3rem' }}>
            <span className="badge" style={{ color: 'var(--cyan-400)', background: 'rgba(6, 182, 212, 0.12)', fontSize: '0.85rem', padding: '0.35rem 0.9rem', letterSpacing: '0.06em' }}>
              WORKFLOW ARCHITECTURE
            </span>
            <h2 style={{ fontSize: '2.4rem', fontWeight: 800, marginTop: '0.75rem', color: '#ffffff' }}>
              How SatQuery Works
            </h2>
          </div>

          <div style={{
            display: 'grid',
            gridTemplateColumns: 'repeat(auto-fit, minmax(240px, 1fr))',
            gap: '1.75rem',
            position: 'relative',
          }}>
            <div style={{ padding: '1.75rem 1.5rem', borderRadius: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.92rem', color: 'var(--cyan-400)', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em' }}>
                  STAGE 01
                </span>
                <Layers size={20} color="var(--cyan-400)" />
              </div>
              <h4 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.75rem' }}>
                Ingest & Validate
              </h4>
              <p style={{ fontSize: '0.96rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
                Upload 1, 2, or multi-epoch satellite imagery. Sensor metadata, resolution (GSD), and spectral profiles are validated.
              </p>
            </div>

            <div style={{ padding: '1.75rem 1.5rem', borderRadius: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.92rem', color: '#c084fc', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em' }}>
                  STAGE 02
                </span>
                <Cpu size={20} color="#c084fc" />
              </div>
              <h4 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.75rem' }}>
                Autonomous Routing
              </h4>
              <p style={{ fontSize: '0.96rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
                The agent parses query intent and decompiles complex inquiries into specialist subtasks across optical, radar, or temporal tools.
              </p>
            </div>

            <div style={{ padding: '1.75rem 1.5rem', borderRadius: 16, background: 'rgba(255, 255, 255, 0.03)', border: '1px solid rgba(255, 255, 255, 0.08)' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.92rem', color: '#fbbf24', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em' }}>
                  STAGE 03
                </span>
                <Eye size={20} color="#fbbf24" />
              </div>
              <h4 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.75rem' }}>
                Evidence Extraction
              </h4>
              <p style={{ fontSize: '0.96rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
                Specialists run inference, generating localized bounding boxes, difference masks, and confidence distributions.
              </p>
            </div>

            <div style={{
              gridColumn: '1 / -1',
              maxWidth: '320px',
              width: '100%',
              justifySelf: 'center',
              margin: '0 auto',
              padding: '1.75rem 1.5rem',
              borderRadius: 16,
              background: 'rgba(255, 255, 255, 0.03)',
              border: '1px solid rgba(255, 255, 255, 0.08)'
            }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.92rem', color: '#10b981', fontFamily: 'var(--font-mono)', fontWeight: 800, letterSpacing: '0.08em' }}>
                  STAGE 04
                </span>
                <CheckCircle2 size={20} color="#10b981" />
              </div>
              <h4 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff', marginBottom: '0.75rem' }}>
                Auditable Synthesis
              </h4>
              <p style={{ fontSize: '0.96rem', color: 'var(--text-secondary)', lineHeight: 1.65 }}>
                Results are fused, calibrated for sensor conflicts, and rendered with interactive overlays and full provenance tracking.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Supported Constellations & Sensors */}
      <section style={{ maxWidth: '1120px', margin: '2rem auto', padding: '1rem' }}>
        <div style={{ textAlign: 'center', marginBottom: '1.5rem' }}>
          <h3 style={{ fontSize: '1.3rem', fontWeight: 700, color: '#ffffff' }}>
            Supported Sensor Modalities & Satellite Constellations
          </h3>
        </div>
        <div style={{
          display: 'flex',
          justifyContent: 'center',
          gap: '0.85rem',
          flexWrap: 'wrap',
        }}>
          {['Sentinel-1 (C-Band SAR)', 'Sentinel-2 (13-Band Multispectral)', 'PlanetScope (3m Daily Optical)', 'Maxar WorldView-3 (0.3m VHR)', 'Landsat-8/9 (Optical & Thermal)', 'Cartosat / RISAT (ISRO Sensors)'].map((sensor) => (
            <div
              key={sensor}
              style={{
                padding: '0.5rem 1rem',
                borderRadius: '20px',
                background: 'rgba(255, 255, 255, 0.04)',
                border: '1px solid rgba(255, 255, 255, 0.08)',
                fontSize: '1.00rem',
                color: 'var(--text-secondary)',
                display: 'flex',
                alignItems: 'center',
                gap: '0.45rem',
              }}
            >
              <Satellite size={13} color="var(--cyan-400)" />
              <span>{sensor}</span>
            </div>
          ))}
        </div>
      </section>

      {/* Bottom CTA Banner */}
      <section style={{
        maxWidth: '960px',
        margin: '3rem auto 0',
        padding: '2.5rem 2rem',
        borderRadius: '16px',
        background: 'radial-gradient(ellipse at center, rgba(6, 182, 212, 0.15) 0%, rgba(10, 16, 30, 0.85) 100%)',
        border: '1px solid rgba(6, 182, 212, 0.3)',
        textAlign: 'center',
      }}>
        <h3 style={{ fontSize: '1.75rem', fontWeight: 800, color: '#ffffff', marginBottom: '0.75rem' }}>
          Ready to Analyze Satellite Scenes?
        </h3>
        <p style={{ color: 'var(--text-secondary)', maxWidth: '560px', margin: '0 auto 1.5rem', fontSize: '0.92rem' }}>
          Jump directly into the workspace to upload your satellite imagery or try our 1-click presets for Cross-Modal Fusion, Bi-Temporal Changes, and Temporal Sequences.
        </p>
        <div style={{ display: 'flex', justifyContent: 'center', gap: '0.75rem', flexWrap: 'wrap' }}>
          <button
            className="btn btn-primary"
            onClick={() => onNavigateToAsk()}
            style={{ padding: '0.75rem 1.75rem', fontWeight: 700 }}
          >
            Launch Ask SatQuery
          </button>
          <button
            className="btn btn-ghost"
            onClick={() => onNavigateToAsk('cross_modal')}
            style={{ padding: '0.75rem 1.25rem', fontSize: '0.85rem' }}
          >
            Try Optical + SAR + Multi-Spectral Demo
          </button>
          <button
            className="btn btn-ghost"
            onClick={() => onNavigateToAsk('sequence')}
            style={{ padding: '0.75rem 1.25rem', fontSize: '0.85rem' }}
          >
            Try Temporal Sequence Demo
          </button>
        </div>
      </section>

    </div>
  );
}
