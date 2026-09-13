import React, { useState, useEffect, useRef } from 'react';
import Header from './components/Header';
import SpaceBackground from './components/SpaceBackground';
import AgentRoutingBadge from './components/AgentRoutingBadge';
import UploadZone from './components/UploadZone';
import QueryPanel from './components/QueryPanel';
import SessionDrawer from './components/SessionDrawer';
import AnswerCard from './components/results/AnswerCard';
import ConfidenceGauge from './components/results/ConfidenceGauge';
import EvidenceViewer from './components/results/EvidenceViewer';
import ProvenanceCard from './components/results/ProvenanceCard';
import TraceTimeline from './components/results/TraceTimeline';
import HomePage from './components/HomePage';
import { api } from './api/client';
import { AlertOctagon } from 'lucide-react';

const DEMO_CASES = {
  cross_modal: {
    query: 'Use the optical and SAR images together to identify built-up and water-covered regions.',
    files: [
      ['/sample_imagery/pair_00000_optical.jpg', 'demo_optical_scene.jpg'],
      ['/sample_imagery/pair_00000_sar.jpg', 'demo_sar_scene.jpg'],
    ],
  },
  single: {
    query: 'Describe the land-cover and major visible objects in this image.',
    files: [
      ['/sample_imagery/pair_00000_optical.jpg', 'demo_optical_scene.jpg'],
    ],
  },
  change: {
    query: 'What changed between these two dates, and where did the change occur?',
    files: [
      ['/sample_imagery/pair_00001_optical.jpg', 'demo_epoch_t1.jpg'],
      ['/sample_imagery/pair_00002_optical.jpg', 'demo_epoch_t2.jpg'],
    ],
  },
  sequence: {
    query: 'Analyze the multi-temporal timeline progression and cumulative land transformation from T1 to T3.',
    files: [
      ['/sample_imagery/pair_00000_optical.jpg', 'demo_epoch_t1.jpg'],
      ['/sample_imagery/pair_00001_optical.jpg', 'demo_epoch_t2.jpg'],
      ['/sample_imagery/pair_00002_optical.jpg', 'demo_epoch_t3.jpg'],
    ],
  },
};

async function loadSampleFile(url, filename) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`Demo asset unavailable: ${filename}`);
  const blob = await response.blob();
  return new File([blob], filename, { type: blob.type || 'image/jpeg' });
}

export default function App() {
  const [activeTab, setActiveTab] = useState(() => {
    const hash = window.location.hash.replace('#', '').toLowerCase();
    if (['home', 'ask'].includes(hash)) return hash;
    return 'home';
  });
  const [sessionId, setSessionId] = useState('');
  const [sessions, setSessions] = useState([]);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [systemStatus, setSystemStatus] = useState('CHECKING'); // ONLINE, BUSY, OFFLINE
  const [uploadedImages, setUploadedImages] = useState([]); // ImageMetadataEnvelope[]
  const [isUploading, setIsUploading] = useState(false);
  const [query, setQuery] = useState('');
  const [validationResult, setValidationResult] = useState(null);
  const [isExecuting, setIsExecuting] = useState(false);
  const [executionResult, setExecutionResult] = useState(null);
  const [errorMessage, setErrorMessage] = useState(null);

  const resultsRef = useRef(null);

  // Initialize Session and Backend Health
  useEffect(() => {
    async function init() {
      try {
        const health = await api.getHealth();
        if (health.status === 'HEALTHY') {
          setSystemStatus('ONLINE');
        }
      } catch (err) {
        console.warn('Backend offline:', err);
        setSystemStatus('OFFLINE');
      }

      try {
        const sessRes = await api.createSession();
        setSessionId(sessRes.session_id);
        const listRes = await api.listSessions();
        if (listRes.sessions) {
          setSessions(listRes.sessions);
        }
      } catch (err) {
        console.error('Failed to init session:', err);
      }
    }
    init();
  }, []);

  // Auto-validate whenever active images or the query change (query changes debounced)
  useEffect(() => {
    let cancelled = false;
    async function runValidation() {
      if (sessionId && uploadedImages.length > 0) {
        try {
          const imageIds = uploadedImages.map((img) => img.image_id);
          const valRes = await api.validateInputs(sessionId, imageIds, null, query.trim() || null);
          if (!cancelled) setValidationResult(valRes);
        } catch (err) {
          if (!cancelled) setValidationResult({ valid: false, message: err.message });
        }
      } else {
        setValidationResult(null);
      }
    }
    const handle = setTimeout(runValidation, 300);
    return () => {
      cancelled = true;
      clearTimeout(handle);
    };
  }, [sessionId, uploadedImages, query]);

  // Upload multiple files
  const handleUploadFiles = async (files) => {
    try {
      setErrorMessage(null);
      setIsUploading(true);
      const res = await api.ingestImages(sessionId, files);
      if (res.images && res.images.length > 0) {
        setUploadedImages((prev) => [...prev, ...res.images]);
      }
    } catch (err) {
      setErrorMessage(`Upload failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleLoadDemo = async (presetId) => {
    const demo = DEMO_CASES[presetId];
    if (!demo || isUploading || !sessionId) return;
    try {
      setErrorMessage(null);
      setIsUploading(true);
      const files = await Promise.all(demo.files.map(([url, filename]) => loadSampleFile(url, filename)));
      const res = await api.ingestImages(sessionId, files);
      setUploadedImages(res.images || []);
      setQuery(demo.query);
      setExecutionResult(null);
    } catch (err) {
      setErrorMessage(`Demo load failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  const handleRemoveImage = (imageId) => {
    setUploadedImages((prev) => prev.filter((img) => img.image_id !== imageId));
    setExecutionResult(null);
  };


  // Execution Handler
  const handleExecute = async () => {
    if (uploadedImages.length === 0 || !query.trim() || isExecuting) return;
    if (validationResult && validationResult.valid === false) return;

    try {
      setIsExecuting(true);
      setSystemStatus('BUSY');
      setErrorMessage(null);

      const imageIds = uploadedImages.map((img) => img.image_id);
      const res = await api.executeQuery(sessionId, query.trim(), imageIds);
      setExecutionResult(res);

      setTimeout(() => {
        resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' });
      }, 150);
    } catch (err) {
      setErrorMessage(`Execution error: ${err.message}`);
      if (err.status === 422) {
        setValidationResult({ valid: false, message: err.message });
      }
    } finally {
      setIsExecuting(false);
      setSystemStatus('ONLINE');
    }
  };

  const handleStartFresh = async () => {
    try {
      setIsExecuting(false);
      setUploadedImages([]);
      setQuery('');
      setExecutionResult(null);
      setValidationResult(null);
      setErrorMessage(null);
      const res = await api.createSession();
      setSessionId(res.session_id);
      const listRes = await api.listSessions();
      if (listRes.sessions) setSessions(listRes.sessions);
      window.scrollTo({ top: 0, behavior: 'smooth' });
    } catch (err) {
      setErrorMessage(`Failed to reset workspace: ${err.message}`);
    }
  };

  const handleNewSession = handleStartFresh;

  const handleSelectSession = (sid) => {
    setSessionId(sid);
    setUploadedImages([]);
    setExecutionResult(null);
    setValidationResult(null);
  };

  // Listen for hash navigation changes
  useEffect(() => {
    const handleHashChange = () => {
      const hash = window.location.hash.replace('#', '').toLowerCase();
      if (['home', 'ask'].includes(hash)) {
        setActiveTab(hash);
      }
    };
    window.addEventListener('hashchange', handleHashChange);
    return () => window.removeEventListener('hashchange', handleHashChange);
  }, []);


  const handleSelectTab = (tab) => {
    setActiveTab(tab);
    window.location.hash = tab;
    window.scrollTo({ top: 0, behavior: 'smooth' });
  };

  // Map uploaded images dictionary for legacy components if needed
  const slotImagesMap = {};
  uploadedImages.forEach((env, i) => {
    slotImagesMap[env.image_id || `slot_${i}`] = env;
  });

  return (
    <>
      {/* Interactive Space Starfield & Earth Atmosphere Canvas */}
      <SpaceBackground />

      <Header
        sessionId={sessionId}
        systemStatus={systemStatus}
        activeTab={activeTab}
        onSelectTab={handleSelectTab}
        onNewSession={handleNewSession}
        onToggleHistory={() => setIsDrawerOpen(true)}
      />

      <main className="app-container" style={{ position: 'relative', zIndex: 1 }}>
        {/* Error Banner */}
        {errorMessage && (
          <div
            style={{
              padding: '0.85rem 1.25rem',
              borderRadius: '10px',
              background: 'rgba(239, 68, 68, 0.15)',
              border: '1px solid var(--danger)',
              color: '#fca5a5',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'space-between',
              gap: '1rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
              <AlertOctagon size={18} color="var(--danger)" />
              <span style={{ fontSize: '0.88rem' }}>{errorMessage}</span>
            </div>
            <button
              className="btn btn-ghost btn-sm"
              onClick={() => setErrorMessage(null)}
              style={{ padding: '2px 8px', fontSize: '0.75rem' }}
            >
              Dismiss
            </button>
          </div>
        )}

        {/* View 1: Home Page */}
        {activeTab === 'home' && (
          <HomePage onNavigateToAsk={() => handleSelectTab('ask')} />
        )}

        {/* View 3: Ask SatQuery Workspace */}
        {activeTab === 'ask' && (
          <>
            {/* Cinematic Space Hero matching Reference */}
            <section id="hero-section" className="space-hero-container">
              <div
                className="ask-satquery-capsule"
                style={{ cursor: 'pointer' }}
                onClick={() => document.getElementById('query-section')?.scrollIntoView({ behavior: 'smooth' })}
              >
                ASK SATQUERY
              </div>

              <div className="hero-titles-wrap">
                <h1 className="hero-main-title">SATQUERY</h1>
                <h2 className="hero-sub-title">REMOTE-SENSING</h2>
              </div>

              <div className="hero-earth-divider" />

              <div
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  gap: '1rem',
                  maxWidth: '680px',
                  margin: '0 auto',
                  textAlign: 'left',
                  background: 'rgba(10, 16, 30, 0.65)',
                  backdropFilter: 'blur(16px)',
                  padding: '1rem 1.25rem',
                  borderRadius: '12px',
                  border: '1px solid rgba(255, 255, 255, 0.08)',
                }}
              >
                <div
                  style={{
                    width: 3,
                    height: 44,
                    background: 'var(--cyan-400)',
                    boxShadow: '0 0 10px var(--cyan-400)',
                    borderRadius: 2,
                    flexShrink: 0,
                  }}
                />
                <p style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', lineHeight: 1.45 }}>
                  Autonomous Multimodal Remote Sensing Assistant. Ingest optical, optional multi-band, and SAR inputs plus multi-epoch sequences. The agent classifies the query and routes it to the appropriate specialist model.
                </p>
              </div>
            </section>

            {/* Autonomous Agent Routing Card (No manual architecture selection) */}
            <AgentRoutingBadge
              imagesCount={uploadedImages.length}
              imagesList={uploadedImages}
              query={query}
            />

            {/* Imagery & Telemetry Ingestion Zone */}
            <UploadZone
              uploadedImages={uploadedImages}
              onUploadFiles={handleUploadFiles}
              onRemoveImage={handleRemoveImage}
              onStartFresh={handleStartFresh}
              isUploading={isUploading}
            />

            {/* Vision-Language Query & Reasoning Panel */}
            <div id="query-section">
              <QueryPanel
                query={query}
                onQueryChange={setQuery}
                canExecute={uploadedImages.length > 0}
                isExecuting={isExecuting}
                validationResult={validationResult}
                onExecute={handleExecute}
              />
            </div>

            {/* Bundled Demo Samples */}
            <div className="glass-panel" style={{ marginTop: '0.75rem', padding: '0.85rem 1rem', display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '0.5rem' }}>
              <span style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                Demo Samples
              </span>
              {[
                ['single', 'Single Optical'],
                ['cross_modal', 'Optical + SAR'],
                ['change', 'Change (T1/T2)'],
                ['sequence', 'Sequence (T1-T3)'],
              ].map(([presetId, label]) => (
                <button
                  key={presetId}
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => handleLoadDemo(presetId)}
                  disabled={isUploading || isExecuting || !sessionId}
                >
                  {label}
                </button>
              ))}
              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                Bundled benchmark samples (not scenario imagery)
              </span>
            </div>

            {/* 5 Pillars Intelligence Dashboard */}
            {executionResult && (
              <div ref={resultsRef} className="intelligence-dashboard" id="results-section">
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.75rem' }}>
                  <div>
                    <h2 style={{ fontSize: '1.25rem', fontWeight: 800 }}>
                      5-Pillars Evidence-Backed Intelligence Dashboard
                    </h2>
                    <p style={{ fontSize: '0.82rem', color: 'var(--text-secondary)' }}>
                      Auditable, multimodal evidence synthesized for ISRO PS 26167.
                    </p>
                  </div>


                </div>

                {/* Pillar 1: Synthesized Answer */}
                <AnswerCard result={executionResult} />

                {/* Two-Column Row: Pillar 2 & Pillar 3 */}
                <div className="pillars-row-split">
                  {/* Pillar 2: Model-Reported Confidence */}
                  <ConfidenceGauge
                    confidence={executionResult.confidence}
                    uncertaintyFlag={executionResult.uncertainty_flag}
                    conflictDetected={executionResult.conflict_detected}
                    uncertaintyExplanation={executionResult.uncertainty_explanation}
                  />

                  {/* Pillar 3: Visual Evidence Overlays */}
                  <EvidenceViewer
                    result={executionResult}
                    slotImages={slotImagesMap}
                    sessionId={sessionId}
                  />
                </div>

                {/* Pillar 4: Spectral & Telemetry Provenance */}
                <ProvenanceCard slotImages={slotImagesMap} />

                {/* Pillar 5: Auditable Execution Trace */}
                <TraceTimeline result={executionResult} />
              </div>
            )}
          </>
        )}
      </main>

      {/* Session Sandbox Drawer */}
      <SessionDrawer
        isOpen={isDrawerOpen}
        onClose={() => setIsDrawerOpen(false)}
        sessions={sessions}
        currentSessionId={sessionId}
        onSelectSession={handleSelectSession}
        onNewSession={handleNewSession}
      />
    </>
  );
}
