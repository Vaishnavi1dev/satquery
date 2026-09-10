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
import IntelligencePage from './components/IntelligencePage';
import { api } from './api/client';
import { Sparkles, AlertOctagon, Layers, Radio, Clock, ArrowDown, ChevronRight, Eye } from 'lucide-react';

export default function App() {
  const [activeTab, setActiveTab] = useState(() => {
    const hash = window.location.hash.replace('#', '').toLowerCase();
    if (['home', 'ask', 'intelligence'].includes(hash)) return hash;
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

  // Auto-validate whenever active images or query change
  useEffect(() => {
    async function runValidation() {
      if (sessionId && uploadedImages.length > 0) {
        try {
          const imageIds = uploadedImages.map((img) => img.image_id);
          const valRes = await api.validateInputs(sessionId, imageIds);
          setValidationResult(valRes);
        } catch (err) {
          setValidationResult({ valid: false, message: err.message });
        }
      } else {
        setValidationResult(null);
      }
    }
    runValidation();
  }, [sessionId, uploadedImages]);

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

  const handleRemoveImage = (imageId) => {
    setUploadedImages((prev) => prev.filter((img) => img.image_id !== imageId));
    setExecutionResult(null);
  };

  // Quick Demo Helper: Synthesizes realistic remote sensing scenes
  const handleLoadDemo = async (type) => {
    try {
      setErrorMessage(null);
      setIsUploading(true);

      const canvas = document.createElement('canvas');
      canvas.width = 512;
      canvas.height = 512;
      const ctx = canvas.getContext('2d');
      const makeBlob = () => new Promise((resolve) => canvas.toBlob(resolve, 'image/png'));

      let filesToUpload = [];

      if (type === 'optical') {
        // High-res optical seaport
        ctx.fillStyle = '#0f2b48';
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = '#2d5a27';
        ctx.beginPath();
        ctx.moveTo(180, 0);
        ctx.bezierCurveTo(240, 180, 140, 320, 200, 512);
        ctx.lineTo(0, 512);
        ctx.lineTo(0, 0);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = '#556575';
        ctx.fillRect(180, 200, 120, 16);
        ctx.fillRect(160, 270, 140, 18);
        ctx.fillStyle = '#d97706';
        ctx.fillRect(210, 192, 45, 14);
        ctx.fillStyle = '#dc2626';
        ctx.fillRect(230, 262, 50, 15);

        const b = await makeBlob();
        filesToUpload.push(new File([b], 'seaport_optical.png', { type: 'image/png' }));
        setQuery('Identify and count all cargo vessels docked along the harbor piers.');
      } else if (type === 'multispectral') {
        // High-resolution multispectral scene (agricultural parcels, vegetation, water channel)
        ctx.fillStyle = '#1e3a5f'; // Deep water channel (SWIR absorption)
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = '#16a34a'; // Healthy vegetation canopy (high NIR reflectance)
        ctx.beginPath();
        ctx.moveTo(140, 0);
        ctx.bezierCurveTo(220, 160, 120, 340, 190, 512);
        ctx.lineTo(0, 512);
        ctx.lineTo(0, 0);
        ctx.closePath();
        ctx.fill();
        ctx.fillStyle = '#84cc16'; // Active cropland (B04/B08 high NDVI)
        ctx.fillRect(220, 40, 120, 90);
        ctx.fillRect(360, 60, 120, 120);
        ctx.fillRect(230, 260, 150, 140);
        ctx.fillStyle = '#b45309'; // Dry soil / harvested parcel
        ctx.fillRect(370, 220, 100, 90);

        const bMsi = await makeBlob();
        filesToUpload.push(new File([bMsi], 'sentinel2_l2a_multispectral_b04_b08.png', { type: 'image/png' }));
        setQuery('Analyze the multispectral spectral profile: estimate vegetation density via NDVI and delineate water body boundaries.');
      } else if (type === 'single_sar') {
        // Single SAR radar observation (speckle + metallic returns)
        ctx.clearRect(0, 0, 512, 512);
        const imgData = ctx.createImageData(512, 512);
        for (let i = 0; i < imgData.data.length; i += 4) {
          const speckle = Math.floor(Math.random() * 60);
          imgData.data[i] = speckle;
          imgData.data[i + 1] = speckle;
          imgData.data[i + 2] = speckle;
          imgData.data[i + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        ctx.fillStyle = '#ffffff'; // Corner reflectors / metallic structures
        ctx.fillRect(200, 200, 40, 16);
        ctx.fillRect(270, 250, 36, 14);
        ctx.fillRect(160, 320, 50, 20);
        const bSar = await makeBlob();
        filesToUpload.push(new File([bSar], 'sentinel1_sar_vv_single.png', { type: 'image/png' }));
        setQuery('Examine SAR backscatter intensity: identify metallic infrastructure and explain dielectric double-bounce signatures.');
      } else if (type === 'cross_modal') {
        // Optical with cloud cover + SAR radar
        // 1. Optical (partially clouded)
        ctx.fillStyle = '#0f2b48';
        ctx.fillRect(0, 0, 512, 512);
        ctx.fillStyle = '#2d5a27';
        ctx.fillRect(0, 0, 220, 512);
        ctx.fillStyle = 'rgba(240, 245, 255, 0.75)'; // dense cloud
        ctx.beginPath();
        ctx.arc(260, 260, 140, 0, Math.PI * 2);
        ctx.fill();
        const bOpt = await makeBlob();
        filesToUpload.push(new File([bOpt], 'optical_clouded.png', { type: 'image/png' }));

        // 2. SAR radar (penetrates clouds)
        ctx.clearRect(0, 0, 512, 512);
        const imgData = ctx.createImageData(512, 512);
        for (let i = 0; i < imgData.data.length; i += 4) {
          const speckle = Math.floor(Math.random() * 70);
          imgData.data[i] = speckle;
          imgData.data[i + 1] = speckle;
          imgData.data[i + 2] = speckle;
          imgData.data[i + 3] = 255;
        }
        ctx.putImageData(imgData, 0, 0);
        ctx.fillStyle = '#ffffff'; // Strong metallic returns
        ctx.fillRect(240, 240, 32, 10);
        ctx.fillRect(290, 270, 28, 9);
        const bSar = await makeBlob();
        filesToUpload.push(new File([bSar], 'sentinel1_sar_vv.png', { type: 'image/png' }));
        setQuery('Detect metallic vessels through cloud cover by fusing optical scene and SAR radar backscatter.');
      } else if (type === 'sequence') {
        // 3-Epoch Temporal Sequence
        for (let ep = 1; ep <= 3; ep++) {
          ctx.clearRect(0, 0, 512, 512);
          ctx.fillStyle = '#0f2b48';
          ctx.fillRect(0, 0, 512, 512);
          ctx.fillStyle = ep === 1 ? '#2d5a27' : (ep === 2 ? '#3d5027' : '#454a2f');
          ctx.beginPath();
          ctx.moveTo(180, 0);
          ctx.bezierCurveTo(240, 180, 140, 320, 200, 512);
          ctx.lineTo(0, 512);
          ctx.lineTo(0, 0);
          ctx.closePath();
          ctx.fill();

          if (ep >= 2) {
            ctx.fillStyle = '#785535'; // excavation
            ctx.fillRect(190, 180, 80, 60);
          }
          if (ep >= 3) {
            ctx.fillStyle = '#334155'; // commercial buildings
            ctx.fillRect(180, 160, 130, 90);
            ctx.fillStyle = '#f59e0b';
            ctx.fillRect(210, 180, 45, 24);
          }
          const bEp = await makeBlob();
          filesToUpload.push(new File([bEp], `epoch_t${ep}.png`, { type: 'image/png' }));
        }
        setQuery('Analyze multi-temporal timeline progression and cumulative urban expansion from T1 to T3.');
      }

      const res = await api.ingestImages(sessionId, filesToUpload);
      if (res.images && res.images.length > 0) {
        setUploadedImages(res.images);
      }
    } catch (err) {
      setErrorMessage(`Demo loading failed: ${err.message}`);
    } finally {
      setIsUploading(false);
    }
  };

  // Execution Handler
  const handleExecute = async () => {
    if (uploadedImages.length === 0 || !query.trim() || isExecuting) return;

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
    } finally {
      setIsExecuting(false);
      setSystemStatus('ONLINE');
    }
  };

  const handleNewSession = async () => {
    try {
      const res = await api.createSession();
      setSessionId(res.session_id);
      setUploadedImages([]);
      setExecutionResult(null);
      setValidationResult(null);
      const listRes = await api.listSessions();
      if (listRes.sessions) setSessions(listRes.sessions);
    } catch (err) {
      setErrorMessage(`Failed to create session: ${err.message}`);
    }
  };

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
      if (['home', 'ask', 'intelligence'].includes(hash)) {
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
          <HomePage
            onNavigateToAsk={(preset) => {
              handleSelectTab('ask');
              if (preset) handleLoadDemo(preset);
            }}
            onNavigateToIntelligence={() => handleSelectTab('intelligence')}
          />
        )}

        {/* View 2: Intelligence Hub */}
        {activeTab === 'intelligence' && (
          <IntelligencePage
            onLaunchDemo={(preset) => {
              handleSelectTab('ask');
              if (preset) handleLoadDemo(preset);
            }}
            systemStatus={systemStatus}
          />
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
                  Autonomous Multimodal Remote Sensing Assistant. Ingest optical, multi-spectral (MSI), SAR radar backscatter, and multi-epoch temporal telemetry. The agent autonomously classifies, validates, and routes queries to specialized foundation models.
                </p>
              </div>
            </section>

            {/* 1-Click Quick Demo Presets */}
            <div
              style={{
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'space-between',
                padding: '0.75rem 1.25rem',
                borderRadius: '12px',
                background: 'rgba(6, 182, 212, 0.06)',
                border: '1px solid rgba(6, 182, 212, 0.2)',
                flexWrap: 'wrap',
                gap: '0.75rem',
              }}
            >
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Sparkles size={16} color="var(--cyan-400)" />
                <span style={{ fontSize: '0.82rem', color: 'var(--cyan-400)', fontWeight: 600 }}>
                  Try 1-Click Satellite Presets:
                </span>
                <span style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                  See autonomous agent architecture routing in action.
                </span>
              </div>

              <div style={{ display: 'flex', gap: '0.45rem', flexWrap: 'wrap' }}>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => handleLoadDemo('optical')}
                  disabled={isUploading}
                >
                  <Eye size={13} style={{ color: 'var(--optical-color)' }} />
                  <span>Single Optical</span>
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => handleLoadDemo('multispectral')}
                  disabled={isUploading}
                >
                  <Layers size={13} style={{ color: 'var(--msi-color)' }} />
                  <span>Single MSI (Multispectral)</span>
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => handleLoadDemo('single_sar')}
                  disabled={isUploading}
                >
                  <Radio size={13} style={{ color: 'var(--sar-color)' }} />
                  <span>Single SAR Radar</span>
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => handleLoadDemo('cross_modal')}
                  disabled={isUploading}
                >
                  <Radio size={13} />
                  <span>Optical + SAR + MSI Pair</span>
                </button>
                <button
                  className="btn btn-ghost btn-sm"
                  onClick={() => handleLoadDemo('sequence')}
                  disabled={isUploading}
                >
                  <Clock size={13} />
                  <span>T1...T3 Sequence</span>
                </button>
              </div>
            </div>

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
              isUploading={isUploading}
            />

            {/* Vision-Language Query & Reasoning Panel */}
            <div id="query-section">
              <QueryPanel
                query={query}
                onQueryChange={setQuery}
                selectedModality={uploadedImages.length >= 3 ? 'temporal_sequence' : uploadedImages.length === 2 ? 'cross_modal' : 'optical'}
                canExecute={uploadedImages.length > 0}
                isExecuting={isExecuting}
                validationResult={validationResult}
                onExecute={handleExecute}
              />
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
                      Calibrated, auditable, and multimodal evidence synthesized for ISRO PS 26167.
                    </p>
                  </div>

                  <span className="tag-pill" style={{ color: 'var(--success)', background: 'var(--success-glow)' }}>
                    AUDITED & VERIFIED
                  </span>
                </div>

                {/* Pillar 1: Synthesized Answer */}
                <AnswerCard result={executionResult} />

                {/* Two-Column Row: Pillar 2 & Pillar 3 */}
                <div className="pillars-row-split">
                  {/* Pillar 2: Calibrated Confidence */}
                  <ConfidenceGauge
                    confidence={executionResult.confidence}
                    uncertaintyFlag={executionResult.uncertainty_flag}
                    conflictDetected={executionResult.conflict_detected}
                    uncertaintyExplanation={executionResult.uncertainty_explanation}
                    temperature={1.15}
                  />

                  {/* Pillar 3: Visual Evidence Overlays */}
                  <EvidenceViewer
                    result={executionResult}
                    slotImages={slotImagesMap}
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
