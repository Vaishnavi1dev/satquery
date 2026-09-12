import React, { useState } from 'react';
import { Eye, Layers, Box, CheckCircle2, Image as ImageIcon, Sparkles, Clock, TrendingUp, ArrowRight, Download, MapPin, Activity, Map } from 'lucide-react';
import BeforeAfterSlider from './BeforeAfterSlider.jsx';
import SpectralIndexViewer from './SpectralIndexViewer.jsx';
import InteractiveMapViewer from './InteractiveMapViewer.jsx';

export default function EvidenceViewer({ result, slotImages, sessionId }) {
  const [activeTab, setActiveTab] = useState('evidence'); // 'evidence', 'slider', 'spectral', 'map', or slotId
  const [selectedEvidenceIndex, setSelectedEvidenceIndex] = useState(null);
  const [activeTransitionIndex, setActiveTransitionIndex] = useState(0);

  if (!result) return null;

  const evidenceUrl = result.evidence_url;
  const boxes = result.boxes || [];
  const evidenceList = Array.isArray(result.evidence) ? result.evidence : [];
  const temporalEvents = result.temporal_events || [];

  const availableSlots = Object.entries(slotImages || {}).filter(([_, env]) => !!env);

  // Find candidate image for spectral index calculation (prefer multispectral, then optical)
  const spectralSlot = availableSlots.find(([_, env]) => env.modality === 'multispectral')
    || availableSlots.find(([_, env]) => env.modality === 'optical')
    || availableSlots[0];
  const spectralCandidateImageId = spectralSlot ? (spectralSlot[1].image_id || spectralSlot[0]) : null;
  const spectralCandidateThumb = spectralSlot ? spectralSlot[1].thumbnail_base64 : null;

  // Active highlighted box calculation
  let activeHighlightBox = null;
  let activeHighlightLabel = null;
  let activeHighlightGeo = null;

  if (selectedEvidenceIndex !== null && evidenceList[selectedEvidenceIndex]) {
    const item = evidenceList[selectedEvidenceIndex];
    if (item.region && item.region.length === 4) {
      activeHighlightBox = item.region;
      activeHighlightLabel = item.category || item.type || 'Focused Region';
      activeHighlightGeo = item.geo_coordinates;
    }
  } else if (temporalEvents.length > 0 && temporalEvents[activeTransitionIndex]) {
    const ev = temporalEvents[activeTransitionIndex];
    if (ev.region && ev.region.length === 4) {
      activeHighlightBox = ev.region;
      activeHighlightLabel = `${ev.transition}: ${ev.category}`;
      activeHighlightGeo = ev.geo_coordinates;
    }
  }

  // Dimension reference (defaulting to 512 if not in env)
  const refWidth = 512;
  const refHeight = 512;

  const handleEvidenceClick = (index) => {
    if (selectedEvidenceIndex === index) {
      setSelectedEvidenceIndex(null);
    } else {
      setSelectedEvidenceIndex(index);
      setActiveTab('evidence');
    }
  };

  const handleTransitionClick = (index) => {
    setActiveTransitionIndex(index);
    setSelectedEvidenceIndex(null);
    setActiveTab('evidence');
  };

  return (
    <div className="pillar-evidence-card glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="pillar-header-badge tag-pill" style={{ background: 'rgba(56, 189, 248, 0.15)', color: 'var(--optical-color)' }}>
          PILLAR 3 • INTERACTIVE VISUAL EVIDENCE OVERLAYS
        </div>

        {/* View Switcher Tabs & GIS Export */}
        <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', alignItems: 'center' }}>
          {result.task === 'change_vqa' && availableSlots.length >= 2 && (
            <button
              className={`btn btn-sm ${activeTab === 'slider' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setActiveTab('slider')}
              style={{ color: activeTab === 'slider' ? undefined : '#38bdf8', border: '1px solid rgba(56, 189, 248, 0.3)' }}
            >
              <span>↔ Split Slider</span>
            </button>
          )}

          {evidenceUrl && (
            <button
              className={`btn btn-sm ${activeTab === 'evidence' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => setActiveTab('evidence')}
            >
              <Layers size={13} />
              <span>{temporalEvents.length > 0 ? 'Sequence Filmstrip' : 'Evidence Overlay'}</span>
            </button>
          )}

          {/* Spectral Index Calculation Tab (NDVI / NDWI) */}
          {spectralCandidateImageId && (
            <button
              className={`btn btn-sm ${activeTab === 'spectral' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => {
                setActiveTab('spectral');
                setSelectedEvidenceIndex(null);
              }}
              style={{ color: activeTab === 'spectral' ? undefined : '#10b981', border: '1px solid rgba(16, 185, 129, 0.35)' }}
              title="Compute real-time NDVI & NDWI vegetation and water indices"
            >
              <Activity size={13} />
              <span>Spectral (NDVI/NDWI)</span>
            </button>
          )}

          {/* Interactive GIS Satellite Map Tab */}
          <button
            className={`btn btn-sm ${activeTab === 'map' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => {
              setActiveTab('map');
              setSelectedEvidenceIndex(null);
            }}
            style={{ color: activeTab === 'map' ? undefined : '#06b6d4', border: '1px solid rgba(6, 182, 212, 0.35)' }}
            title="Interactive Esri World Imagery & OpenStreetMap GIS Basemap with WGS84 GeoJSON"
          >
            <Map size={13} />
            <span>GIS Map View</span>
          </button>

          {availableSlots.map(([slotId, env], i) => (
            <button
              key={slotId}
              className={`btn btn-sm ${activeTab === slotId ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => {
                setActiveTab(slotId);
                setSelectedEvidenceIndex(null);
              }}
            >
              <ImageIcon size={13} />
              <span>Observation {i + 1} ({env.modality})</span>
            </button>
          ))}

          {result.geojson_url && (
            <a
              href={result.geojson_url}
              download={`spatial_evidence_${result.trace_id}.geojson`}
              className="btn btn-sm btn-ghost"
              style={{ 
                display: 'inline-flex', 
                alignItems: 'center', 
                gap: '0.35rem', 
                color: '#10b981', 
                border: '1px solid rgba(16, 185, 129, 0.35)',
                textDecoration: 'none'
              }}
              title="Download standard GeoJSON for QGIS / ArcGIS"
            >
              <Download size={13} />
              <span>GeoJSON (GIS)</span>
            </a>
          )}
        </div>
      </div>

      {/* Interactive Image Display Viewport */}
      <div 
        className={`evidence-viewport ${['spectral', 'map', 'slider'].includes(activeTab) ? 'evidence-viewport-interactive' : ''}`}
        style={{ position: 'relative' }}
      >
        {activeTab === 'slider' && availableSlots.length >= 2 ? (
          <div style={{ width: '100%' }}>
            <BeforeAfterSlider
              img1Url={availableSlots[0][1].thumbnail_base64}
              img2Url={availableSlots[1][1].thumbnail_base64}
              diffMaskUrl={result.diff_mask_url}
              diffOverlayUrl={result.diff_overlay_url}
              label1={`T1: ${availableSlots[0][1].filename} (${availableSlots[0][1].modality})`}
              label2={`T2: ${availableSlots[1][1].filename} (${availableSlots[1][1].modality})`}
            />
          </div>
        ) : activeTab === 'spectral' ? (
          <div style={{ width: '100%' }}>
            <SpectralIndexViewer
              sessionId={sessionId}
              activeImageId={spectralCandidateImageId}
              originalThumbnail={spectralCandidateThumb}
            />
          </div>
        ) : activeTab === 'map' ? (
          <div style={{ width: '100%' }}>
            <InteractiveMapViewer
              result={result}
              slotImages={slotImages}
              sessionId={sessionId}
            />
          </div>
        ) : activeTab === 'evidence' && evidenceUrl ? (
          <img src={evidenceUrl} alt="Visual Evidence Overlay" style={{ width: '100%', height: '100%', objectFit: 'contain' }} />
        ) : activeTab !== 'evidence' && slotImages?.[activeTab]?.thumbnail_base64 ? (
          <img
            src={slotImages[activeTab].thumbnail_base64}
            alt={slotImages[activeTab].filename}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        ) : availableSlots.length > 0 && availableSlots[0][1]?.thumbnail_base64 ? (
          <img
            src={availableSlots[0][1].thumbnail_base64}
            alt={availableSlots[0][1].filename}
            style={{ width: '100%', height: '100%', objectFit: 'contain' }}
          />
        ) : (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
            <Layers size={36} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
            <p>No direct visual overlay generated for this query.</p>
          </div>
        )}

        {/* Interactive SVG Focus Highlight Overlay */}
        {activeHighlightBox && (
          <svg
            style={{
              position: 'absolute',
              inset: 0,
              width: '100%',
              height: '100%',
              pointerEvents: 'none',
              zIndex: 10,
            }}
            viewBox={`0 0 ${refWidth} ${refHeight}`}
          >
            {/* Pulsing Highlight Box */}
            <rect
              x={activeHighlightBox[0]}
              y={activeHighlightBox[1]}
              width={activeHighlightBox[2] - activeHighlightBox[0]}
              height={activeHighlightBox[3] - activeHighlightBox[1]}
              fill="rgba(6, 182, 212, 0.2)"
              stroke="var(--cyan-400)"
              strokeWidth="3"
              strokeDasharray="6 3"
            >
              <animate attributeName="stroke-opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite" />
            </rect>

            {/* Corner Markers */}
            <circle cx={activeHighlightBox[0]} cy={activeHighlightBox[1]} r="4" fill="#fff" />
            <circle cx={activeHighlightBox[2]} cy={activeHighlightBox[1]} r="4" fill="#fff" />
            <circle cx={activeHighlightBox[0]} cy={activeHighlightBox[3]} r="4" fill="#fff" />
            <circle cx={activeHighlightBox[2]} cy={activeHighlightBox[3]} r="4" fill="#fff" />
          </svg>
        )}

        {/* Highlighted Region Coordinates Banner */}
        {activeHighlightBox && (
          <div
            style={{
              position: 'absolute',
              top: 10,
              left: 10,
              background: 'rgba(7, 10, 18, 0.88)',
              backdropFilter: 'blur(10px)',
              padding: '0.4rem 0.75rem',
              borderRadius: '8px',
              border: '1px solid var(--cyan-400)',
              fontSize: '0.78rem',
              color: 'var(--text-primary)',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              zIndex: 20,
            }}
          >
            <Sparkles size={14} color="var(--cyan-400)" />
            <span style={{ fontWeight: 600 }}>{activeHighlightLabel}</span>
            <span className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              [{activeHighlightBox.join(', ')}]
            </span>
          </div>
        )}

        {boxes.length > 0 && (
          <div className="evidence-controls-overlay">
            <span className="tag-pill mono" style={{ color: 'var(--cyan-400)' }}>
              <Box size={12} style={{ display: 'inline', marginRight: 3, verticalAlign: -1 }} />
              {boxes.length} Bounding {boxes.length === 1 ? 'Box' : 'Boxes'} Detected
            </span>
          </div>
        )}
      </div>

      {/* Multi-Temporal Timeline Progression Scrubber */}
      {temporalEvents.length > 0 && (
        <div style={{ padding: '0.85rem', borderRadius: '10px', background: 'rgba(0, 0, 0, 0.35)', border: '1px solid var(--border-subtle)' }}>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.65rem' }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.45rem', fontSize: '0.82rem', fontWeight: 600, color: 'var(--temporal-color)' }}>
              <Clock size={15} />
              <span>Multi-Temporal Sequence Timeline (Click an epoch to inspect):</span>
            </div>
            <span className="tag-pill mono" style={{ color: 'var(--text-muted)' }}>
              {temporalEvents.length} Sequential Transitions
            </span>
          </div>

          <div style={{ display: 'grid', gridTemplateColumns: `repeat(${temporalEvents.length}, 1fr)`, gap: '0.6rem' }}>
            {temporalEvents.map((ev, i) => {
              const isSelected = activeTransitionIndex === i;
              return (
                <div
                  key={i}
                  onClick={() => handleTransitionClick(i)}
                  style={{
                    padding: '0.75rem',
                    borderRadius: '8px',
                    cursor: 'pointer',
                    background: isSelected ? 'rgba(251, 191, 36, 0.14)' : 'rgba(255, 255, 255, 0.03)',
                    border: `1px solid ${isSelected ? 'var(--temporal-color)' : 'var(--border-subtle)'}`,
                    transition: 'all 0.2s ease',
                    display: 'flex',
                    flexDirection: 'column',
                    gap: '0.3rem',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span className="mono" style={{ fontSize: '0.75rem', fontWeight: 700, color: isSelected ? 'var(--temporal-color)' : 'var(--text-primary)' }}>
                      {ev.transition}
                    </span>
                    <span
                      className="tag-pill mono"
                      style={{
                        fontSize: '0.68rem',
                        color: ev.delta_pct >= 0 ? 'var(--success)' : 'var(--warning)',
                        background: 'rgba(0, 0, 0, 0.3)',
                      }}
                    >
                      {ev.delta_pct >= 0 ? `+${ev.delta_pct}%` : `${ev.delta_pct}%`}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.3 }}>
                    {ev.category}
                  </div>
                  <div className="mono" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                    Confidence: {(ev.confidence * 100).toFixed(0)}%
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}

      {/* Clickable Structured Evidence Badges */}
      {evidenceList.length > 0 && (
        <div>
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
            <span style={{ fontSize: '0.78rem', fontWeight: 600, color: 'var(--text-secondary)' }}>
              Observable Evidence & Feature Predictions (Click to highlight in image):
            </span>
            {selectedEvidenceIndex !== null && (
              <button
                className="btn btn-ghost btn-sm"
                onClick={() => setSelectedEvidenceIndex(null)}
                style={{ padding: '1px 6px', fontSize: '0.7rem' }}
              >
                Clear Highlight
              </button>
            )}
          </div>

          <div className="evidence-badge-list">
            {evidenceList.map((item, idx) => {
              const isSelected = selectedEvidenceIndex === idx;
              return (
                <div
                  key={idx}
                  className="evidence-item-card"
                  onClick={() => handleEvidenceClick(idx)}
                  style={{
                    cursor: 'pointer',
                    borderColor: isSelected ? 'var(--cyan-400)' : 'var(--border-subtle)',
                    background: isSelected ? 'rgba(6, 182, 212, 0.12)' : 'rgba(0, 0, 0, 0.3)',
                    transform: isSelected ? 'translateY(-2px)' : 'none',
                    boxShadow: isSelected ? '0 0 15px rgba(6, 182, 212, 0.3)' : 'none',
                    transition: 'all 0.2s ease',
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <span className="evidence-item-type">
                      {item.type?.toUpperCase() || 'FEATURE'}
                    </span>
                    {item.confidence && (
                      <span className="tag-pill mono" style={{ color: 'var(--success)' }}>
                        {(item.confidence * 100).toFixed(0)}%
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                    Source: {item.source_model || item.source || 'Specialist Engine'}
                  </div>
                  {item.description && (
                    <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.3 }}>
                      {item.description}
                    </div>
                  )}
                  {item.region && (
                    <div className="mono" style={{ fontSize: '0.68rem', color: 'var(--cyan-400)', marginTop: 2 }}>
                      Pixel: [{item.region.join(', ')}]
                    </div>
                  )}
                  {item.geo_coordinates && (
                    <div className="mono" style={{ fontSize: '0.68rem', color: '#10b981', marginTop: 2, display: 'flex', alignItems: 'center', gap: '3px' }}>
                      <MapPin size={11} />
                      <span>{item.geo_coordinates}</span>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
