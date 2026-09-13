import React, { useState } from 'react';
import { Eye, Layers, Box, CheckCircle2, Image as ImageIcon, Sparkles, Clock, TrendingUp, ArrowRight, Download, MapPin, Activity, Map } from 'lucide-react';
import BeforeAfterSlider from './BeforeAfterSlider.jsx';
import SpectralIndexViewer from './SpectralIndexViewer.jsx';
import InteractiveMapViewer from './InteractiveMapViewer.jsx';

export default function EvidenceViewer({ result, slotImages, sessionId }) {
  const [activeTab, setActiveTab] = useState('evidence'); // 'evidence', 'slider', 'spectral', 'map', or slotId
  const [selectedEvidenceIndex, setSelectedEvidenceIndex] = useState(0);
  const [activeTransitionIndex, setActiveTransitionIndex] = useState(0);

  if (!result) return null;

  const evidenceUrl = result.evidence_url;
  const boxes = result.boxes || [];
  const evidenceList = Array.isArray(result.evidence) ? result.evidence : [];
  const temporalEvents = result.temporal_events || [];

  const availableSlots = Object.entries(slotImages || {}).filter(([_, env]) => !!env);
  const primaryEnv = availableSlots.length > 0 ? availableSlots[0][1] : null;

  // Genuine multispectral detection: modality flag or >=4 bands (band_count preferred, then bands)
  const getBandCount = (env) => {
    const parsed = Number(env?.band_count ?? env?.bands);
    return Number.isFinite(parsed) ? parsed : null;
  };
  const isMultispectralEnvelope = (env) => {
    if (!env) return false;
    if (env.modality === 'multispectral') return true;
    const bandCount = getBandCount(env);
    return bandCount !== null && bandCount >= 4;
  };
  // Georeferenced only when real bounds/CRS/GeoTIFF tags exist (no fabricated defaults)
  const isGeoreferencedEnvelope = (env) => {
    if (!env) return false;
    if (env.tags?.geotiff) return true;
    const b = env.bounds || env.geo_bbox;
    if (Array.isArray(b) && b.length === 4) return true;
    return env.crs != null && env.crs !== '';
  };

  // Spectral tab is only valid when the PRIMARY image is genuinely multispectral
  const primaryIsMultispectral = isMultispectralEnvelope(primaryEnv);
  const primaryIsGeoreferenced = isGeoreferencedEnvelope(primaryEnv);

  // Candidate image for spectral index calculation (multispectral scenes only)
  const spectralSlot = availableSlots.find(([_, env]) => isMultispectralEnvelope(env)) || availableSlots[0];
  const spectralCandidateImageId = spectralSlot ? (spectralSlot[1].image_id || spectralSlot[0]) : null;
  const spectralCandidateThumb = spectralSlot ? spectralSlot[1].thumbnail_base64 : null;

  // Dynamic image dimensions for SVG viewport alignment
  const refWidth = primaryEnv?.width || 512;
  const refHeight = primaryEnv?.height || 512;

  // Active highlighted box calculation
  let activeHighlightBox = null;
  let activeHighlightLabel = null;
  let activeHighlightGeo = null;

  if (selectedEvidenceIndex !== null && evidenceList[selectedEvidenceIndex]?.region?.length === 4) {
    const item = evidenceList[selectedEvidenceIndex];
    activeHighlightBox = item.region;
    activeHighlightLabel = item.description || item.category || item.type || 'Focused Region';
    activeHighlightGeo = item.geo_coordinates;
  } else if (temporalEvents.length > 0 && temporalEvents[activeTransitionIndex]) {
    const ev = temporalEvents[activeTransitionIndex];
    if (ev.region && ev.region.length === 4) {
      activeHighlightBox = ev.region;
      activeHighlightLabel = `${ev.transition}: ${ev.category}`;
      activeHighlightGeo = ev.geo_coordinates;
    }
  } else if (boxes.length > 0 && boxes[0].length === 4) {
    activeHighlightBox = boxes[0];
    activeHighlightLabel = 'Primary Target Feature';
  }

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
          {spectralCandidateImageId && primaryIsMultispectral && (
            <button
              className={`btn btn-sm ${activeTab === 'spectral' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => {
                setActiveTab('spectral');
                setSelectedEvidenceIndex(null);
              }}
              style={{ color: activeTab === 'spectral' ? undefined : '#10b981', border: '1px solid rgba(16, 185, 129, 0.35)' }}
              title="NDVI/NDWI (requires ≥4-band input; NIR is approximated for RGB)"
            >
              <Activity size={13} />
              <span>Spectral (NDVI/NDWI)</span>
            </button>
          )}

          {/* Interactive GIS Satellite Map Tab (only for genuinely georeferenced inputs) */}
          {primaryIsGeoreferenced ? (
            <button
              className={`btn btn-sm ${activeTab === 'map' ? 'btn-primary' : 'btn-ghost'}`}
              onClick={() => {
                setActiveTab('map');
                setSelectedEvidenceIndex(null);
              }}
              style={{ color: activeTab === 'map' ? undefined : '#06b6d4', border: '1px solid rgba(6, 182, 212, 0.35)' }}
              title="Interactive Esri World Imagery & OpenStreetMap GIS Basemap with real image bounds/CRS"
            >
              <Map size={13} />
              <span>GIS Map View</span>
            </button>
          ) : (
            <span
              className="tag-pill"
              style={{ display: 'inline-flex', alignItems: 'center', gap: '0.35rem', color: 'var(--text-muted)', border: '1px solid var(--border-subtle)', fontSize: '0.72rem' }}
              title="The primary image has no real bounds or CRS, so a GIS map cannot be shown."
            >
              <Map size={12} />
              <span>Not georeferenced — no map</span>
            </span>
          )}

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
              <span>Observation {i + 1} ({env.modality === 'multispectral' ? 'Multispectral MSI' : env.modality === 'sar' ? 'SAR Radar' : 'Optical RGB'})</span>
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
              isMultispectral={primaryIsMultispectral}
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
          <img
            src={evidenceUrl}
            alt="Visual Evidence Overlay"
            style={{ maxWidth: '100%', maxHeight: '520px', width: 'auto', height: 'auto', objectFit: 'contain' }}
          />
        ) : activeTab !== 'evidence' && slotImages?.[activeTab]?.thumbnail_base64 ? (
          <img
            src={slotImages[activeTab].thumbnail_base64}
            alt={slotImages[activeTab].filename}
            style={{ maxWidth: '100%', maxHeight: '520px', width: 'auto', height: 'auto', objectFit: 'contain' }}
          />
        ) : availableSlots.length > 0 && availableSlots[0][1]?.thumbnail_base64 ? (
          <img
            src={availableSlots[0][1].thumbnail_base64}
            alt={availableSlots[0][1].filename}
            style={{ maxWidth: '100%', maxHeight: '520px', width: 'auto', height: 'auto', objectFit: 'contain' }}
          />
        ) : (
          <div style={{ textAlign: 'center', color: 'var(--text-muted)' }}>
            <Layers size={36} style={{ marginBottom: '0.5rem', opacity: 0.5 }} />
            <p>No direct visual overlay generated for this query.</p>
          </div>
        )}

        {/* Text-only caption tasks intentionally have no localization boxes */}
        {!['spectral', 'map', 'slider'].includes(activeTab) && result.task === 'caption' && (
          <div className="evidence-caption-note">
            Captioning is text-only — no localization overlay for this task.
          </div>
        )}
        {!['spectral', 'map', 'slider'].includes(activeTab) &&
          result.task !== 'caption' &&
          boxes.length === 0 &&
          temporalEvents.length === 0 && (
            <div className="evidence-caption-note">
              No localization reported for this task.
            </div>
          )}
        {/* Interactive SVG Focus Highlight Overlay */}
        {!['spectral', 'map', 'slider'].includes(activeTab) && activeHighlightBox && (() => {
          const isFusionSplit = activeTab === 'evidence' && (
            result?.evidence_type === 'opt_sar_pair' ||
            result?.evidence_type === 'bi_temporal_pair' ||
            (evidenceUrl && (evidenceUrl.includes('fusion') || evidenceUrl.includes('pair')))
          );
          const vbWidth = isFusionSplit ? 1040 : refWidth;
          const vbHeight = isFusionSplit ? 552 : refHeight;
          const bw = Math.abs(activeHighlightBox[2] - activeHighlightBox[0]);
          const bh = Math.abs(activeHighlightBox[3] - activeHighlightBox[1]);
          const minX = Math.min(activeHighlightBox[0], activeHighlightBox[2]);
          const minY = Math.min(activeHighlightBox[1], activeHighlightBox[3]);

          if (isFusionSplit) {
            // Optical side (left: 0..512) and SAR side (right: 528..1040)
            const ox = minX;
            const oy = minY + 40;
            const sx = minX + 528;
            const sy = minY + 40;

            return (
              <svg
                style={{
                  position: 'absolute',
                  inset: 0,
                  width: '100%',
                  height: '100%',
                  pointerEvents: 'none',
                  zIndex: 10,
                }}
                viewBox={`0 0 ${vbWidth} ${vbHeight}`}
                preserveAspectRatio="xMidYMid meet"
              >
                {/* Optical Focus Rectangle */}
                <rect
                  x={ox}
                  y={oy}
                  width={bw}
                  height={bh}
                  fill="rgba(6, 182, 212, 0.22)"
                  stroke="var(--cyan-400)"
                  strokeWidth="3"
                  strokeDasharray="6 3"
                >
                  <animate attributeName="stroke-opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite" />
                </rect>

                {/* SAR Focus Rectangle */}
                <rect
                  x={sx}
                  y={sy}
                  width={bw}
                  height={bh}
                  fill="rgba(168, 85, 247, 0.22)"
                  stroke="#c084fc"
                  strokeWidth="3"
                  strokeDasharray="6 3"
                >
                  <animate attributeName="stroke-opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite" />
                </rect>

                {/* Optical Corner Markers */}
                <circle cx={ox} cy={oy} r="4" fill="#fff" />
                <circle cx={ox + bw} cy={oy} r="4" fill="#fff" />
                <circle cx={ox} cy={oy + bh} r="4" fill="#fff" />
                <circle cx={ox + bw} cy={oy + bh} r="4" fill="#fff" />

                {/* SAR Corner Markers */}
                <circle cx={sx} cy={sy} r="4" fill="#fff" />
                <circle cx={sx + bw} cy={sy} r="4" fill="#fff" />
                <circle cx={sx} cy={sy + bh} r="4" fill="#fff" />
                <circle cx={sx + bw} cy={sy + bh} r="4" fill="#fff" />
              </svg>
            );
          }

          // Standard Single-Image Overlay
          return (
            <svg
              style={{
                position: 'absolute',
                inset: 0,
                width: '100%',
                height: '100%',
                pointerEvents: 'none',
                zIndex: 10,
              }}
              viewBox={`0 0 ${vbWidth} ${vbHeight}`}
              preserveAspectRatio="xMidYMid meet"
            >
              {/* Secondary detected boxes in scene */}
              {boxes.map((box, bi) => {
                if (!box || box.length !== 4) return null;
                const bx = Math.min(box[0], box[2]);
                const by = Math.min(box[1], box[3]);
                const bbw = Math.abs(box[2] - box[0]);
                const bbh = Math.abs(box[3] - box[1]);
                if (bx === minX && by === minY && bbw === bw && bbh === bh) return null;
                const colors = ['#10b981', '#f59e0b', '#c084fc', '#38bdf8'];
                const c = colors[bi % colors.length];
                return (
                  <rect
                    key={bi}
                    x={bx}
                    y={by}
                    width={bbw}
                    height={bbh}
                    fill="rgba(16, 185, 129, 0.1)"
                    stroke={c}
                    strokeWidth="2"
                    strokeDasharray="4 2"
                  />
                );
              })}

              {/* Active Focused Target Box */}
              <rect
                x={minX}
                y={minY}
                width={bw}
                height={bh}
                fill="rgba(6, 182, 212, 0.22)"
                stroke="var(--cyan-400)"
                strokeWidth="3"
                strokeDasharray="6 3"
              >
                <animate attributeName="stroke-opacity" values="1;0.4;1" dur="1.5s" repeatCount="indefinite" />
              </rect>
              <circle cx={minX} cy={minY} r="4" fill="#fff" />
              <circle cx={minX + bw} cy={minY} r="4" fill="#fff" />
              <circle cx={minX + bw} cy={minY + bh} r="4" fill="#fff" />
              <circle cx={minX} cy={minY + bh} r="4" fill="#fff" />
            </svg>
          );
        })()}

        {/* Highlighted Region Coordinates Banner */}
        {!['spectral', 'map', 'slider'].includes(activeTab) && activeHighlightBox && (
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

        {boxes.length > 0 && activeTab === 'evidence' && (
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
                        color:
                          typeof ev.delta_pct === 'number'
                            ? ev.delta_pct >= 0
                              ? 'var(--success)'
                              : 'var(--warning)'
                            : 'var(--text-muted)',
                        background: 'rgba(0, 0, 0, 0.3)',
                      }}
                    >
                      {typeof ev.delta_pct === 'number'
                        ? ev.delta_pct >= 0
                          ? `+${ev.delta_pct}%`
                          : `${ev.delta_pct}%`
                        : 'N/A'}
                    </span>
                  </div>
                  <div style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', lineHeight: 1.3 }}>
                    {ev.category}
                  </div>
                  <div className="mono" style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: 'auto' }}>
                    Confidence: {typeof ev.confidence === 'number' ? `${(ev.confidence * 100).toFixed(0)}%` : 'N/A'}
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
              const itemConfidence = item.confidence ?? item.score;
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
                    {typeof itemConfidence === 'number' && (
                      <span className="tag-pill mono" style={{ color: 'var(--success)' }}>
                        {(itemConfidence * 100).toFixed(0)}%
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
