import React, { useState, useEffect } from 'react';
import { Layers, Droplets, Leaf, Activity, Sparkles, RefreshCw, BarChart2 } from 'lucide-react';
import { api } from '../../api/client.js';

export default function SpectralIndexViewer({ sessionId, activeImageId, originalThumbnail, isMultispectral = false }) {
  const [indexMode, setIndexMode] = useState('ndvi'); // 'ndvi' or 'ndwi'
  const [data, setData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [blendOpacity, setBlendOpacity] = useState(0.85);

  useEffect(() => {
    let isCancelled = false;
    async function loadIndices() {
      if (!sessionId || !activeImageId) return;
      try {
        setIsLoading(true);
        setError(null);
        const res = await api.getSpectralIndices(sessionId, activeImageId);
        if (!isCancelled && res && res.indices) {
          setData(res.indices);
        }
      } catch (err) {
        if (!isCancelled) setError(err.message);
      } finally {
        if (!isCancelled) setIsLoading(false);
      }
    }
    loadIndices();
    return () => { isCancelled = true; };
  }, [sessionId, activeImageId]);

  if (isLoading) {
    return (
      <div className="glass-panel" style={{ padding: '2rem', textAlign: 'center' }}>
        <div className="spinner" style={{ color: 'var(--cyan-400)', margin: '0 auto 1rem' }} />
        <span style={{ fontSize: '0.9rem', color: 'var(--cyan-400)', fontWeight: 600 }}>
          Computing Pixel-Level NDVI & NDWI Spectral Datacubes...
        </span>
      </div>
    );
  }

  if (error || !data) {
    return (
      <div className="glass-panel" style={{ padding: '1.5rem', textAlign: 'center', color: 'var(--text-muted)' }}>
        <Activity size={24} style={{ marginBottom: 6, opacity: 0.6 }} />
        <p style={{ fontSize: '0.85rem' }}>
          {error ? `Spectral calculation failed: ${error}` : 'Select a multispectral or optical scene to compute spectral indices.'}
        </p>
      </div>
    );
  }

  const activeOverlay = indexMode === 'ndvi' ? data.ndvi_overlay_b64 : data.ndwi_overlay_b64;

  return (
    <div className="glass-panel" style={{ padding: '1.25rem', borderRadius: '12px', display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      {/* Header & Mode Switcher */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sparkles size={18} color="var(--cyan-400)" />
          <h3 style={{ fontSize: '0.95rem', fontWeight: 700 }}>Spectral Index Quick-Calculator</h3>
          {isMultispectral ? (
            <span className="tag-pill mono" style={{ fontSize: '0.7rem' }}>Multispectral (≥4 bands) - NIR available</span>
          ) : (
            <span
              className="tag-pill mono"
              style={{ fontSize: '0.7rem', color: '#f59e0b', border: '1px solid rgba(245, 158, 11, 0.45)' }}
              title="This input is not a true ≥4-band multispectral scene. NIR is synthesised from RGB."
            >
              NIR estimated from RGB — not true multispectral
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '0.4rem', background: 'rgba(0,0,0,0.4)', padding: '3px', borderRadius: '8px' }}>
          <button
            className={`btn btn-sm ${indexMode === 'ndvi' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setIndexMode('ndvi')}
            style={{ fontSize: '0.75rem', padding: '3px 10px' }}
          >
            <Leaf size={13} style={{ marginRight: 4 }} />
            <span>NDVI (Vegetation Vigor)</span>
          </button>
          <button
            className={`btn btn-sm ${indexMode === 'ndwi' ? 'btn-primary' : 'btn-ghost'}`}
            onClick={() => setIndexMode('ndwi')}
            style={{ fontSize: '0.75rem', padding: '3px 10px' }}
          >
            <Droplets size={13} style={{ marginRight: 4 }} />
            <span>NDWI (Water Delineation)</span>
          </button>
        </div>
      </div>

      {/* Metrics Row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(130px, 1fr))', gap: '0.65rem' }}>
        <div className="glass-panel" style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Mean NDVI</div>
          <div className="mono" style={{ fontSize: '1.15rem', fontWeight: 700, color: data.mean_ndvi > 0.3 ? '#4ade80' : '#facc15' }}>
            {data.mean_ndvi > 0 ? `+${data.mean_ndvi}` : data.mean_ndvi}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>{data.vegetation_vigor}</div>
        </div>

        <div className="glass-panel" style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Dense Canopy</div>
          <div className="mono" style={{ fontSize: '1.15rem', fontWeight: 700, color: '#22c55e' }}>
            {data.dense_vegetation_pct}%
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>NDVI &gt; 0.45</div>
        </div>

        <div className="glass-panel" style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Mean NDWI</div>
          <div className="mono" style={{ fontSize: '1.15rem', fontWeight: 700, color: '#38bdf8' }}>
            {data.mean_ndwi > 0 ? `+${data.mean_ndwi}` : data.mean_ndwi}
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>Water Absorptance</div>
        </div>

        <div className="glass-panel" style={{ padding: '0.75rem', borderRadius: '8px', background: 'rgba(15, 23, 42, 0.6)' }}>
          <div style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Water Surface</div>
          <div className="mono" style={{ fontSize: '1.15rem', fontWeight: 700, color: '#0284c7' }}>
            {data.water_body_pct}%
          </div>
          <div style={{ fontSize: '0.65rem', color: 'var(--text-secondary)' }}>NDWI &gt; 0.15</div>
        </div>
      </div>

      {/* Interactive Composite Image Display */}
      <div style={{ position: 'relative', width: '100%', maxHeight: '420px', overflow: 'hidden', borderRadius: '10px', background: '#070a12', display: 'flex', justifyContent: 'center' }}>
        {/* Original Base Image */}
        {originalThumbnail && (
          <img
            src={originalThumbnail}
            alt="Original Scene"
            style={{ width: '100%', maxHeight: '420px', objectFit: 'contain', display: 'block' }}
          />
        )}

        {/* Colormapped Spectral Overlay with Opacity */}
        {activeOverlay && (
          <img
            src={activeOverlay}
            alt={`${indexMode.toUpperCase()} Colormap Overlay`}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'contain',
              opacity: blendOpacity,
              mixBlendMode: 'screen',
              transition: 'opacity 0.2s ease',
            }}
          />
        )}
      </div>

      {/* Controls & Colormap Legend */}
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.75rem', fontSize: '0.75rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <span style={{ color: 'var(--text-muted)' }}>Overlay Blend:</span>
          <input
            type="range"
            min="0"
            max="1"
            step="0.05"
            value={blendOpacity}
            onChange={(e) => setBlendOpacity(parseFloat(e.target.value))}
            style={{ width: '90px', accentColor: 'var(--cyan-400)' }}
          />
          <span className="mono" style={{ color: 'var(--cyan-400)' }}>{Math.round(blendOpacity * 100)}%</span>
        </div>

        {/* Legend */}
        {indexMode === 'ndvi' ? (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>NDVI Legend:</span>
            <span className="tag-pill" style={{ background: '#b91c1c', color: '#fff', fontSize: '0.65rem' }}>Barren &lt; 0.1</span>
            <span className="tag-pill" style={{ background: '#ca8a04', color: '#fff', fontSize: '0.65rem' }}>Grass 0.2 - 0.4</span>
            <span className="tag-pill" style={{ background: '#15803d', color: '#fff', fontSize: '0.65rem' }}>Dense Forest &gt; 0.5</span>
          </div>
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ color: 'var(--text-muted)' }}>NDWI Legend:</span>
            <span className="tag-pill" style={{ background: '#78350f', color: '#fff', fontSize: '0.65rem' }}>Dry Land &lt; 0.0</span>
            <span className="tag-pill" style={{ background: '#0284c7', color: '#fff', fontSize: '0.65rem' }}>Moist 0.0 - 0.2</span>
            <span className="tag-pill" style={{ background: '#1e3a8a', color: '#fff', fontSize: '0.65rem' }}>Open Water &gt; 0.2</span>
          </div>
        )}
      </div>
    </div>
  );
}
