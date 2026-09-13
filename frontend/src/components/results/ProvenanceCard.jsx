import React from 'react';
import { Database, ShieldCheck, MapPin, Hash, Ruler } from 'lucide-react';

const hasRealBounds = (env) => Array.isArray(env?.bounds) && env.bounds.length === 4;

export default function ProvenanceCard({ slotImages }) {
  const envelopes = Object.values(slotImages || {}).filter(Boolean);

  if (envelopes.length === 0) return null;

  return (
    <div className="pillar-provenance-card glass-panel">
      <div className="pillar-header-badge tag-pill" style={{ background: 'rgba(192, 132, 252, 0.15)', color: 'var(--sar-color)' }}>
        PILLAR 4 • SENSOR & TELEMETRY PROVENANCE
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
        {envelopes.map((env) => (
          <div
            key={env.image_id}
            style={{
              padding: '1rem',
              borderRadius: '10px',
              background: 'rgba(0, 0, 0, 0.25)',
              border: '1px solid var(--border-subtle)',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem',
            }}
          >
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Database size={16} color="var(--sar-color)" />
                <span style={{ fontWeight: 600, fontSize: '0.9rem' }}>{env.filename}</span>
              </div>
              <span className={`tag-pill ${env.modality === 'sar' ? 'modality-sar' : env.modality === 'multispectral' ? 'modality-msi' : 'modality-opt'}`}>
                {env.modality === 'multispectral' ? 'MULTISPECTRAL' : env.modality === 'sar' ? 'SAR RADAR' : 'OPTICAL RGB'}
              </span>
            </div>

            <div className="provenance-grid">
              <div className="provenance-field">
                <span className="provenance-field-label">Platform (inferred from modality)</span>
                <span className="provenance-field-val">
                  {env.sensor
                    ? env.sensor
                    : env.modality === 'sar'
                      ? 'SAR (radar) — platform not provided'
                      : env.modality === 'multispectral'
                        ? 'Multispectral (MSI) — platform not provided'
                        : 'Optical RGB — platform not provided'}
                </span>
              </div>

              <div className="provenance-field">
                <span className="provenance-field-label">Spatial GSD</span>
                <span className="provenance-field-val">
                  {env.resolution_m != null ? `${env.resolution_m} m/pixel` : 'not provided'}
                </span>
              </div>

              <div className="provenance-field">
                <span className="provenance-field-label">Dimensions & Bands</span>
                <span className="provenance-field-val">
                  {env.width} × {env.height} px ({env.bands} {env.bands > 1 ? 'bands' : 'band'}, {env.dtype})
                </span>
              </div>

              <div className="provenance-field">
                <span className="provenance-field-label">Coordinate Reference</span>
                <span className="provenance-field-val">{env.crs ? env.crs : (hasRealBounds(env) ? 'not provided' : 'not georeferenced')}</span>
              </div>

              <div className="provenance-field" style={{ gridColumn: '1 / -1' }}>
                <span className="provenance-field-label">SHA-256 Cryptographic Hash</span>
                <span className="provenance-field-val mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {env.sha256}
                </span>
              </div>

              <div className="provenance-field" style={{ gridColumn: '1 / -1' }}>
                <span className="provenance-field-label">Bounding Footprint [W, S, E, N]</span>
                <span className="provenance-field-val mono" style={{ fontSize: '0.75rem' }}>
                  {hasRealBounds(env) ? env.bounds.map((b) => b.toFixed(4)).join(', ') : 'not georeferenced'}
                </span>
              </div>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
