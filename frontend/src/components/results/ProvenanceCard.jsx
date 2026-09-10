import React from 'react';
import { Database, ShieldCheck, MapPin, Hash, Ruler } from 'lucide-react';

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
              <span className={`tag-pill ${env.modality === 'sar' ? 'modality-sar' : 'modality-opt'}`}>
                {env.modality.toUpperCase()}
              </span>
            </div>

            <div className="provenance-grid">
              <div className="provenance-field">
                <span className="provenance-field-label">Sensor Platform</span>
                <span className="provenance-field-val">
                  {env.sensor || (env.modality === 'sar' ? 'Sentinel-1 C-Band SAR' : 'Sentinel-2 MSI / Optical')}
                </span>
              </div>

              <div className="provenance-field">
                <span className="provenance-field-label">Spatial GSD</span>
                <span className="provenance-field-val">
                  {env.resolution_m ? `${env.resolution_m} m/pixel` : '10.0 m (Estimated)'}
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
                <span className="provenance-field-val">{env.crs || 'EPSG:4326 (WGS 84)'}</span>
              </div>

              <div className="provenance-field" style={{ gridColumn: '1 / -1' }}>
                <span className="provenance-field-label">SHA-256 Cryptographic Hash</span>
                <span className="provenance-field-val mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                  {env.sha256}
                </span>
              </div>

              {env.bounds && (
                <div className="provenance-field" style={{ gridColumn: '1 / -1' }}>
                  <span className="provenance-field-label">Bounding Footprint [W, S, E, N]</span>
                  <span className="provenance-field-val mono" style={{ fontSize: '0.75rem' }}>
                    {env.bounds.map((b) => b.toFixed(4)).join(', ')}
                  </span>
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
