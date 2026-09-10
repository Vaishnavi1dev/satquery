import React from 'react';

const PRESETS = [
  {
    id: 'flood',
    icon: '🌊',
    label: 'Flood & Inundation',
    category: 'Disaster Scan',
    query: 'Analyze flood inundation boundaries, water extent expansion, and cloud-penetrating radar backscatter change.'
  },
  {
    id: 'damage',
    icon: '🏚️',
    label: 'Building Damage (xBD)',
    category: 'Disaster Scan',
    query: 'Perform post-disaster structural damage assessment: categorize buildings as destroyed, major damage, or intact.'
  },
  {
    id: 'wildfire',
    icon: '🔥',
    label: 'Wildfire Burn Scars',
    category: 'Disaster Scan',
    query: 'Map wildfire burn scars, canopy degradation, and vegetative index drop between observations.'
  },
  {
    id: 'urban',
    icon: '🏗️',
    label: 'Urban Sprawl & Grading',
    category: 'Infrastructure',
    query: 'Identify newly constructed buildings, road foundations, and impervious surface growth between T1 and T2.'
  }
];

export default function DisasterPresets({ onSelectPreset, disabled }) {
  return (
    <div style={{ marginTop: '0.75rem', marginBottom: '0.5rem' }}>
      <div style={{ 
        display: 'flex', 
        alignItems: 'center', 
        gap: '0.5rem', 
        fontSize: '0.75rem', 
        color: '#94a3b8', 
        textTransform: 'uppercase', 
        letterSpacing: '0.05em',
        marginBottom: '0.4rem',
        fontWeight: 600
      }}>
        <span style={{ color: '#38bdf8' }}>⚡</span>
        <span>ISRO / Disaster Management Quick-Scans</span>
      </div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '0.4rem' }}>
        {PRESETS.map((p) => (
          <button
            key={p.id}
            type="button"
            disabled={disabled}
            onClick={() => onSelectPreset(p.query)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
              padding: '0.3rem 0.65rem',
              background: 'rgba(30, 41, 59, 0.7)',
              border: '1px solid rgba(56, 189, 248, 0.25)',
              borderRadius: '9999px',
              color: '#e2e8f0',
              fontSize: '0.75rem',
              cursor: disabled ? 'not-allowed' : 'pointer',
              transition: 'all 0.15s ease',
              backdropFilter: 'blur(8px)',
              fontWeight: 500
            }}
            onMouseEnter={(e) => {
              if (!disabled) {
                e.currentTarget.style.background = 'rgba(56, 189, 248, 0.15)';
                e.currentTarget.style.borderColor = '#38bdf8';
                e.currentTarget.style.transform = 'translateY(-1px)';
              }
            }}
            onMouseLeave={(e) => {
              if (!disabled) {
                e.currentTarget.style.background = 'rgba(30, 41, 59, 0.7)';
                e.currentTarget.style.borderColor = 'rgba(56, 189, 248, 0.25)';
                e.currentTarget.style.transform = 'translateY(0)';
              }
            }}
          >
            <span>{p.icon}</span>
            <span>{p.label}</span>
          </button>
        ))}
      </div>
    </div>
  );
}
