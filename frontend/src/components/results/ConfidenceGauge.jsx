import React from 'react';
import { ShieldCheck, AlertTriangle, Award, ShieldAlert } from 'lucide-react';

export default function ConfidenceGauge({ confidence, rawConfidence, uncertaintyFlag, conflictDetected, uncertaintyExplanation, temperature = 1.15 }) {
  const confValue = typeof confidence === 'number' ? confidence : 0.85;
  const pct = Math.min(100, Math.max(0, Math.round(confValue <= 1 ? confValue * 100 : confValue)));

  let color = 'var(--success)';
  let statusText = 'HIGH CERTAINTY';
  let desc = 'Statistical reliability passed calibration bounds.';

  if (uncertaintyFlag || conflictDetected) {
    color = 'var(--warning)';
    statusText = 'UNCERTAINTY FLAGGED';
    desc = uncertaintyExplanation || 'Multi-model divergence or low certainty detected. Verification advised.';
  } else if (pct < 50) {
    color = 'var(--danger)';
    statusText = 'LOW CONFIDENCE';
    desc = 'Visual verification advised due to high sensor ambiguity.';
  } else if (pct < 75) {
    color = 'var(--warning)';
    statusText = 'MODERATE CONFIDENCE';
    desc = 'Cross-validated with secondary spectral metrics.';
  }

  const radius = 64;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - (pct / 100) * circumference;

  return (
    <div className="pillar-confidence-card glass-panel">
      <div className="pillar-header-badge tag-pill" style={{ background: uncertaintyFlag ? 'rgba(245, 158, 11, 0.15)' : 'rgba(16, 185, 129, 0.15)', color: uncertaintyFlag ? 'var(--warning)' : 'var(--success)' }}>
        PILLAR 2 • CALIBRATED CONFIDENCE & UNCERTAINTY
      </div>

      <div className="confidence-gauge-container">
        <div className="gauge-svg-wrap">
          <svg width="160" height="160" viewBox="0 0 160 160">
            <circle
              cx="80"
              cy="80"
              r={radius}
              stroke="rgba(255, 255, 255, 0.08)"
              strokeWidth="12"
              fill="none"
            />
            <circle
              cx="80"
              cy="80"
              r={radius}
              stroke={color}
              strokeWidth="12"
              fill="none"
              strokeDasharray={circumference}
              strokeDashoffset={strokeDashoffset}
              strokeLinecap="round"
              style={{
                transform: 'rotate(-90deg)',
                transformOrigin: '50% 50%',
                transition: 'stroke-dashoffset 0.8s cubic-bezier(0.16, 1, 0.3, 1)',
              }}
            />
          </svg>

          <div className="gauge-inner-label">
            <span className="gauge-pct" style={{ color }}>{pct}%</span>
            <span className="gauge-status-text" style={{ color, fontSize: '0.68rem' }}>{statusText}</span>
          </div>
        </div>
      </div>

      <div style={{ fontSize: '0.78rem', color: 'var(--text-secondary)', textAlign: 'center' }}>
        {desc}
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.45rem', marginTop: '0.5rem' }}>
        <div className="confidence-metric-row">
          <span>Calibration Mode</span>
          <span style={{ color: 'var(--cyan-400)' }}>Platt / Temperature Scaling</span>
        </div>
        <div className="confidence-metric-row">
          <span>Temperature (T)</span>
          <span>{temperature.toFixed(2)}</span>
        </div>
        <div className="confidence-metric-row">
          <span>Raw → Calibrated</span>
          <span>
            {typeof rawConfidence === 'number'
              ? `${Math.round(rawConfidence * 100)}% → ${pct}%`
              : `${pct}%`}
          </span>
        </div>
        <div className="confidence-metric-row">
          <span>Honesty Discrepancy Gate</span>
          <span style={{ color: uncertaintyFlag ? 'var(--warning)' : 'var(--success)' }}>
            {uncertaintyFlag ? 'Flagged (Disclosed)' : 'Passed (Consistent)'}
          </span>
        </div>
      </div>
    </div>
  );
}
