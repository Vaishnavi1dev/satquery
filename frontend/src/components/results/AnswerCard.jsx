import React, { useState } from 'react';
import { Bot, FileText, Check, Copy, ExternalLink, AlertTriangle, GitBranch, ArrowRight, ShieldAlert } from 'lucide-react';
import { api } from '../../api/client';

export default function AnswerCard({ result }) {
  const [copied, setCopied] = useState(false);

  if (!result) return null;

  const handleCopy = () => {
    navigator.clipboard.writeText(result.answer || '');
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const reportUrl = result.report_url 
    ? (result.report_url.startsWith('http') ? result.report_url : result.report_url)
    : null;

  return (
    <div className="pillar-answer-card glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div className="pillar-header-badge tag-pill" style={{ background: 'rgba(6, 182, 212, 0.15)', color: 'var(--cyan-400)' }}>
          PILLAR 1 • EVIDENCE-BACKED SYNTHESIS & MULTI-MODEL FUSION
        </div>

        {result.is_decomposed && (
          <span className="tag-pill mono" style={{ background: 'rgba(192, 132, 252, 0.15)', color: 'var(--sar-color)', border: '1px solid rgba(192, 132, 252, 0.3)' }}>
            <GitBranch size={12} style={{ display: 'inline', marginRight: 4, verticalAlign: -1 }} />
            DECOMPOSED WORKFLOW
          </span>
        )}
      </div>

      {/* Uncertainty / Conflict Flag Warning Banner */}
      {result.uncertainty_flag && (
        <div
          style={{
            margin: '0.75rem 0 1rem',
            padding: '0.85rem 1rem',
            borderRadius: '10px',
            background: 'rgba(245, 158, 11, 0.12)',
            border: '1px solid var(--warning)',
            color: '#fde68a',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.35rem',
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontWeight: 700, fontSize: '0.88rem', color: 'var(--warning)' }}>
            <AlertTriangle size={17} />
            <span>Uncertainty & Multi-Model Discrepancy Flag</span>
          </div>
          <div style={{ fontSize: '0.8rem', color: '#fef3c7', lineHeight: 1.4 }}>
            {result.uncertainty_explanation || 'Discrepancy or ambiguous sensor features observed between specialist predictions. Honest uncertainty bounds active; manual analyst audit recommended.'}
          </div>
        </div>
      )}

      {/* Query Decomposition Workflow Box */}
      {result.is_decomposed && result.decomposition_reasoning && (
        <div
          style={{
            margin: '0.5rem 0 1rem',
            padding: '0.75rem 1rem',
            borderRadius: '8px',
            background: 'rgba(0, 0, 0, 0.3)',
            border: '1px solid rgba(192, 132, 252, 0.25)',
            display: 'flex',
            flexDirection: 'column',
            gap: '0.4rem',
          }}
        >
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--sar-color)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
            Query Decomposition Plan:
          </div>
          <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            {result.decomposition_reasoning}
          </div>
          {result.subtasks && result.subtasks.length > 0 && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap', marginTop: '0.25rem' }}>
              {result.subtasks.map((st, i) => (
                <React.Fragment key={i}>
                  <span className="tag-pill mono" style={{ fontSize: '0.72rem', background: 'rgba(255, 255, 255, 0.06)' }}>
                    Subtask {i + 1}: {st.model} ({st.tool})
                  </span>
                  {i < result.subtasks.length - 1 && (
                    <ArrowRight size={12} color="var(--text-muted)" />
                  )}
                </React.Fragment>
              ))}
            </div>
          )}
        </div>
      )}

      <div className="answer-top-bar">
        <div className="model-attribution">
          <div className="specialist-badge">
            <Bot size={15} />
            <span>Specialist: {result.selected_model || result.selected_tool || 'EarthDial-4B'}</span>
          </div>

          {result.task && (
            <span className="tag-pill mono" style={{ background: 'rgba(255, 255, 255, 0.06)' }}>
              TASK: {result.task.toUpperCase()}
            </span>
          )}

          {result.duration_ms && (
            <span className="tag-pill mono" style={{ color: 'var(--text-muted)' }}>
              {result.duration_ms.toFixed(0)} ms
            </span>
          )}
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button className="btn btn-ghost btn-sm" onClick={handleCopy}>
            {copied ? <Check size={14} color="var(--success)" /> : <Copy size={14} />}
            <span>{copied ? 'Copied' : 'Copy'}</span>
          </button>

          {reportUrl && (
            <a
              href={reportUrl}
              target="_blank"
              rel="noopener noreferrer"
              className="btn btn-ghost btn-sm"
              style={{ color: 'var(--cyan-400)', borderColor: 'rgba(6, 182, 212, 0.3)' }}
            >
              <FileText size={14} />
              <span>Full HTML Report</span>
              <ExternalLink size={12} />
            </a>
          )}
        </div>
      </div>

      <div className="answer-body">
        {result.answer || "No response text generated."}
      </div>

      <div className="answer-actions-bar">
        <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
          Session: <code className="mono">{result.session_id}</code> • Trace ID: <code className="mono">{result.trace_id}</code>
        </span>
      </div>
    </div>
  );
}
