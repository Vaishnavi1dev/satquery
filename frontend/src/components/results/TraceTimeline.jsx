import React, { useState } from 'react';
import { GitCommit, CheckCircle2, Clock, ChevronDown, ChevronUp, Download, Terminal, Layers, ShieldCheck } from 'lucide-react';
import { api } from '../../api/client';

export default function TraceTimeline({ result }) {
  const [showRawLogs, setShowRawLogs] = useState(false);

  if (!result) return null;

  const traceSteps = result.trace_view?.steps || [];
  const rawLogs = result.trace || [];
  const traceUrl = api.getTraceUrl(result.trace_id, result.session_id);

  // Canonical 6-phase pipeline (deduplicated, canonical order)
  const phaseOrder = ['InputValidation', 'TaskIdentification', 'ModelSelection', 'Execution', 'EvidenceCollection', 'FinalResult'];

  const phaseLabels = {
    InputValidation: '1. Input Compatibility Validation',
    TaskIdentification: '2. Task Identification & Query Decomposition',
    ModelSelection: '3. Model Selection & Linear Step Planning',
    Execution: '4. Specialist Model Inference Execution',
    EvidenceCollection: '5. Evidence Collection & Multi-Model Fusion',
    FinalResult: '6. Final Result Synthesis & Audit Reporting'
  };

  const groupedSteps = phaseOrder
    .map((phase) => {
      const events = traceSteps.filter((s) => s && (s.step_name === phase || (phase === 'Execution' && typeof s.step_name === 'string' && s.step_name.startsWith('step_'))));
      if (events.length === 0) return null;

      const succeeded = events.some((e) => e.status === 'SUCCESS' || e.status === 'COMPLETED');
      const durations = events
        .map((e) => (typeof e.duration_ms === 'number' ? e.duration_ms : 0))
        .filter((d) => d > 0);
      const durationMs = durations.length > 0 ? Math.max(...durations) : 0;

      const toolName = (events.find((e) => e.tool_name) || {}).tool_name
        || (events.find((e) => e.model_name) || {}).model_name
        || 'Agent Controller';

      const detailParts = [];
      events.forEach((e) => {
        const d = e.details !== undefined ? e.details : e.payload;
        if (d && typeof d === 'object') {
          Object.entries(d).forEach(([k, v]) => {
            if (v === undefined || v === null || v === '') return;
            detailParts.push(`${k}: ${typeof v === 'object' ? JSON.stringify(v) : String(v)}`);
          });
        } else if (d !== undefined && d !== null && d !== '') {
          detailParts.push(String(d));
        }
      });
      const detail = detailParts.join(' • ');

      return {
        step_name: phase,
        status: succeeded ? 'SUCCESS' : (events[0].status || 'COMPLETED'),
        tool_name: toolName,
        duration_ms: durationMs,
        detail,
      };
    })
    .filter(Boolean);

  const displaySteps = groupedSteps;

  return (
    <div className="pillar-trace-card glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <div className="pillar-header-badge tag-pill" style={{ background: 'rgba(251, 191, 36, 0.15)', color: 'var(--temporal-color)' }}>
            PILLAR 5 • EXPLAINABLE 6-PHASE EXECUTION TRACE
          </div>
          <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
            input validation → task identification → model selection → execution → evidence collection → final result
          </div>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem' }}>
          <button
            className="btn btn-ghost btn-sm"
            onClick={() => setShowRawLogs(!showRawLogs)}
          >
            <Terminal size={14} />
            <span>{showRawLogs ? 'Hide Logs' : 'Raw Trace Logs'}</span>
            {showRawLogs ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
          </button>

          <a
            href={traceUrl}
            target="_blank"
            rel="noopener noreferrer"
            className="btn btn-ghost btn-sm"
            title="Download full JSONL audit trace"
          >
            <Download size={14} />
            <span>Export JSONL</span>
          </a>
        </div>
      </div>

      {/* Step by Step Timeline */}
      <div className="trace-timeline" style={{ marginTop: '0.75rem' }}>
        {displaySteps.length === 0 && (
          <div className="mono" style={{ fontSize: '0.78rem', color: 'var(--text-muted)', padding: '0.5rem 0' }}>
            No execution trace recorded.
          </div>
        )}
        {displaySteps.map((step, idx) => {
          const title = phaseLabels[step.step_name] || step.step_name;
          return (
            <div key={idx} className="trace-step-row">
              <div className={`trace-step-node ${step.status === 'COMPLETED' || step.status === 'SUCCESS' ? 'completed' : ''}`}>
                <CheckCircle2 size={18} />
              </div>

              <div className="trace-step-content">
                <div className="trace-step-header">
                  <span className="trace-step-name">{title}</span>
                  <span className="trace-step-duration">
                    {step.duration_ms ? `${step.duration_ms.toFixed(1)} ms` : '< 1 ms'}
                  </span>
                </div>

                <div className="trace-step-meta">
                  <span style={{ color: 'var(--cyan-400)' }}>{step.tool_name || step.model_name || 'Agent Controller'}</span>
                  {step.detail ? (
                    <span style={{ color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                      • {step.detail}
                    </span>
                  ) : step.details ? (
                    <span style={{ color: 'var(--text-muted)', marginLeft: '0.5rem' }}>
                      • {typeof step.details === 'object' ? JSON.stringify(step.details) : String(step.details)}
                    </span>
                  ) : null}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Expandable Raw Logs Section */}
      {showRawLogs && (
        <div
          style={{
            marginTop: '1rem',
            padding: '1rem',
            borderRadius: '8px',
            background: 'rgba(0, 0, 0, 0.5)',
            border: '1px solid var(--border-subtle)',
            maxHeight: '260px',
            overflowY: 'auto',
          }}
        >
          <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', marginBottom: '0.5rem' }}>
            Event Stream & Execution Logs:
          </div>
          {rawLogs.length > 0 ? (
            rawLogs.map((log, i) => (
              <div key={i} className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-secondary)', padding: '2px 0' }}>
                {log}
              </div>
            ))
          ) : (
            <div className="mono" style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
              No intermediate console messages logged.
            </div>
          )}
        </div>
      )}
    </div>
  );
}
