import React from 'react';
import { Send, Sparkles, CheckCircle2, AlertTriangle, ShieldCheck } from 'lucide-react';


const PROMPT_SUGGESTIONS = [
  "Describe the land-cover and major objects visible in this image.",
  "Identify and count the cargo vessels in this image.",
  "Highlight the water body referred to in the query.",
  "Use the optical and SAR images together to identify built-up and water-covered regions.",
  "What changed between these two dates, and where did the change occur?",
  "Analyze the multi-temporal timeline progression and cumulative land transformation from T1 to T3."
];

export default function QueryPanel({
  query,
  onQueryChange,
  canExecute,
  isExecuting,
  validationResult,
  onExecute,
}) {
  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) {
      e.preventDefault();
      if (canExecute && !isExecuting && validationResult?.valid !== false) {
        onExecute();
      }
    }
  };

  return (
    <div className="query-panel-card glass-panel">
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Sparkles size={18} color="var(--cyan-400)" />
          <h2 style={{ fontSize: '1.05rem', fontWeight: 700 }}>
            Vision-Language Query & Reasoning
          </h2>
        </div>
        <span className="tag-pill mono" style={{ fontSize: '0.72rem' }}>
          Deterministic intent classification & linear step planning
        </span>
      </div>

      <div className="query-input-wrap">
        <textarea
          className="query-textarea"
          placeholder="Ask a natural language query about the ingested satellite scene(s), e.g. 'Identify and count all cargo vessels in the harbor'..."
          value={query}
          onChange={(e) => onQueryChange(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={isExecuting}
        />
      </div>


      <div className="query-examples-bar">
        <span className="example-label">Prompt Suggestions:</span>
        {PROMPT_SUGGESTIONS.map((ex, i) => (
          <button
            key={i}
            className="example-pill"
            type="button"
            onClick={() => onQueryChange(ex)}
            disabled={isExecuting}
          >
            {ex}
          </button>
        ))}
      </div>

      <div className="query-actions-bar">
        <div className="validation-status-box">
          {validationResult?.valid ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--success)' }}>
              <ShieldCheck size={16} />
              <span>Image count and modality pairing checked (no CRS validation)</span>
            </div>
          ) : validationResult?.message ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--warning)' }}>
              <AlertTriangle size={16} />
              <span>{validationResult.message}</span>
            </div>
          ) : !canExecute ? (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--text-muted)' }}>
              <AlertTriangle size={16} />
              <span>Upload satellite scene(s) above to execute autonomous multi-modal analysis</span>
            </div>
          ) : (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', color: 'var(--cyan-400)' }}>
              <CheckCircle2 size={16} />
              <span>Ready for agentic task planning</span>
            </div>
          )}
        </div>

        <button
          className="btn btn-primary"
          onClick={onExecute}
          disabled={!canExecute || isExecuting || !query.trim() || validationResult?.valid === false}
          style={{ minWidth: '180px' }}
        >
          {isExecuting ? (
            <>
              <div className="spinner" />
              <span>Synthesizing Evidence...</span>
            </>
          ) : (
            <>
              <Send size={16} />
              <span>Execute Analysis</span>
            </>
          )}
        </button>
      </div>
    </div>
  );
}
