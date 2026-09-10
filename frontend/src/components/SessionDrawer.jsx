import React from 'react';
import { X, Folder, Plus, Check } from 'lucide-react';

export default function SessionDrawer({
  isOpen,
  onClose,
  sessions,
  currentSessionId,
  onSelectSession,
  onNewSession,
}) {
  if (!isOpen) return null;

  return (
    <div className="drawer-backdrop" onClick={onClose}>
      <div className="drawer-panel" onClick={(e) => e.stopPropagation()}>
        <div className="drawer-header">
          <div>
            <h3 style={{ fontSize: '1.1rem', fontWeight: 700 }}>Active Sandboxes</h3>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Manage isolated session workspaces
            </p>
          </div>
          <button className="btn btn-ghost btn-sm btn-icon" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <button className="btn btn-primary" onClick={onNewSession} style={{ width: '100%' }}>
          <Plus size={16} />
          <span>Create New Session</span>
        </button>

        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.5rem', marginTop: '0.5rem' }}>
          {sessions && sessions.length > 0 ? (
            sessions.map((sid) => {
              const isActive = sid === currentSessionId;
              return (
                <div
                  key={sid}
                  className={`session-list-item ${isActive ? 'active' : ''}`}
                  onClick={() => {
                    onSelectSession(sid);
                    onClose();
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                      <Folder size={15} color={isActive ? 'var(--cyan-400)' : 'var(--text-muted)'} />
                      <span className="mono" style={{ fontSize: '0.8rem', fontWeight: 600 }}>
                        {sid}
                      </span>
                    </div>
                    {isActive && <Check size={14} color="var(--cyan-400)" />}
                  </div>
                  <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>
                    {isActive ? 'Current active workspace' : 'Click to load sandbox'}
                  </span>
                </div>
              );
            })
          ) : (
            <div style={{ textAlign: 'center', padding: '2rem 1rem', color: 'var(--text-muted)' }}>
              No sessions found on backend.
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
