import React from 'react';
import { Satellite, History, PlusCircle, ExternalLink } from 'lucide-react';

export default function Header({ 
  sessionId, 
  systemStatus, 
  activeTab = 'home',
  onSelectTab,
  onNewSession, 
  onToggleHistory 
}) {
  return (
    <header className="app-header">
      <div className="brand-section" style={{ cursor: 'pointer' }} onClick={() => onSelectTab && onSelectTab('home')}>
        <div className="brand-logo-badge">
          <Satellite size={22} />
        </div>
        <div>
          <h1 className="brand-title" style={{ fontSize: '1.25rem', letterSpacing: '-0.02em' }}>
            SatQuery
          </h1>
          <div className="brand-subtitle">
            <span className="badge">ISRO PS 26167</span>
            <span>Multimodal Vision-Language Assistant</span>
          </div>
        </div>
      </div>

      <nav className="header-nav-links">
        <button 
          className={`nav-link ${activeTab === 'home' ? 'active' : ''}`}
          onClick={() => onSelectTab && onSelectTab('home')}
          style={{ background: 'none', border: 'none', cursor: 'pointer', font: 'inherit' }}
        >
          Home
        </button>
        <button 
          className={`nav-link ${activeTab === 'ask' ? 'active' : ''}`}
          onClick={() => onSelectTab && onSelectTab('ask')}
          style={{ background: 'none', border: 'none', cursor: 'pointer', font: 'inherit' }}
        >
          Ask SatQuery
        </button>
        <button 
          className={`nav-link ${activeTab === 'intelligence' ? 'active' : ''}`}
          onClick={() => onSelectTab && onSelectTab('intelligence')}
          style={{ background: 'none', border: 'none', cursor: 'pointer', font: 'inherit' }}
        >
          Intelligence
        </button>
      </nav>

      <div className="header-actions">
        <div className="system-status-indicator">
          <div className={`status-dot ${systemStatus === 'ONLINE' ? 'online' : systemStatus === 'BUSY' ? 'busy' : 'offline'}`} />
          <span className="mono" style={{ fontSize: '0.72rem' }}>
            {systemStatus === 'ONLINE' ? 'BACKEND READY' : systemStatus === 'BUSY' ? 'PROCESSING' : 'DISCONNECTED'}
          </span>
        </div>

        {sessionId && (
          <span className="tag-pill mono" title={`Session: ${sessionId}`}>
            {sessionId.slice(0, 12)}...
          </span>
        )}

        <button 
          className="btn btn-ghost btn-sm" 
          onClick={onNewSession}
          title="Start fresh session sandbox"
        >
          <PlusCircle size={14} />
          <span>New Session</span>
        </button>

        <button 
          className="btn btn-ghost btn-sm btn-icon" 
          onClick={onToggleHistory}
          title="Browse session history"
        >
          <History size={15} />
        </button>
      </div>
    </header>
  );
}
