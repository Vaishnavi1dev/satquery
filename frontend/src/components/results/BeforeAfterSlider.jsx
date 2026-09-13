import React, { useState, useRef, useCallback, useEffect, useLayoutEffect } from 'react';

export default function BeforeAfterSlider({
  img1Url,
  img2Url,
  diffMaskUrl,
  diffOverlayUrl,
  label1 = 'Observation T1 (Baseline)',
  label2 = 'Observation T2 (Follow-up)'
}) {
  const [sliderPos, setSliderPos] = useState(50); // percentage 0..100
  const [isDragging, setIsDragging] = useState(false);
  const [showHeatmap, setShowHeatmap] = useState(false);
  const [containerWidth, setContainerWidth] = useState(0);
  const containerRef = useRef(null);

  useLayoutEffect(() => {
    const node = containerRef.current;
    if (!node) return undefined;
    const measure = () => {
      const width = node.getBoundingClientRect().width || node.clientWidth || 0;
      setContainerWidth(width > 0 ? width : 0);
    };
    measure();
    let observer = null;
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(measure);
      observer.observe(node);
    } else {
      window.addEventListener('resize', measure);
    }
    return () => {
      if (observer) observer.disconnect();
      else window.removeEventListener('resize', measure);
    };
  }, []);

  const handleMove = useCallback((clientX) => {
    if (!containerRef.current) return;
    const rect = containerRef.current.getBoundingClientRect();
    if (rect.width <= 0) return;
    const x = clientX - rect.left;
    const pct = Math.max(0, Math.min(100, (x / rect.width) * 100));
    setSliderPos(pct);
  }, []);

  const handleMouseDown = (e) => {
    setIsDragging(true);
    handleMove(e.clientX);
  };

  const handleTouchMove = (e) => {
    if (e.touches && e.touches[0]) {
      handleMove(e.touches[0].clientX);
    }
  };

  useEffect(() => {
    const handleMouseUp = () => setIsDragging(false);
    const handleMouseMoveWindow = (e) => {
      if (isDragging) {
        handleMove(e.clientX);
      }
    };

    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMoveWindow);
      window.addEventListener('mouseup', handleMouseUp);
    }
    return () => {
      window.removeEventListener('mousemove', handleMouseMoveWindow);
      window.removeEventListener('mouseup', handleMouseUp);
    };
  }, [isDragging, handleMove]);

  return (
    <div style={{ marginTop: '1rem', marginBottom: '1rem' }}>
      <div style={{ 
        display: 'flex', 
        justifyContent: 'space-between', 
        alignItems: 'center', 
        marginBottom: '0.5rem' 
      }}>
        <div style={{ fontSize: '0.8rem', fontWeight: 600, color: '#38bdf8', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
          <span>🛰️</span>
          <span>Interactive Split-Screen Difference Slider</span>
        </div>
        {diffMaskUrl && (
          <button
            type="button"
            onClick={() => setShowHeatmap(!showHeatmap)}
            style={{
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.4rem',
              padding: '0.25rem 0.6rem',
              fontSize: '0.72rem',
              fontWeight: 600,
              borderRadius: '6px',
              border: '1px solid',
              borderColor: showHeatmap ? '#f43f5e' : 'rgba(148, 163, 184, 0.3)',
              background: showHeatmap ? 'rgba(244, 63, 94, 0.2)' : 'rgba(15, 23, 42, 0.6)',
              color: showHeatmap ? '#fda4af' : '#94a3b8',
              cursor: 'pointer',
              transition: 'all 0.15s ease'
            }}
          >
            <span>🔥</span>
            <span>{showHeatmap ? 'Hide Difference Heatmap' : 'Highlight Change Heatmap'}</span>
          </button>
        )}
      </div>

      {/* Slider Container */}
      <div
        ref={containerRef}
        onMouseDown={handleMouseDown}
        onTouchMove={handleTouchMove}
        style={{
          position: 'relative',
          width: '100%',
          height: '420px',
          overflow: 'hidden',
          borderRadius: '10px',
          border: '1px solid rgba(56, 189, 248, 0.3)',
          background: '#0a0e17',
          cursor: 'ew-resize',
          userSelect: 'none'
        }}
      >
        {/* Layer 2: Image T2 (Full Width) */}
        <img
          src={showHeatmap && diffOverlayUrl ? diffOverlayUrl : img2Url}
          alt={label2}
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: '100%',
            height: '100%',
            objectFit: 'cover',
            pointerEvents: 'none'
          }}
        />

        {/* Optional Heatmap Mask Layer */}
        {showHeatmap && diffMaskUrl && !diffOverlayUrl && (
          <img
            src={diffMaskUrl}
            alt="Difference Heatmap Mask"
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: '100%',
              height: '100%',
              objectFit: 'cover',
              pointerEvents: 'none',
              mixBlendMode: 'screen',
              opacity: 0.95
            }}
          />
        )}

        {/* Layer 1: Image T1 (Clipped by slider position) */}
        <div
          style={{
            position: 'absolute',
            top: 0,
            left: 0,
            width: `${sliderPos}%`,
            height: '100%',
            overflow: 'hidden',
            borderRight: '2px solid #38bdf8',
            boxShadow: '2px 0 12px rgba(56, 189, 248, 0.5)'
          }}
        >
          <img
            src={img1Url}
            alt={label1}
            style={{
              position: 'absolute',
              top: 0,
              left: 0,
              width: containerWidth > 0 ? `${containerWidth}px` : '100%',
              maxWidth: 'none',
              height: '100%',
              objectFit: 'cover',
              pointerEvents: 'none'
            }}
          />
        </div>

        {/* Floating Labels */}
        <div style={{
          position: 'absolute',
          top: '10px',
          left: '12px',
          padding: '0.2rem 0.5rem',
          background: 'rgba(15, 23, 42, 0.8)',
          color: '#38bdf8',
          fontSize: '0.72rem',
          fontWeight: 600,
          borderRadius: '4px',
          border: '1px solid rgba(56, 189, 248, 0.3)',
          backdropFilter: 'blur(4px)',
          pointerEvents: 'none'
        }}>
          {label1}
        </div>

        <div style={{
          position: 'absolute',
          top: '10px',
          right: '12px',
          padding: '0.2rem 0.5rem',
          background: 'rgba(15, 23, 42, 0.8)',
          color: '#f59e0b',
          fontSize: '0.72rem',
          fontWeight: 600,
          borderRadius: '4px',
          border: '1px solid rgba(245, 158, 11, 0.3)',
          backdropFilter: 'blur(4px)',
          pointerEvents: 'none'
        }}>
          {label2}
        </div>

        {/* Draggable Divider Handle */}
        <div
          style={{
            position: 'absolute',
            top: '50%',
            left: `${sliderPos}%`,
            transform: 'translate(-50%, -50%)',
            width: '32px',
            height: '32px',
            borderRadius: '50%',
            background: '#38bdf8',
            border: '2px solid #ffffff',
            boxShadow: '0 0 15px rgba(56, 189, 248, 0.8)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            color: '#0f172a',
            fontSize: '0.75rem',
            fontWeight: 800,
            cursor: 'ew-resize',
            pointerEvents: 'none'
          }}
        >
          ↔
        </div>
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.68rem', color: '#64748b', marginTop: '0.25rem' }}>
        <span>Drag center handle to swipe between T1 and T2</span>
        <span>Click 'Highlight Change Heatmap' to reveal altered zones</span>
      </div>
    </div>
  );
}
