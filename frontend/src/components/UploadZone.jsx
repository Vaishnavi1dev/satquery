import React, { useState, useRef } from 'react';
import { UploadCloud, CheckCircle, Trash2, Layers, Plus, FileText, RotateCcw } from 'lucide-react';

export default function UploadZone({
  uploadedImages,      // ImageMetadataEnvelope[]
  onUploadFiles,       // async (File[]) => void
  onRemoveImage,       // (imageId) => void
  onStartFresh,        // () => void
  isUploading,         // boolean
}) {
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = async (e) => {
    e.preventDefault();
    setIsDragging(false);
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) {
      await onUploadFiles(files);
    }
  };

  const handleFileInputChange = async (e) => {
    const files = Array.from(e.target.files);
    if (files.length > 0) {
      await onUploadFiles(files);
      e.target.value = '';
    }
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
      <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '0.5rem' }}>
        <div>
          <h2 style={{ fontSize: '1.15rem', fontWeight: 700 }}>Telemetry & Imagery Ingestion</h2>
          <p style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>
            Upload single scenes, optical-SAR pairs, or multi-epoch sequences (GeoTIFF, TIFF, PNG, JPEG). The agent auto-identifies modalities.
          </p>
        </div>

        <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
          {uploadedImages.length > 0 && onStartFresh && (
            <button
              className="btn btn-ghost btn-sm"
              onClick={onStartFresh}
              disabled={isUploading}
              style={{
                color: '#f87171',
                borderColor: 'rgba(248, 113, 113, 0.35)',
                background: 'rgba(248, 113, 113, 0.08)',
                display: 'inline-flex',
                alignItems: 'center',
                gap: '0.35rem',
                fontWeight: 600
              }}
              title="Clear all uploaded imagery and reset workspace"
            >
              <RotateCcw size={13} />
              <span>Start Fresh</span>
            </button>
          )}

          <button
            className="btn btn-ghost btn-sm"
            onClick={() => fileInputRef.current?.click()}
            disabled={isUploading}
            style={{ color: 'var(--cyan-400)', borderColor: 'rgba(6, 182, 212, 0.3)' }}
          >
            <Plus size={14} />
            <span>Upload Satellite Image(s)</span>
          </button>
        </div>
      </div>

      {/* Uploaded Imagery Grid */}
      {uploadedImages.length > 0 && (
        <div className="upload-grid" style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(280px, 1fr))' }}>
          {uploadedImages.map((env, index) => (
            <div
              key={env.image_id || index}
              className="glass-panel"
              style={{
                padding: '1rem',
                borderRadius: '12px',
                display: 'flex',
                gap: '0.85rem',
                position: 'relative',
                background: 'rgba(15, 23, 42, 0.75)',
                border: '1px solid var(--border-subtle)',
              }}
            >
              <div className="preview-thumb-wrap" style={{ width: 85, height: 85 }}>
                {env.thumbnail_base64 ? (
                  <img src={env.thumbnail_base64} alt={env.filename} />
                ) : (
                  <Layers size={28} color="var(--text-muted)" />
                )}
              </div>

              <div className="preview-meta-wrap">
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', marginBottom: 2 }}>
                  <span className="step-circle" style={{ width: 18, height: 18, fontSize: '0.65rem' }}>
                    {index + 1}
                  </span>
                  <span className="preview-filename" title={env.filename} style={{ maxWidth: 140 }}>
                    {env.filename}
                  </span>
                </div>

                <div style={{ display: 'flex', gap: '0.35rem', flexWrap: 'wrap', margin: '2px 0' }}>
                  <span className={`tag-pill ${env.modality === 'sar' ? 'modality-sar' : env.modality === 'multispectral' ? 'modality-msi' : 'modality-opt'}`}>
                    ⚡ Auto: {env.modality === 'multispectral' ? 'Multispectral (MSI)' : env.modality === 'sar' ? 'SAR Radar' : 'Optical (RGB)'}
                  </span>
                  <span className="tag-pill mono">
                    {env.width}×{env.height}
                  </span>
                  <span className="tag-pill mono">
                    {env.bands} {env.bands > 1 ? 'bands' : 'band'}
                  </span>
                </div>

                <div className="mono" style={{ fontSize: '0.68rem', color: 'var(--text-muted)' }}>
                  {formatFileSize(env.file_size_bytes)} • ID: {env.image_id?.slice(0, 10)}
                </div>
              </div>

              <button
                className="btn btn-ghost btn-sm btn-icon"
                onClick={() => onRemoveImage(env.image_id)}
                title="Remove image"
                style={{ position: 'absolute', top: 6, right: 6, color: 'var(--danger)', width: 26, height: 26 }}
              >
                <Trash2 size={13} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Main Drag & Drop Box */}
      <div
        className={`dropzone-inner ${isDragging ? 'dragging' : ''}`}
        style={{
          padding: uploadedImages.length > 0 ? '1.25rem 1rem' : '2.5rem 1.5rem',
          border: isDragging ? '1px dashed var(--cyan-400)' : '1px dashed rgba(255, 255, 255, 0.16)',
          background: isDragging ? 'rgba(6, 182, 212, 0.08)' : 'rgba(10, 15, 26, 0.45)',
          backdropFilter: 'blur(12px)',
          borderRadius: '14px',
        }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
      >
        <input
          type="file"
          ref={fileInputRef}
          style={{ display: 'none' }}
          accept=".tif,.tiff,.png,.jpg,.jpeg"
          multiple
          onChange={handleFileInputChange}
        />

        {isUploading ? (
          <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.6rem' }}>
            <div className="spinner" style={{ color: 'var(--cyan-400)' }} />
            <span style={{ fontSize: '0.85rem', color: 'var(--cyan-400)', fontWeight: 600 }}>
              Ingesting GeoTIFF / Telemetry & Computing Hashes...
            </span>
          </div>
        ) : (
          <>
            <div className="dropzone-icon" style={{ width: 48, height: 48 }}>
              <UploadCloud size={26} />
            </div>
            <div style={{ fontSize: '0.95rem', fontWeight: 600 }}>
              {uploadedImages.length > 0 ? 'Drop additional satellite scenes to fuse or create sequences' : 'Drop satellite scenes here or click to browse'}
            </div>
            <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
              Supports GeoTIFF, TIFF, PNG, and JPEG. Single optical / multispectral (MSI) scenes, SAR radar, optical/MSI + SAR cross-modal pairs, and multi-temporal sequences supported.
            </div>
            <div style={{ display: 'flex', gap: '0.5rem', marginTop: 4 }}>
              <span className="tag-pill mono">Single (Optical / MSI / SAR)</span>
              <span className="tag-pill mono">Pairs (Optical/MSI + SAR / Change)</span>
              <span className="tag-pill mono">Sequences (T1...TN)</span>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
