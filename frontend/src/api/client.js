/**
 * SatQuery AI API Client
 * Interfaces directly with FastAPI backend routes
 */

const BASE_URL = '';

async function handleResponse(res) {
  if (!res.ok) {
    let errorDetail = 'API request failed';
    try {
      const errorData = await res.json();
      if (typeof errorData.detail === 'string') {
        errorDetail = errorData.detail;
      } else if (typeof errorData.detail === 'object') {
        errorDetail = errorData.detail.message || JSON.stringify(errorData.detail);
      } else if (errorData.message) {
        errorDetail = errorData.message;
      }
    } catch {
      errorDetail = `HTTP ${res.status}: ${res.statusText}`;
    }
    throw new Error(errorDetail);
  }
  return res.json();
}

export const api = {
  async getHealth() {
    const res = await fetch(`${BASE_URL}/api/health`);
    return handleResponse(res);
  },

  async getRegistry() {
    const res = await fetch(`${BASE_URL}/api/registry`);
    return handleResponse(res);
  },

  async getBenchmarks() {
    const res = await fetch(`${BASE_URL}/api/benchmarks`);
    return handleResponse(res);
  },

  async createSession() {
    const res = await fetch(`${BASE_URL}/api/sessions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
    });
    return handleResponse(res);
  },

  async listSessions() {
    const res = await fetch(`${BASE_URL}/api/sessions`);
    return handleResponse(res);
  },

  async ingestImages(sessionId, files, modality = null) {
    const formData = new FormData();
    if (sessionId) {
      formData.append('session_id', sessionId);
    }
    if (modality) {
      formData.append('modality', modality);
    }
    for (const file of files) {
      formData.append('files', file);
    }

    const res = await fetch(`${BASE_URL}/api/ingest`, {
      method: 'POST',
      body: formData,
    });
    return handleResponse(res);
  },

  async validateInputs(sessionId, imageIds, task = null) {
    const res = await fetch(`${BASE_URL}/api/validate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        image_ids: imageIds,
        task: task || undefined,
      }),
    });
    return handleResponse(res);
  },

  async executeQuery(sessionId, query, imageIds, parameters = {}) {
    const res = await fetch(`${BASE_URL}/api/query`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        query,
        image_ids: imageIds,
        parameters,
      }),
    });
    return handleResponse(res);
  },

  getEvidenceUrl(filename, sessionId) {
    return `${BASE_URL}/api/evidence/${encodeURIComponent(filename)}?session_id=${encodeURIComponent(sessionId)}`;
  },

  getReportUrl(filename, sessionId) {
    return `${BASE_URL}/api/report/${encodeURIComponent(filename)}?session_id=${encodeURIComponent(sessionId)}`;
  },

  getTraceUrl(traceId, sessionId) {
    return `${BASE_URL}/api/trace/${encodeURIComponent(traceId)}?session_id=${encodeURIComponent(sessionId)}`;
  },

  async getSpectralIndices(sessionId, imageId) {
    const res = await fetch(`${BASE_URL}/api/spectral-indices`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        session_id: sessionId,
        image_id: imageId,
      }),
    });
    return handleResponse(res);
  },
};
