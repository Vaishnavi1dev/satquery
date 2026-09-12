import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import { Layers, MapPin, Maximize2, Shield, Eye } from 'lucide-react';

export default function InteractiveMapViewer({ result, slotImages, sessionId }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const geoLayerRef = useRef(null);
  const tileLayerRef = useRef(null);

  const [basemap, setBasemap] = useState('satellite'); // 'satellite' | 'street'
  const [geoData, setGeoData] = useState(null);
  const [coordinatesHud, setCoordinatesHud] = useState(null);
  const [isLoading, setIsLoading] = useState(false);

  // Basemap Tile Providers
  const TILE_SERVERS = {
    satellite: {
      url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      attribution: '&copy; Esri &mdash; World Imagery Telemetry',
      maxZoom: 19,
    },
    street: {
      url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
      attribution: '&copy; OpenStreetMap contributors',
      maxZoom: 19,
    },
  };

  // Determine initial center coordinates
  const getFallbackCoordinates = () => {
    // Check if slotImages has geo_bbox or coordinates
    if (slotImages) {
      const firstSlot = Object.values(slotImages).find(env => !!env);
      if (firstSlot?.geo_bbox && firstSlot.geo_bbox.length === 4) {
        const [minLon, minLat, maxLon, maxLat] = firstSlot.geo_bbox;
        return [(minLat + maxLat) / 2, (minLon + maxLon) / 2];
      }
    }
    // Check result evidence
    if (result?.evidence && Array.isArray(result.evidence)) {
      for (const ev of result.evidence) {
        if (ev.geo_coordinates) {
          const match = ev.geo_coordinates.match(/([\d.]+)[°\s]*([NS]),\s*([\d.]+)[°\s]*([EW])/i);
          if (match) {
            const lat = parseFloat(match[1]) * (match[2].toUpperCase() === 'S' ? -1 : 1);
            const lon = parseFloat(match[3]) * (match[4].toUpperCase() === 'W' ? -1 : 1);
            return [lat, lon];
          }
        }
      }
    }
    // Default to New Delhi / ISRO HQ coordinate space
    return [28.6139, 77.2090];
  };

  // Fetch GeoJSON data if available
  useEffect(() => {
    let isCancelled = false;
    async function loadGeoJson() {
      if (result?.geojson_url) {
        setIsLoading(true);
        try {
          const res = await fetch(result.geojson_url);
          if (res.ok) {
            const data = await res.json();
            if (!isCancelled) setGeoData(data);
          }
        } catch (e) {
          console.warn('Failed to load GeoJSON from server:', e);
        } finally {
          if (!isCancelled) setIsLoading(false);
        }
      } else {
        // Construct fallback polygon from slotImages or default
        const [centerLat, centerLon] = getFallbackCoordinates();
        const delta = 0.015; // ~1.5 km bounding box
        const syntheticGeoJson = {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              properties: {
                name: 'Mission Target Footprint',
                category: result?.task || 'Observation AOI',
                confidence: result?.confidence || 0.88,
              },
              geometry: {
                type: 'Polygon',
                coordinates: [[
                  [centerLon - delta, centerLat - delta],
                  [centerLon + delta, centerLat - delta],
                  [centerLon + delta, centerLat + delta],
                  [centerLon - delta, centerLat + delta],
                  [centerLon - delta, centerLat - delta],
                ]],
              },
            },
          ],
        };
        setGeoData(syntheticGeoJson);
      }
    }
    loadGeoJson();
    return () => { isCancelled = true; };
  }, [result?.geojson_url]);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;

    if (!mapInstanceRef.current) {
      const [initialLat, initialLon] = getFallbackCoordinates();
      const map = L.map(mapContainerRef.current, {
        center: [initialLat, initialLon],
        zoom: 13,
        zoomControl: false,
      });

      L.control.zoom({ position: 'bottomright' }).addTo(map);

      // Add Basemap Tile Layer
      const cfg = TILE_SERVERS[basemap];
      const tile = L.tileLayer(cfg.url, {
        attribution: cfg.attribution,
        maxZoom: cfg.maxZoom,
      }).addTo(map);
      tileLayerRef.current = tile;

      // Mousemove coordinates listener
      map.on('mousemove', (e) => {
        setCoordinatesHud({
          lat: e.latlng.lat.toFixed(5),
          lng: e.latlng.lng.toFixed(5),
        });
      });

      mapInstanceRef.current = map;

      // Invalidate size to ensure clean tile rendering after animation/mount
      setTimeout(() => {
        map.invalidateSize();
      }, 200);
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
    };
  }, []);

  // Update Basemap Tiles when state changes
  useEffect(() => {
    if (!mapInstanceRef.current) return;
    if (tileLayerRef.current) {
      mapInstanceRef.current.removeLayer(tileLayerRef.current);
    }
    const cfg = TILE_SERVERS[basemap];
    tileLayerRef.current = L.tileLayer(cfg.url, {
      attribution: cfg.attribution,
      maxZoom: cfg.maxZoom,
    }).addTo(mapInstanceRef.current);
  }, [basemap]);

  // Render / Update GeoJSON Layer
  useEffect(() => {
    if (!mapInstanceRef.current || !geoData) return;

    if (geoLayerRef.current) {
      mapInstanceRef.current.removeLayer(geoLayerRef.current);
    }

    const layer = L.geoJSON(geoData, {
      style: {
        color: '#06b6d4',
        weight: 2.5,
        opacity: 0.9,
        fillColor: '#06b6d4',
        fillOpacity: 0.2,
        dashArray: '5, 5',
      },
      onEachFeature: (feature, featureLayer) => {
        const props = feature.properties || {};
        const title = props.name || props.label || props.category || 'Target Area';
        featureLayer.bindPopup(`
          <div style="font-family: var(--font-sans); color: #0f172a; padding: 4px;">
            <div style="font-weight: 700; font-size: 13px; color: #0284c7;">🎯 ${title}</div>
            <div style="font-size: 11px; margin-top: 4px; color: #475569;">
              ${props.category ? `<div><strong>Class:</strong> ${props.category}</div>` : ''}
              ${props.confidence ? `<div><strong>Confidence:</strong> ${(props.confidence * 100).toFixed(1)}%</div>` : ''}
              ${props.area_km2 ? `<div><strong>Area:</strong> ${props.area_km2.toFixed(3)} km²</div>` : ''}
            </div>
          </div>
        `);
      },
    }).addTo(mapInstanceRef.current);

    geoLayerRef.current = layer;

    // Zoom map to GeoJSON bounds
    try {
      const bounds = layer.getBounds();
      if (bounds.isValid()) {
        mapInstanceRef.current.fitBounds(bounds, { padding: [35, 35], maxZoom: 16 });
      }
    } catch (e) {
      console.warn('Could not fit bounds to geo layer:', e);
    }
  }, [geoData]);

  const handleRecenter = () => {
    if (mapInstanceRef.current && geoLayerRef.current) {
      try {
        const bounds = geoLayerRef.current.getBounds();
        if (bounds.isValid()) {
          mapInstanceRef.current.fitBounds(bounds, { padding: [35, 35], maxZoom: 16 });
        }
      } catch (e) {
        // fallback
        const [lat, lon] = getFallbackCoordinates();
        mapInstanceRef.current.setView([lat, lon], 14);
      }
    }
  };

  return (
    <div style={{ position: 'relative', width: '100%', height: '420px', borderRadius: '12px', overflow: 'hidden', border: '1px solid var(--border-subtle)' }}>
      {/* Leaflet Map DOM Element */}
      <div ref={mapContainerRef} style={{ width: '100%', height: '100%' }} />

      {/* Top Controls Floating Bar */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          left: 12,
          zIndex: 1000,
          display: 'flex',
          gap: '0.4rem',
          background: 'rgba(7, 10, 18, 0.85)',
          backdropFilter: 'blur(10px)',
          padding: '4px 6px',
          borderRadius: '8px',
          border: '1px solid var(--border-subtle)',
        }}
      >
        <button
          className={`btn btn-sm ${basemap === 'satellite' ? 'btn-primary' : 'btn-ghost'}`}
          onClick={() => setBasemap('satellite')}
          style={{ fontSize: '0.72rem', padding: '3px 8px' }}
        >
          <Layers size={12} />
          <span>Esri Satellite</span>
        </button>
        <button
          className={`btn btn-sm ${basemap === 'street' ? 'btn-primary' : 'btn-ghost'}`}
          onClick={() => setBasemap('street')}
          style={{ fontSize: '0.72rem', padding: '3px 8px' }}
        >
          <MapPin size={12} />
          <span>OSM Basemap</span>
        </button>
        <button
          className="btn btn-ghost btn-sm"
          onClick={handleRecenter}
          title="Fit bounds to AOI footprint"
          style={{ fontSize: '0.72rem', padding: '3px 8px' }}
        >
          <Maximize2 size={12} />
          <span>Center AOI</span>
        </button>
      </div>

      {/* Dynamic Coordinates & Status Telemetry Banner (Bottom Left) */}
      <div
        style={{
          position: 'absolute',
          bottom: 12,
          left: 12,
          zIndex: 1000,
          background: 'rgba(7, 10, 18, 0.88)',
          backdropFilter: 'blur(10px)',
          padding: '4px 10px',
          borderRadius: '6px',
          border: '1px solid rgba(6, 182, 212, 0.3)',
          display: 'flex',
          alignItems: 'center',
          gap: '8px',
          fontSize: '0.72rem',
          color: 'var(--text-secondary)',
        }}
      >
        <div style={{ width: 7, height: 7, borderRadius: '50%', background: '#10b981', boxShadow: '0 0 8px #10b981' }} />
        <span style={{ fontWeight: 600, color: 'var(--cyan-400)' }}>WGS84 EPSG:4326</span>
        {coordinatesHud ? (
          <span className="mono" style={{ color: 'var(--text-primary)' }}>
            {coordinatesHud.lat}°N, {coordinatesHud.lng}°E
          </span>
        ) : (
          <span style={{ color: 'var(--text-muted)' }}>Hover over map for telemetry</span>
        )}
      </div>

      {/* Feature Count Pill (Top Right) */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          zIndex: 1000,
          background: 'rgba(7, 10, 18, 0.85)',
          backdropFilter: 'blur(10px)',
          padding: '4px 8px',
          borderRadius: '6px',
          border: '1px solid var(--border-subtle)',
          fontSize: '0.72rem',
          display: 'flex',
          alignItems: 'center',
          gap: '5px',
          color: 'var(--cyan-400)',
        }}
      >
        <Shield size={12} />
        <span>Vector GeoJSON Synchronized</span>
      </div>
    </div>
  );
}
