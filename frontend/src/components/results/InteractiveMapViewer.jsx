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

  // Available image slot candidate for telemetry and georeferencing
  const availableSlotsList = Object.values(slotImages || {}).filter(Boolean);
  const firstSlot = availableSlotsList.length > 0 ? availableSlotsList[0] : null;

  // Real georeference only: GeoTIFF tag, non-null CRS, real bounds, or a server GeoJSON layer
  const envBounds = firstSlot?.bounds || firstSlot?.geo_bbox;
  const hasRealBounds = Array.isArray(envBounds) && envBounds.length === 4;
  const isGeoreferenced = Boolean(
    firstSlot?.tags?.geotiff ||
    (firstSlot?.crs != null && firstSlot?.crs !== '') ||
    hasRealBounds ||
    result?.geojson_url
  );

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

  // Determine initial center coordinates from real georeference only (never fabricated)
  const getInitialCoordinates = () => {
    if (hasRealBounds) {
      const [minLon, minLat, maxLon, maxLat] = envBounds;
      return [(minLat + maxLat) / 2, (minLon + maxLon) / 2];
    }
    // Real coordinates parsed from backend evidence (if any)
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
    // No real georeference available
    return null;
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
      } else if (hasRealBounds) {
        const [minLon, minLat, maxLon, maxLat] = envBounds;
        const coords = [[
          [minLon, minLat],
          [maxLon, minLat],
          [maxLon, maxLat],
          [minLon, maxLat],
          [minLon, minLat],
        ]];
        const footprintGeoJson = {
          type: 'FeatureCollection',
          features: [
            {
              type: 'Feature',
              properties: {
                name: 'Image Bounds Footprint',
              },
              geometry: {
                type: 'Polygon',
                coordinates: coords,
              },
            },
          ],
        };
        setGeoData(footprintGeoJson);
      } else {
        // Not georeferenced: never fabricate an AOI
        setGeoData(null);
      }
    }
    loadGeoJson();
    return () => { isCancelled = true; };
  }, [result?.geojson_url]);

  // Initialize Leaflet Map (re-runs when georeference availability changes)
  useEffect(() => {
    const container = mapContainerRef.current;
    if (!container || !isGeoreferenced) return undefined;

    if (!mapInstanceRef.current) {
      if (container._leaflet_id) {
        delete container._leaflet_id;
      }

      const initialCenter = getInitialCoordinates();
      const map = L.map(container, {
        center: initialCenter || [0, 0],
        zoom: initialCenter ? 13 : 2,
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
        if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize();
      }, 100);
      setTimeout(() => {
        if (mapInstanceRef.current) mapInstanceRef.current.invalidateSize();
      }, 350);
    }

    return () => {
      if (mapInstanceRef.current) {
        mapInstanceRef.current.remove();
        mapInstanceRef.current = null;
      }
      tileLayerRef.current = null;
      if (container && container._leaflet_id) {
        delete container._leaflet_id;
      }
    };
  }, [isGeoreferenced]);

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

  // Helper function to color features by class (water, forest, vegetation, built-up)
  const getFeatureColor = (feature) => {
    const props = feature?.properties || {};
    if (props.color) return props.color;
    const cat = (props.category || props.type || props.label || '').toLowerCase();
    if (cat.includes('water')) return '#0284c7';      // Deep Blue
    if (cat.includes('forest')) return '#15803d';     // Forest Green
    if (cat.includes('crop') || cat.includes('veg') || cat.includes('field')) return '#84cc16'; // Lime / Vegetative
    if (cat.includes('built') || cat.includes('urban') || cat.includes('struct')) return '#f59e0b'; // Amber / Built-up
    return '#06b6d4'; // Default Cyan
  };

  // Render / Update GeoJSON Layer
  useEffect(() => {
    if (!mapInstanceRef.current || !geoData) return;

    if (geoLayerRef.current) {
      mapInstanceRef.current.removeLayer(geoLayerRef.current);
    }

    const layer = L.geoJSON(geoData, {
      style: (feature) => {
        const featureColor = getFeatureColor(feature);
        return {
          color: featureColor,
          weight: 3,
          opacity: 0.95,
          fillColor: featureColor,
          fillOpacity: 0.28,
          dashArray: '6, 3',
        };
      },
      onEachFeature: (feature, featureLayer) => {
        const props = feature.properties || {};
        const title = props.name || props.label || props.category || 'Target Area';
        const color = getFeatureColor(feature);
        featureLayer.bindPopup(`
          <div style="font-family: var(--font-sans); color: #0f172a; padding: 6px; min-width: 170px;">
            <div style="display: flex; align-items: center; gap: 6px; margin-bottom: 6px;">
              <span style="width: 10px; height: 10px; border-radius: 2px; background: ${color}; display: inline-block;"></span>
              <div style="font-weight: 700; font-size: 13px; color: #0f172a;">${title}</div>
            </div>
            <div style="font-size: 11px; color: #475569; display: flex; flex-direction: column; gap: 3px;">
              ${props.category ? `<div><strong>Classification:</strong> <span style="color: ${color}; font-weight: 600;">${props.category.toUpperCase()}</span></div>` : ''}
              ${typeof props.confidence === 'number' ? `<div><strong>Confidence:</strong> ${(props.confidence * 100).toFixed(1)}%</div>` : ''}
              ${props.description ? `<div style="margin-top: 4px; font-size: 10px; color: #64748b; line-height: 1.3;">${props.description}</div>` : ''}
              ${typeof props.area_km2 === 'number' ? `<div><strong>Area:</strong> ${props.area_km2.toFixed(3)} km²</div>` : ''}
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
        const fallbackCenter = getInitialCoordinates();
        if (fallbackCenter) {
          mapInstanceRef.current.setView(fallbackCenter, 14);
        }
      }
    }
  };

  // No real georeference: never render a fabricated map, coordinates, or default AOI
  if (!isGeoreferenced) {
    return (
      <div
        className="glass-panel"
        style={{
          width: '100%',
          height: '420px',
          borderRadius: '12px',
          border: '1px solid var(--border-subtle)',
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          gap: '0.4rem',
          color: 'var(--text-muted)',
          textAlign: 'center',
          padding: '1rem',
        }}
      >
        <MapPin size={28} style={{ opacity: 0.5 }} />
        <span style={{ fontWeight: 600, color: 'var(--text-secondary)' }}>Not georeferenced — map unavailable</span>
        <span style={{ fontSize: '0.75rem', maxWidth: '340px' }}>
          This image has no real bounds or CRS, so no map coordinates or AOI footprint are shown.
        </span>
      </div>
    );
  }

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
          background: 'rgba(7, 10, 18, 0.90)',
          backdropFilter: 'blur(10px)',
          padding: '5px 12px',
          borderRadius: '6px',
          border: '1px solid rgba(6, 182, 212, 0.3)',
          display: 'flex',
          alignItems: 'center',
          gap: '10px',
          fontSize: '0.72rem',
          color: 'var(--text-secondary)',
          maxWidth: '85%',
        }}
      >
        <div 
          style={{ 
            width: 7, 
            height: 7, 
            borderRadius: '50%', 
            background: '#10b981', 
            boxShadow: '0 0 8px #10b981',
            flexShrink: 0
          }} 
        />
        <span style={{ fontWeight: 600, color: 'var(--cyan-400)' }}>WGS84 EPSG:4326</span>
        {coordinatesHud ? (
          <span className="mono" style={{ color: 'var(--text-primary)' }}>
            {coordinatesHud.lat}°N, {coordinatesHud.lng}°E
          </span>
        ) : (
          <span style={{ color: 'var(--text-muted)' }}>Hover for cursor telemetry</span>
        )}
        <span 
          style={{ 
            fontSize: '0.68rem', 
            color: '#10b981',
            borderLeft: '1px solid var(--border-subtle)',
            paddingLeft: '8px',
            whiteSpace: 'nowrap'
          }}
        >
          🛰️ Georeferenced (real bounds/CRS)
        </span>
      </div>

      {/* Feature Count Pill (Top Right) */}
      <div
        style={{
          position: 'absolute',
          top: 12,
          right: 12,
          zIndex: 1000,
          background: 'rgba(7, 10, 18, 0.88)',
          backdropFilter: 'blur(10px)',
          padding: '4px 10px',
          borderRadius: '6px',
          border: '1px solid var(--border-subtle)',
          fontSize: '0.72rem',
          display: 'flex',
          alignItems: 'center',
          gap: '6px',
          color: result?.geojson_url ? '#10b981' : 'var(--cyan-400)',
        }}
      >
        <Shield size={12} />
        <span>
          {result?.geojson_url 
            ? `GeoJSON Vector Layer (${geoData?.features?.length || 0} zones)` 
            : 'AOI footprint from real image bounds'}
        </span>
      </div>

      {/* Classification Multi-Class Legend */}
      <div
        style={{
          position: 'absolute',
          top: 48,
          right: 12,
          zIndex: 1000,
          background: 'rgba(7, 10, 18, 0.90)',
          backdropFilter: 'blur(10px)',
          padding: '6px 10px',
          borderRadius: '6px',
          border: '1px solid var(--border-subtle)',
          display: 'flex',
          flexDirection: 'column',
          gap: '4px',
          fontSize: '0.68rem',
        }}
      >
        <div style={{ fontWeight: 700, color: 'var(--text-muted)', fontSize: '0.62rem', textTransform: 'uppercase', letterSpacing: '0.5px', marginBottom: 2 }}>
          Class Legend
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 9, height: 9, borderRadius: 2, background: '#0284c7', border: '1px solid #38bdf8', display: 'inline-block' }} />
          <span style={{ color: '#bae6fd' }}>Water Body</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 9, height: 9, borderRadius: 2, background: '#15803d', border: '1px solid #22c55e', display: 'inline-block' }} />
          <span style={{ color: '#bbf7d0' }}>Forest Canopy</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 9, height: 9, borderRadius: 2, background: '#84cc16', border: '1px solid #a3e635', display: 'inline-block' }} />
          <span style={{ color: '#d9f99d' }}>Vegetation / Crops</span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
          <span style={{ width: 9, height: 9, borderRadius: 2, background: '#f59e0b', border: '1px solid #fbbf24', display: 'inline-block' }} />
          <span style={{ color: '#fde68a' }}>Built-Up / Urban</span>
        </div>
      </div>
    </div>
  );
}
