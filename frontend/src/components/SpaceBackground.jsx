import React, { useEffect, useRef } from 'react';

export default function SpaceBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let animationFrameId;

    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      initStars();
    };
    window.addEventListener('resize', handleResize);

    // Mouse tracking for parallax
    let mouse = { x: width / 2, y: height / 2, targetX: width / 2, targetY: height / 2 };
    const handleMouseMove = (e) => {
      mouse.targetX = e.clientX;
      mouse.targetY = e.clientY;
    };
    window.addEventListener('mousemove', handleMouseMove);

    // Star generation
    const STAR_COUNT = 220;
    let stars = [];

    function initStars() {
      stars = [];
      for (let i = 0; i < STAR_COUNT; i++) {
        stars.push({
          x: Math.random() * width,
          y: Math.random() * height * 0.85, // mostly in the upper/space region
          size: Math.random() * 1.8 + 0.4,
          alpha: Math.random() * 0.8 + 0.2,
          speed: Math.random() * 0.015 + 0.005,
          color: Math.random() > 0.85 ? '#a5f3fc' : Math.random() > 0.7 ? '#ddd6fe' : '#ffffff',
          depth: Math.random() * 0.8 + 0.2
        });
      }
    }
    initStars();

    // Satellite animation parameters
    let satAngle = 0;

    const render = () => {
      // Smooth mouse easing
      mouse.x += (mouse.targetX - mouse.x) * 0.05;
      mouse.y += (mouse.targetY - mouse.y) * 0.05;

      const parallaxX = (mouse.x - width / 2) * 0.025;
      const parallaxY = (mouse.y - height / 2) * 0.025;

      ctx.clearRect(0, 0, width, height);

      // 1. Deep Space Obsidian Background
      const bgGrad = ctx.createLinearGradient(0, 0, 0, height);
      bgGrad.addColorStop(0, '#03050a');
      bgGrad.addColorStop(0.55, '#050a14');
      bgGrad.addColorStop(1, '#020307');
      ctx.fillStyle = bgGrad;
      ctx.fillRect(0, 0, width, height);

      // 2. Distant Cosmic Nebula Nebulae
      const nebulaGrad = ctx.createRadialGradient(
        width * 0.75 - parallaxX * 0.5,
        height * 0.3 - parallaxY * 0.5,
        20,
        width * 0.75,
        height * 0.3,
        width * 0.38
      );
      nebulaGrad.addColorStop(0, 'rgba(56, 189, 248, 0.06)');
      nebulaGrad.addColorStop(0.5, 'rgba(168, 85, 247, 0.04)');
      nebulaGrad.addColorStop(1, 'rgba(0, 0, 0, 0)');
      ctx.fillStyle = nebulaGrad;
      ctx.fillRect(0, 0, width, height);

      // 3. Render Stars with Parallax
      for (let star of stars) {
        star.alpha += Math.sin(Date.now() * star.speed) * 0.01;
        const currentAlpha = Math.max(0.15, Math.min(1, star.alpha));

        const drawX = star.x + parallaxX * star.depth;
        const drawY = star.y + parallaxY * star.depth;

        ctx.beginPath();
        ctx.arc(drawX, drawY, star.size, 0, Math.PI * 2);
        ctx.fillStyle = star.color;
        ctx.globalAlpha = currentAlpha;
        ctx.shadowBlur = star.size > 1.4 ? 4 : 0;
        ctx.shadowColor = star.color;
        ctx.fill();
      }
      ctx.globalAlpha = 1.0;
      ctx.shadowBlur = 0;

      // 4. Render Orbital Satellite Probe (Upper Right)
      satAngle += 0.012;
      const satBaseX = width * 0.82 - parallaxX * 0.3;
      const satBaseY = height * 0.18 + Math.sin(satAngle) * 8 - parallaxY * 0.3;

      ctx.save();
      ctx.translate(satBaseX, satBaseY);
      ctx.rotate(-0.35 + Math.sin(satAngle * 0.5) * 0.05);

      // Satellite Antenna Rays
      ctx.strokeStyle = 'rgba(255, 255, 255, 0.45)';
      ctx.lineWidth = 1.2;

      // Top antenna
      ctx.beginPath();
      ctx.moveTo(0, -14);
      ctx.lineTo(-65, -70);
      ctx.stroke();

      // Side antenna 1
      ctx.beginPath();
      ctx.moveTo(8, -10);
      ctx.lineTo(55, -80);
      ctx.stroke();

      // Bottom antenna
      ctx.beginPath();
      ctx.moveTo(-10, 10);
      ctx.lineTo(-75, 45);
      ctx.stroke();

      // Satellite Sphere Body
      const satGrad = ctx.createRadialGradient(-4, -4, 2, 0, 0, 15);
      satGrad.addColorStop(0, '#ffffff');
      satGrad.addColorStop(0.35, '#94a3b8');
      satGrad.addColorStop(0.8, '#334155');
      satGrad.addColorStop(1, '#0f172a');
      ctx.fillStyle = satGrad;
      ctx.beginPath();
      ctx.arc(0, 0, 14, 0, Math.PI * 2);
      ctx.shadowBlur = 12;
      ctx.shadowColor = 'rgba(6, 182, 212, 0.6)';
      ctx.fill();

      // Satellite Solar Pulse Blinker
      const blink = (Math.sin(satAngle * 3) + 1) / 2;
      ctx.beginPath();
      ctx.arc(4, -4, 2.5, 0, Math.PI * 2);
      ctx.fillStyle = `rgba(34, 211, 238, ${0.4 + blink * 0.6})`;
      ctx.shadowBlur = 8;
      ctx.shadowColor = '#22d3ee';
      ctx.fill();

      ctx.restore();

      // 5. Curved Earth Horizon Limb & Atmospheric Glow (Bottom of screen)
      const earthRadius = Math.max(width * 1.15, 1200);
      const earthCenterX = width * 0.5;
      const earthCenterY = height + earthRadius - Math.min(height * 0.32, 260);

      // Atmosphere Outer Glow
      const atmoGlow = ctx.createRadialGradient(
        earthCenterX,
        earthCenterY,
        earthRadius - 10,
        earthCenterX,
        earthCenterY,
        earthRadius + 85
      );
      atmoGlow.addColorStop(0, 'rgba(56, 189, 248, 0.45)');
      atmoGlow.addColorStop(0.3, 'rgba(37, 99, 235, 0.28)');
      atmoGlow.addColorStop(0.7, 'rgba(6, 182, 212, 0.10)');
      atmoGlow.addColorStop(1, 'rgba(0, 0, 0, 0)');

      ctx.fillStyle = atmoGlow;
      ctx.beginPath();
      ctx.arc(earthCenterX, earthCenterY, earthRadius + 85, 0, Math.PI * 2);
      ctx.fill();

      // Earth Planet Body
      const planetGrad = ctx.createRadialGradient(
        earthCenterX,
        earthCenterY - earthRadius * 0.7,
        earthRadius * 0.3,
        earthCenterX,
        earthCenterY,
        earthRadius
      );
      planetGrad.addColorStop(0, '#1e3a8a');
      planetGrad.addColorStop(0.5, '#0f264d');
      planetGrad.addColorStop(0.85, '#07162c');
      planetGrad.addColorStop(0.98, '#030a17');
      planetGrad.addColorStop(1, '#000000');

      ctx.fillStyle = planetGrad;
      ctx.beginPath();
      ctx.arc(earthCenterX, earthCenterY, earthRadius, 0, Math.PI * 2);
      ctx.fill();

      // Blue Atmospheric Horizon Rim Line
      ctx.strokeStyle = 'rgba(147, 197, 253, 0.85)';
      ctx.lineWidth = 2.5;
      ctx.shadowBlur = 18;
      ctx.shadowColor = '#38bdf8';
      ctx.beginPath();
      ctx.arc(earthCenterX, earthCenterY, earthRadius, Math.PI * 1.15, Math.PI * 1.85);
      ctx.stroke();

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener('resize', handleResize);
      window.removeEventListener('mousemove', handleMouseMove);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      style={{
        position: 'fixed',
        top: 0,
        left: 0,
        width: '100%',
        height: '100%',
        pointerEvents: 'none',
        zIndex: 0,
      }}
    />
  );
}
