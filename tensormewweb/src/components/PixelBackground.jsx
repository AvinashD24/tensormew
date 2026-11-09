import React, { useEffect, useRef } from "react";

export default function PixelBackground() {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");

    let animationFrameId;
    let width = (canvas.width = window.innerWidth);
    let height = (canvas.height = window.innerHeight);

    // Pixel block size (8-bit grid inspired by reference image)
    const BLOCK_SIZE = 24;
    let cols = Math.ceil(width / BLOCK_SIZE);
    let rows = Math.ceil(height / BLOCK_SIZE);

    // Grid states: heat level [0.0 to 1.0] and random seed character/value
    let grid = new Float32Array(cols * rows);
    let numbers = new Uint8Array(cols * rows);
    for (let i = 0; i < numbers.length; i++) {
      numbers[i] = Math.random() < 0.2 ? Math.floor(Math.random() * 3) + 1 : 0;
    }

    const mouse = { x: -1000, y: -1000, prevX: -1000, prevY: -1000, isHovered: false };

    const handleResize = () => {
      width = canvas.width = window.innerWidth;
      height = canvas.height = window.innerHeight;
      cols = Math.ceil(width / BLOCK_SIZE);
      rows = Math.ceil(height / BLOCK_SIZE);
      grid = new Float32Array(cols * rows);
      numbers = new Uint8Array(cols * rows);
      for (let i = 0; i < numbers.length; i++) {
        numbers[i] = Math.random() < 0.2 ? Math.floor(Math.random() * 3) + 1 : 0;
      }
    };

    const handleMouseMove = (e) => {
      mouse.x = e.clientX;
      mouse.y = e.clientY;
      mouse.isHovered = true;

      // Ignite blocks around the cursor
      const centerCol = Math.floor(mouse.x / BLOCK_SIZE);
      const centerRow = Math.floor(mouse.y / BLOCK_SIZE);
      const radius = 3;

      for (let r = -radius; r <= radius; r++) {
        for (let c = -radius; c <= radius; c++) {
          const col = centerCol + c;
          const row = centerRow + r;
          if (col >= 0 && col < cols && row >= 0 && row < rows) {
            const dist = Math.sqrt(c * c + r * r);
            if (dist <= radius) {
              const idx = row * cols + col;
              const intensity = (1 - dist / radius) * 0.95;
              grid[idx] = Math.max(grid[idx], intensity);
            }
          }
        }
      }
    };

    const handleMouseLeave = () => {
      mouse.isHovered = false;
      mouse.x = -1000;
      mouse.y = -1000;
    };

    window.addEventListener("resize", handleResize);
    window.addEventListener("mousemove", handleMouseMove);
    document.addEventListener("mouseleave", handleMouseLeave);

    const render = () => {
      ctx.clearRect(0, 0, width, height);

      // Subtle base grid lines on off-white
      ctx.strokeStyle = "rgba(0, 0, 0, 0.05)";
      ctx.lineWidth = 1;

      // Draw lit / decaying pixels in dark squares on off-white (Image 1 top half aesthetic)
      for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
          const idx = row * cols + col;
          let heat = grid[idx];

          if (heat > 0.01) {
            // Decay heat smoothly
            grid[idx] *= 0.92;

            const px = col * BLOCK_SIZE;
            const py = row * BLOCK_SIZE;

            // Deep black/charcoal square on off-white
            const alpha = heat * 0.92;
            ctx.fillStyle = `rgba(15, 15, 20, ${alpha})`;
            ctx.fillRect(px + 1, py + 1, BLOCK_SIZE - 2, BLOCK_SIZE - 2);

            // 8-bit retro numeral inside the block in crisp white (like reference image 1)
            const num = numbers[idx];
            if (num > 0 && heat > 0.15) {
              ctx.fillStyle = `rgba(255, 255, 255, ${Math.min(1, heat * 1.2)})`;
              ctx.font = "bold 9px 'Silkscreen', monospace";
              ctx.textAlign = "center";
              ctx.textBaseline = "middle";
              ctx.fillText(num.toString(), px + BLOCK_SIZE / 2, py + BLOCK_SIZE / 2);
            }
          }
        }
      }

      animationFrameId = requestAnimationFrame(render);
    };

    render();

    return () => {
      window.removeEventListener("resize", handleResize);
      window.removeEventListener("mousemove", handleMouseMove);
      document.removeEventListener("mouseleave", handleMouseLeave);
      cancelAnimationFrame(animationFrameId);
    };
  }, []);

  return (
    <canvas
      ref={canvasRef}
      className="pixel-bg-canvas"
      aria-hidden="true"
    />
  );
}
