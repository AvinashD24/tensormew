import React from "react";

/**
 * 8-Bit XP Bar Component
 * Inspired by 8bitcn (https://www.8bitcn.com/docs/components/xp-bar)
 * Provides authentic retro segmented blocks, pixel borders, and level up state.
 */
export default function XpBar({
  value = 0,
  max = 100,
  segments = 20,
  variant = "retro", // "retro" (segmented pixel blocks) | "default" (continuous)
  progressColor = "black", // "black", "gold", "green", "red", "blue"
  height = 14,
  className = "",
  showLevelUp = false,
  levelUpMessage = "LEVEL UP!",
}) {
  const safeMax = max > 0 ? max : 100;
  const percentage = Math.min(100, Math.max(0, (value / safeMax) * 100));
  const filledSquares = Math.round((percentage / 100) * segments);
  const isLevelUp = percentage >= 100;

  return (
    <div className={`bit-xp-root ${className}`}>
      <div className="bit-xp-outer-frame">
        <div
          className="bit-xp-track"
          style={{ height: `${height}px` }}
          role="progressbar"
          aria-valuenow={value}
          aria-valuemin={0}
          aria-valuemax={safeMax}
        >
          {variant === "retro" ? (
            <div className="bit-xp-segments">
              {Array.from({ length: segments }).map((_, i) => (
                <div
                  key={i}
                  className={`bit-xp-square ${
                    i < filledSquares ? `filled ${progressColor}` : "empty"
                  }`}
                />
              ))}
            </div>
          ) : (
            <div
              className={`bit-xp-continuous-fill ${progressColor}`}
              style={{ width: `${percentage}%` }}
            />
          )}

          {showLevelUp && isLevelUp && (
            <div className="bit-xp-levelup-badge">{levelUpMessage}</div>
          )}
        </div>
      </div>
    </div>
  );
}
