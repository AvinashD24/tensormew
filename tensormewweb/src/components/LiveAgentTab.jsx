import React, { useState, useEffect } from "react";
import { Play, Pause, SkipForward, SkipBack, RefreshCw, Terminal, Activity, Cpu, Shield, Zap, Sparkles } from "lucide-react";
import { CHECKPOINTS_STREAM } from "../data/agentTelemetryData";
import { soundEffects } from "../utils/soundEffects";

export default function LiveAgentTab() {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [isPlaying, setIsPlaying] = useState(true);
  const [playbackSpeed, setPlaybackSpeed] = useState(2000); // 2s per step
  const [activeDpad, setActiveDpad] = useState(null);
  const [activeButton, setActiveButton] = useState(null);

  const current = CHECKPOINTS_STREAM[currentIndex];

  // Auto-play through checkpoints
  useEffect(() => {
    let timer;
    if (isPlaying) {
      timer = setInterval(() => {
        setCurrentIndex((prev) => {
          const next = (prev + 1) % CHECKPOINTS_STREAM.length;
          // Trigger button flash
          const stepData = CHECKPOINTS_STREAM[next];
          setActiveDpad(stepData.dpad);
          setActiveButton(stepData.buttons);
          soundEffects.playBlip();
          return next;
        });
      }, playbackSpeed);
    }
    return () => clearInterval(timer);
  }, [isPlaying, playbackSpeed]);

  const handleSelectCheckpoint = (idx) => {
    soundEffects.playConfirm();
    setCurrentIndex(idx);
    setActiveDpad(CHECKPOINTS_STREAM[idx].dpad);
    setActiveButton(CHECKPOINTS_STREAM[idx].buttons);
  };

  const handleTogglePlay = () => {
    soundEffects.playBlip();
    setIsPlaying(!isPlaying);
  };

  const handleNext = () => {
    soundEffects.playBlip();
    const next = (currentIndex + 1) % CHECKPOINTS_STREAM.length;
    setCurrentIndex(next);
    setActiveDpad(CHECKPOINTS_STREAM[next].dpad);
    setActiveButton(CHECKPOINTS_STREAM[next].buttons);
  };

  const handlePrev = () => {
    soundEffects.playBlip();
    const prev = (currentIndex - 1 + CHECKPOINTS_STREAM.length) % CHECKPOINTS_STREAM.length;
    setCurrentIndex(prev);
    setActiveDpad(CHECKPOINTS_STREAM[prev].dpad);
    setActiveButton(CHECKPOINTS_STREAM[prev].buttons);
  };

  return (
    <div className="live-agent-view">
      {/* Tab Header Banner */}
      <div className="view-banner gsc-panel">
        <div className="view-banner-inner">
          <div className="banner-left">
            <span className="live-pill">
              <span className="rec-dot"></span> LIVE POLICY EVALUATION
            </span>
            <h2 className="view-heading">POKÉMIND DUAL-TIER AGENT INSTANCE</h2>
            <p className="view-sub">
              Streaming verified telemetry from the 13.2M parameter recurrent spatial policy
              evaluating on Pokémon Crystal (PyBoy WRAM symbol integration).
            </p>
          </div>
          <div className="banner-right hidden-mobile">
            <div className="badge-display-box">
              <span className="badge-icon">🎖</span>
              <div className="badge-info">
                <span className="badge-title">JOHTO BADGES</span>
                <span className="badge-count">
                  {current.badges} / 8 <span className="badge-name">(ZEPHYR)</span>
                </span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Main Dual Layout: GameBoy Simulator + Real-time Telemetry */}
      <div className="live-main-grid">
        {/* Left: Game Boy Color Console Frame */}
        <div className="gbc-console-wrapper">
          <div className="gbc-chassis">
            {/* Console Brand Top */}
            <div className="gbc-top-bar">
              <div className="power-led-group">
                <span className="power-dot"></span>
                <span className="power-text">POWER</span>
              </div>
              <div className="gbc-series-text">GAME BOY COLOR</div>
            </div>

            {/* GBC Screen Frame */}
            <div className="gbc-screen-frame">
              <div className="screen-inner-bezel">
                <div className="screen-status-bar">
                  <span>STEP: {current.step}</span>
                  <span className="pulse-text">● RUNNING</span>
                  <span>{current.mapName.split(" ")[0]}</span>
                </div>

                <div className="screen-viewport">
                  <img
                    src={current.screenImage}
                    alt={`Agent gameplay at ${current.name}`}
                    className="gbc-display-image"
                  />
                  {/* Overlay scanlines */}
                  <div className="screen-scanlines"></div>
                </div>

                {/* Crystal Battle UI Sub-Box */}
                <div className="gbc-screen-dialog-bar">
                  <div className="dialog-mini-text">
                    {current.reason.slice(0, 72)}...
                  </div>
                </div>
              </div>
            </div>

            {/* Nintendo Logo */}
            <div className="gbc-nintendo-logo">
              <span>Nintendo</span>
            </div>

            {/* Controls: D-Pad & A/B Buttons */}
            <div className="gbc-controls-row">
              {/* D-Pad */}
              <div className="dpad-container">
                <div className={`dpad-btn up ${activeDpad === "up" ? "pressed" : ""}`}></div>
                <div className={`dpad-btn right ${activeDpad === "right" ? "pressed" : ""}`}></div>
                <div className={`dpad-btn down ${activeDpad === "down" ? "pressed" : ""}`}></div>
                <div className={`dpad-btn left ${activeDpad === "left" ? "pressed" : ""}`}></div>
                <div className="dpad-center"></div>
              </div>

              {/* A & B Action Buttons */}
              <div className="action-buttons-wrap">
                <div className="ab-button-col">
                  <div className={`gbc-round-btn ${activeButton === "B" ? "pressed" : ""}`}>
                    <span>B</span>
                  </div>
                </div>
                <div className="ab-button-col offset-top">
                  <div className={`gbc-round-btn ${activeButton === "A" ? "pressed" : ""}`}>
                    <span>A</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Select & Start / Speaker Grille */}
            <div className="gbc-bottom-row">
              <div className="meta-buttons">
                <div className="pill-btn"><span>SELECT</span></div>
                <div className="pill-btn"><span>START</span></div>
              </div>
              <div className="speaker-grille">
                <div className="grille-line"></div>
                <div className="grille-line"></div>
                <div className="grille-line"></div>
                <div className="grille-line"></div>
              </div>
            </div>
          </div>

          {/* Playback Controller Bar */}
          <div className="gbc-playback-bar gsc-panel">
            <button className="control-btn" onClick={handlePrev} title="Previous Milestone">
              <SkipBack size={16} />
            </button>
            <button
              id="playback-toggle-btn"
              className={`control-btn play-btn ${isPlaying ? "playing" : ""}`}
              onClick={handleTogglePlay}
              title={isPlaying ? "Pause Stream" : "Resume Stream"}
            >
              {isPlaying ? <Pause size={18} /> : <Play size={18} fill="currentColor" />}
            </button>
            <button className="control-btn" onClick={handleNext} title="Next Milestone">
              <SkipForward size={16} />
            </button>

            <div className="speed-selector">
              <span className="speed-label">PACE:</span>
              {[
                { label: "1x", val: 3000 },
                { label: "2x", val: 2000 },
                { label: "4x", val: 1000 },
              ].map((s) => (
                <button
                  key={s.label}
                  className={`speed-pill ${playbackSpeed === s.val ? "active" : ""}`}
                  onClick={() => {
                    soundEffects.playBlip();
                    setPlaybackSpeed(s.val);
                  }}
                >
                  {s.label}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Right: Real-time Telemetry, RAM State, & Policy Metrics */}
        <div className="telemetry-dashboard">
          {/* Active Brain Tier Card */}
          <div className="gsc-card telemetry-card">
            <div className="telemetry-card-header">
              <div className="header-icon-title">
                <Cpu size={18} className="telemetry-icon" />
                <span className="card-headline">ACTIVE AGENT MODULE</span>
              </div>
              <span className={`actor-badge ${current.actorType}`}>
                {current.actor}
              </span>
            </div>

            <div className="telemetry-body">
              <div className="reasoning-box gsc-dialog-box">
                <div className="dialog-author-tag">
                  {current.actor.toUpperCase()} • STEP {current.step}
                </div>
                <p className="reasoning-text">"{current.reason}"</p>
                <div className="reasoning-meta-row">
                  <span className="tool-pill">Tool: {current.tool}</span>
                  <span className="trigger-pill">Trigger: {current.trigger}</span>
                </div>
              </div>
            </div>
          </div>

          {/* Party HUD & HP Bars */}
          <div className="gsc-card telemetry-card">
            <div className="telemetry-card-header">
              <div className="header-icon-title">
                <Shield size={18} className="telemetry-icon" />
                <span className="card-headline">WRAM PARTY STATUS (wPartyMons)</span>
              </div>
              <span className="sub-badge">MAP: {current.mapName}</span>
            </div>

            <div className="party-mons-list">
              {current.party.map((pkmn, i) => {
                const hpPercent = Math.round((pkmn.hp / pkmn.maxHp) * 100);
                return (
                  <div key={i} className="pkmn-party-row">
                    <div className="pkmn-row-top">
                      <span className="pkmn-name">{pkmn.name}</span>
                      <span className="pkmn-level">:L{pkmn.level}</span>
                    </div>

                    <div className="gsc-hp-container">
                      <span className="hp-label">HP:</span>
                      <div className="hp-bar-track">
                        <div
                          className={`hp-bar-fill ${
                            hpPercent > 50 ? "green" : hpPercent > 20 ? "yellow" : "red"
                          }`}
                          style={{ width: `${hpPercent}%` }}
                        ></div>
                      </div>
                      <span className="hp-fraction">
                        {pkmn.hp}/{pkmn.maxHp}
                      </span>
                    </div>

                    <div className="moveset-row">
                      {pkmn.moves.map((m, idx) => (
                        <span key={idx} className="move-tag">
                          {m}
                        </span>
                      ))}
                    </div>
                  </div>
                );
              })}
            </div>
          </div>

          {/* Policy Returns & Shannon Entropy Tracker */}
          <div className="gsc-card telemetry-card">
            <div className="telemetry-card-header">
              <div className="header-icon-title">
                <Activity size={18} className="telemetry-icon" />
                <span className="card-headline">RL REWARD & ENTROPY SIGNALS</span>
              </div>
            </div>

            <div className="metrics-bars-grid">
              <div className="metric-bar-item">
                <div className="metric-bar-header">
                  <span>Step Reward (r_t)</span>
                  <span className="metric-val">+{current.lastReward.toFixed(1)}</span>
                </div>
                <div className="stat-track">
                  <div
                    className="stat-fill cyan"
                    style={{ width: `${Math.min(100, current.lastReward * 4)}%` }}
                  ></div>
                </div>
              </div>

              <div className="metric-bar-item">
                <div className="metric-bar-header">
                  <span>Cumulative Return (G_t)</span>
                  <span className="metric-val">{current.cumulativeReturn.toFixed(1)}</span>
                </div>
                <div className="stat-track">
                  <div
                    className="stat-fill gold"
                    style={{ width: `${Math.min(100, (current.cumulativeReturn / 1000) * 100)}%` }}
                  ></div>
                </div>
              </div>

              <div className="metric-bar-item">
                <div className="metric-bar-header">
                  <span>Action Shannon Entropy H(π)</span>
                  <span className="metric-val">{current.entropy.toFixed(2)} / 1.61</span>
                </div>
                <div className="stat-track">
                  <div
                    className="stat-fill purple"
                    style={{ width: `${(current.entropy / 1.61) * 100}%` }}
                  ></div>
                </div>
                <span className="metric-note">Entropy threshold safe (&gt; 0.30 exploration bound)</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Checkpoint Timeline Quick Selector */}
      <div className="checkpoint-timeline-bar gsc-panel">
        <div className="timeline-title">STORY CHECKPOINT SELECTOR:</div>
        <div className="timeline-items-scroll">
          {CHECKPOINTS_STREAM.map((ckpt, idx) => (
            <button
              key={ckpt.checkpointId}
              className={`timeline-step-btn ${idx === currentIndex ? "active" : ""}`}
              onClick={() => handleSelectCheckpoint(idx)}
            >
              <span className="step-idx">{idx + 1}</span>
              <span className="step-label">{ckpt.name}</span>
            </button>
          ))}
        </div>
      </div>

      {/* How to Run Local PyBoy Command Banner */}
      <div className="local-runner-guide gsc-frame">
        <div className="guide-header">
          <Terminal size={18} />
          <h4>Run Live On Your Machine</h4>
        </div>
        <p className="guide-desc">
          To launch the interactive PyBoy window with real-time HUD metrics on your own machine:
        </p>
        <div className="terminal-code-block">
          <code>./run_live.sh --speed 3 --state route29_entry</code>
        </div>
      </div>
    </div>
  );
}
