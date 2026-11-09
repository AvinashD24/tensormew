import React, { useState } from "react";
import { Eye, X } from "lucide-react";
import { SCREENSHOTS_DATA } from "../data/screenshotsData";
import { soundEffects } from "../utils/soundEffects";

export default function LandingTab({ setActiveTab }) {
  const [selectedScreenshot, setSelectedScreenshot] = useState(null);

  // Keep only the 3 specific milestones requested:
  // 1) Rival battle (id: 9)
  // 2) Entering violet gym (id: 17)
  // 3) Zephyr badge officially conquered (id: 21)
  const milestoneScreenshots = SCREENSHOTS_DATA.filter((s) => [9, 17, 21].includes(s.id));

  const openScreenshotModal = (item) => {
    soundEffects.playConfirm();
    setSelectedScreenshot(item);
  };

  const closeScreenshotModal = () => {
    soundEffects.playBlip();
    setSelectedScreenshot(null);
  };

  return (
    <div className="landing-view">
      {/* Hero Section */}
      <section className="hero-section">
        {/* Landing Page Banner: TENSORMEW */}
        <div className="hero-brand-banner">
          <h1 className="hero-brand-title">TENSORMEW</h1>
        </div>

        <h2 className="hero-title">
          <span className="glitch-text" data-text="THE WORLD'S FIRST RL AGENT">
            THE WORLD'S FIRST RL AGENT
          </span>
          <br />
          <span className="hero-sub-highlight">TO PLAY POKÉMON CRYSTAL</span>
        </h2>

        <div className="gsc-dialog-box hero-synopsis-box">
          <div className="dialog-corner-bl"></div>
          <div className="dialog-corner-br"></div>
          <p className="hero-synopsis-text">
            <strong>TensorMew</strong> pairs a 13.2M parameter recurrent spatial vision policy with
            banked Game Boy Color memory introspection to conquer deep overworld frontiers, one-way ledges,
            curriculum battle grinding, and Johto Gym Leader Falkner—achieving genuine autonomous progression
            with zero pre-recorded town routes.
          </p>
          <div className="dialog-prompt-arrow">▼</div>
        </div>
      </section>

      {/* Featured Video / GIF Showcase */}
      <section className="showcase-banner-section">
        <div className="section-header-row">
          <div className="section-title-wrap">
            <span className="section-num">01.1</span>
            <h2 className="section-heading">FALKNER GYM SHOWCASE RUN</h2>
          </div>
          <span className="record-badge">
            <span className="rec-dot"></span> RECORDED RUN (PACED 2X)
          </span>
        </div>

        <div className="showcase-card gsc-frame">
          <div className="showcase-media-wrap">
            <img
              src="/recordings/falkner_showcase.gif"
              alt="TensorMew conquering Falkner in Pokémon Crystal"
              className="showcase-gif"
              loading="lazy"
            />
          </div>
          <div className="showcase-info-panel">
            <div className="gsc-badge-tag">HIGHLIGHT REEL</div>
            <h3 className="showcase-info-title">Violet City to Zephyr Badge Swept</h3>
            <p className="showcase-info-desc">
              TensorMew enters Violet City, navigates through the gate, completes party recovery at the Pokémon Center,
              enters the Violet Gym catwalk, and outplays Falkner's Mud-Slap Pidgeotto using Cyndaquil/Quilava's STAB Ember.
            </p>
            <div className="showcase-specs-list">
              <div className="spec-item">
                <span className="spec-k">Lead Pokémon:</span>
                <span className="spec-v">Quilava Lv.16</span>
              </div>
              <div className="spec-item">
                <span className="spec-k">Key Moves:</span>
                <span className="spec-v">Ember, SmokeScreen, Tackle</span>
              </div>
              <div className="spec-item">
                <span className="spec-k">RAM Flags:</span>
                <span className="spec-v">wJohtoBadges = 0x01</span>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Gameplay Milestones & Screenshot Highlights (Only Rival, Gym, Badge) */}
      <section className="gallery-section">
        <div className="section-header-row">
          <div className="section-title-wrap">
            <span className="section-num">01.2</span>
            <h2 className="section-heading">GAMEPLAY MILESTONES &amp; SCREENSHOT HIGHLIGHTS</h2>
          </div>
        </div>

        <div className="screenshots-grid">
          {milestoneScreenshots.map((item) => (
            <div
              key={item.id}
              className="screenshot-card gsc-card"
              onClick={() => openScreenshotModal(item)}
            >
              <div className="card-thumb-wrap">
                <img
                  src={`/screenshots/${item.file}`}
                  alt={item.title}
                  className="card-thumb-img"
                  loading="lazy"
                />
                <div className="card-hover-overlay">
                  <span className="inspect-pill">
                    <Eye size={14} /> INSPECT
                  </span>
                </div>
                <div className="card-step-badge">Step {item.step}</div>
              </div>
              <div className="card-body">
                <div className="card-milestone-pill">{item.badge}</div>
                <h4 className="card-title">{item.title}</h4>
                <div className="card-lead-badge">{item.lead}</div>
                <p className="card-desc-snippet">{item.desc}</p>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* Modal for Full Screenshot Inspection */}
      {selectedScreenshot && (
        <div className="retro-modal-backdrop" onClick={closeScreenshotModal}>
          <div
            className="retro-modal-content gsc-dialog-frame"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div className="modal-title-wrap">
                <span className="modal-badge">{selectedScreenshot.badge}</span>
                <h3 className="modal-title">{selectedScreenshot.title}</h3>
              </div>
              <button
                className="modal-close-btn"
                onClick={closeScreenshotModal}
                aria-label="Close modal"
              >
                <X size={20} />
              </button>
            </div>

            <div className="modal-body">
              <div className="modal-media-wrap">
                <img
                  src={`/screenshots/${selectedScreenshot.file}`}
                  alt={selectedScreenshot.title}
                  className="modal-pixel-img"
                />
              </div>

              <div className="modal-details">
                <div className="gsc-meta-table">
                  <div className="meta-row">
                    <span className="meta-k">Step Number:</span>
                    <span className="meta-v">Step {selectedScreenshot.step}</span>
                  </div>
                  <div className="meta-row">
                    <span className="meta-k">Active Lead:</span>
                    <span className="meta-v">{selectedScreenshot.lead}</span>
                  </div>
                  <div className="meta-row">
                    <span className="meta-k">Map Location:</span>
                    <span className="meta-v">{selectedScreenshot.location}</span>
                  </div>
                </div>

                <div className="gsc-dialog-box modal-dialog-box">
                  <div className="dialog-prompt-header">WRAM TELEMETRY REPORT</div>
                  <p className="modal-desc-full">{selectedScreenshot.desc}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
