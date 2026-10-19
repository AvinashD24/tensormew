import React from "react";
import { Volume2, VolumeX, Terminal, ShieldCheck, Zap } from "lucide-react";
import { soundEffects } from "../utils/soundEffects";

export default function Navbar({ activeTab, setActiveTab, isMuted, setIsMuted }) {
  const tabs = [
    { id: "landing", label: "Overview", num: "01" },
    { id: "pokedex", label: "Pokédex", num: "02" },
    { id: "blog", label: "Research & Blog", num: "03" },
    { id: "founders", label: "Founders", num: "04" },
  ];

  const handleTabClick = (tabId) => {
    soundEffects.playBlip();
    setActiveTab(tabId);
  };

  const handleMuteToggle = () => {
    const muted = soundEffects.toggleMute();
    setIsMuted(muted);
    if (!muted) {
      soundEffects.playConfirm();
    }
  };

  return (
    <header className="site-header">
      <div className="header-inner">
        {/* Logo and Brand */}
        <div className="brand-group" onClick={() => handleTabClick("landing")}>
          <img src="/mew_logo.png" alt="TensorMew Logo" className="brand-logo-img" />
          <div className="brand-text">
            <div className="brand-title">
              <span className="crystal-gradient-text">TENSOR</span>
              <span className="brand-accent">MEW</span>
            </div>
            <div className="brand-tagline">POKÉMON CRYSTAL • RL AGENT</div>
          </div>
        </div>

        {/* Navigation Tabs */}
        <nav className="header-nav">
          {tabs.map((tab) => {
            const isActive = activeTab === tab.id;
            return (
              <button
                key={tab.id}
                id={`nav-tab-${tab.id}`}
                className={`retro-nav-button ${isActive ? "active" : ""}`}
                onClick={() => handleTabClick(tab.id)}
              >
                <span className="tab-num">{tab.num}</span>
                <span className="tab-label">{tab.label}</span>
                {isActive && <span className="tab-indicator">▼</span>}
              </button>
            );
          })}
        </nav>

        {/* Sound Toggle */}
        <div className="audio-control-wrap">
          <button
            id="sound-toggle-btn"
            className="retro-sound-btn"
            onClick={handleMuteToggle}
            title={isMuted ? "Unmute 8-Bit Audio" : "Mute Audio"}
            aria-label={isMuted ? "Unmute audio" : "Mute audio"}
          >
            {isMuted ? <VolumeX size={16} /> : <Volume2 size={16} />}
            <span className="sound-text hidden-mobile">{isMuted ? "MUTE" : "8-BIT FX"}</span>
          </button>
        </div>
      </div>
    </header>
  );
}
