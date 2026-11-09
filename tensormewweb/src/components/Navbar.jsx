import React, { useState } from "react";
import { Volume2, VolumeX, Menu, X } from "lucide-react";
import { soundEffects } from "../utils/soundEffects";

export default function Navbar({ activeTab, setActiveTab, isMuted, setIsMuted }) {
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  const tabs = [
    { id: "landing", label: "Overview", num: "01" },
    { id: "pokedex", label: "Pokédex", num: "02" },
    { id: "blog", label: "Research & Blog", num: "03" },
    { id: "founders", label: "Founders", num: "04" },
  ];

  const handleTabClick = (tabId) => {
    soundEffects.playBlip();
    setActiveTab(tabId);
    setMobileMenuOpen(false);
  };

  const handleMuteToggle = () => {
    const muted = soundEffects.toggleMute();
    setIsMuted(muted);
    if (!muted) {
      soundEffects.playConfirm();
    }
  };

  const toggleMobileMenu = () => {
    soundEffects.playBlip();
    setMobileMenuOpen((prev) => !prev);
  };

  return (
    <header className="site-header">
      <div className="header-inner">
        {/* Logo and Brand */}
        <div
          className="brand-group"
          onClick={() => {
            soundEffects.playBlip();
            setActiveTab("landing");
            setMobileMenuOpen(false);
          }}
        >
          <img src="/mew_logo.png" alt="TensorMew Logo" className="brand-logo-img" />
          <div className="brand-text">
            <div className="brand-title">
              <span className="crystal-gradient-text">TENSOR</span>
              <span className="brand-accent">MEW</span>
            </div>
            <div className="brand-tagline">POKÉMON CRYSTAL • RL AGENT</div>
          </div>
        </div>

        {/* Desktop Navigation Tabs */}
        <nav className="header-nav desktop-nav">
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

        {/* Right Header Actions: Sound + Mobile Hamburger */}
        <div className="header-actions-group">
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

          {/* Mobile Menu Hamburger Button */}
          <button
            id="mobile-menu-toggle-btn"
            className="retro-mobile-toggle"
            onClick={toggleMobileMenu}
            aria-label={mobileMenuOpen ? "Close menu" : "Open menu"}
            aria-expanded={mobileMenuOpen}
          >
            {mobileMenuOpen ? <X size={20} /> : <Menu size={20} />}
            <span className="mobile-toggle-label">{mobileMenuOpen ? "CLOSE" : "MENU"}</span>
          </button>
        </div>
      </div>

      {/* Mobile Drawer Dropdown */}
      {mobileMenuOpen && (
        <div className="mobile-menu-overlay" onClick={() => setMobileMenuOpen(false)}>
          <div className="mobile-menu-dropdown gsc-frame" onClick={(e) => e.stopPropagation()}>
            <div className="mobile-menu-header">
              <span className="mobile-menu-title">SELECT MODE</span>
              <span className="mobile-menu-badge">GBC RAM LINK</span>
            </div>
            <nav className="mobile-nav-list">
              {tabs.map((tab) => {
                const isActive = activeTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    id={`mobile-nav-${tab.id}`}
                    className={`mobile-nav-item ${isActive ? "active" : ""}`}
                    onClick={() => handleTabClick(tab.id)}
                  >
                    <div className="mobile-nav-item-left">
                      <span className="mobile-tab-num">{tab.num}</span>
                      <span className="mobile-tab-label">{tab.label}</span>
                    </div>
                    {isActive ? (
                      <span className="mobile-active-cursor">▶ ACTIVE</span>
                    ) : (
                      <span className="mobile-cursor-idle">▷</span>
                    )}
                  </button>
                );
              })}
            </nav>
            <div className="mobile-menu-footer">
              <button className="mobile-menu-sound-btn" onClick={handleMuteToggle}>
                {isMuted ? <VolumeX size={15} /> : <Volume2 size={15} />}
                <span>{isMuted ? "Audio: Muted" : "Audio: 8-Bit FX Active"}</span>
              </button>
            </div>
          </div>
        </div>
      )}
    </header>
  );
}
