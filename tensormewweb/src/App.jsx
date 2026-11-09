import React, { useState } from "react";
import PixelBackground from "./components/PixelBackground";
import Navbar from "./components/Navbar";
import LandingTab from "./components/LandingTab";
import PokedexTab from "./components/PokedexTab";
import BlogTab from "./components/BlogTab";
import FoundersTab from "./components/FoundersTab";
import { soundEffects } from "./utils/soundEffects";
import { Analytics } from "@vercel/analytics/react";

export default function App() {
  const [activeTab, setActiveTab] = useState("landing");
  const [isMuted, setIsMuted] = useState(false);

  return (
    <div className="app-container">
      {/* 8-bit Interactive Pixel Canvas Background (Inspo from Image 1) */}
      <PixelBackground />

      {/* Navigation Header */}
      <Navbar
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        isMuted={isMuted}
        setIsMuted={setIsMuted}
      />

      {/* Dynamic Tab Body */}
      <main className="main-content">
        {activeTab === "landing" && <LandingTab setActiveTab={setActiveTab} />}
        {activeTab === "pokedex" && <PokedexTab />}
        {activeTab === "blog" && <BlogTab />}
        {activeTab === "founders" && <FoundersTab />}
      </main>

      {/* Retro Site Footer */}
      <footer className="site-footer">
        <div className="footer-inner">
          <div className="footer-left">
            <div className="footer-brand-wrap">
              <img src="/mew_logo.png" alt="TensorMew Logo" className="footer-logo-img" />
              <div className="footer-brand-text">
                <span className="footer-brand">TENSORMEW • POKÉMIND</span>
                <span className="footer-tagline">
                  First Johto Badge Autonomous RL Agent (Pokémon Crystal US 1.1)
                </span>
              </div>
            </div>
          </div>

          <div className="footer-right">
            <button
              className="footer-link"
              onClick={() => {
                soundEffects.playBlip();
                setActiveTab("landing");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            >
              Overview
            </button>
            <button
              className="footer-link"
              onClick={() => {
                soundEffects.playBlip();
                setActiveTab("pokedex");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            >
              Pokédex
            </button>
            <button
              className="footer-link"
              onClick={() => {
                soundEffects.playBlip();
                setActiveTab("blog");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            >
              Research &amp; Blog
            </button>
            <button
              className="footer-link"
              onClick={() => {
                soundEffects.playBlip();
                setActiveTab("founders");
                window.scrollTo({ top: 0, behavior: "smooth" });
              }}
            >
              Founders
            </button>
          </div>
        </div>
      </footer>

      {/* Vercel Web Analytics */}
      <Analytics />
    </div>
  );
}
