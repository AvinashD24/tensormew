import React, { useState } from "react";
import { Search, CheckCircle2, HelpCircle, X } from "lucide-react";
import { POKEDEX_DATA } from "../data/pokedexData";
import { soundEffects } from "../utils/soundEffects";
import XpBar from "./ui/XpBar";

export default function PokedexTab() {
  const [pokemonList] = useState(POKEDEX_DATA);
  const [filterMode, setFilterMode] = useState("all"); // 'all', 'discovered', 'undiscovered'
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedMon, setSelectedMon] = useState(null);

  const handleMonClick = (mon) => {
    if (!mon.discovered) {
      soundEffects.playBlip();
      return;
    }
    soundEffects.playPokedexCry();
    setSelectedMon(mon);
  };

  const handleCloseModal = () => {
    soundEffects.playBlip();
    setSelectedMon(null);
  };

  const discoveredCount = pokemonList.filter((p) => p.discovered).length;
  const totalCount = pokemonList.length;

  const filtered = pokemonList.filter((mon) => {
    const matchesFilter =
      filterMode === "all"
        ? true
        : filterMode === "discovered"
        ? mon.discovered
        : !mon.discovered;

    const matchesSearch =
      mon.name.toLowerCase().includes(searchQuery.toLowerCase()) ||
      mon.type.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase())) ||
      mon.id.toString().includes(searchQuery);

    return matchesFilter && matchesSearch;
  });

  return (
    <div className="pokedex-view">
      {/* Progress Bar for Discovered Pokémon Tracker (Matching User Pill Spec & 8bitcn XP Bar) */}
      <div className="dex-tracker-wrapper">
        <div className="dex-tracker-card gsc-frame">
          <div className="dex-count-row">
            <span className="dex-count-label">DISCOVERED:</span>
            <span className="dex-count-val">
              {discoveredCount} / {totalCount}
            </span>
          </div>
          <XpBar
            value={discoveredCount}
            max={totalCount}
            segments={25}
            variant="retro"
            progressColor="black"
            height={16}
            className="dex-main-xp-bar"
          />
        </div>
      </div>

      {/* Controls & Filter Bar */}
      <div className="dex-controls-bar gsc-frame">
        <div className="search-wrap">
          <Search size={16} className="search-icon" />
          <input
            type="text"
            className="dex-search-input"
            placeholder="Search by name, type (#155, Fire, Cyndaquil)..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />
        </div>

        <div className="filter-group">
          {[
            { id: "all", label: `ALL (${totalCount})` },
            { id: "discovered", label: `DISCOVERED (${discoveredCount})` },
            { id: "undiscovered", label: `UNDISCOVERED (${totalCount - discoveredCount})` },
          ].map((mode) => (
            <button
              key={mode.id}
              className={`filter-btn ${filterMode === mode.id ? "active" : ""}`}
              onClick={() => {
                soundEffects.playBlip();
                setFilterMode(mode.id);
              }}
            >
              {mode.label}
            </button>
          ))}
        </div>
      </div>

      {/* Pokédex Sprite Grid */}
      <div className="dex-grid">
        {filtered.map((mon) => {
          const spriteUrl = `https://img.pokemondb.net/sprites/crystal/normal/${mon.slug}.png`;
          return (
            <div
              key={mon.id}
              className={`dex-card gsc-card ${mon.discovered ? "discovered" : "grayed-out"}`}
              onClick={() => handleMonClick(mon)}
            >
              <div className="dex-num-badge">
                #{mon.id.toString().padStart(3, "0")}
              </div>

              <div className="dex-sprite-wrap">
                <img
                  src={spriteUrl}
                  alt={mon.discovered ? mon.name : "Undiscovered Pokémon"}
                  className="dex-8bit-sprite"
                  loading="lazy"
                  onError={(e) => {
                    // Fallback to gold normal if crystal specific fails
                    e.currentTarget.src = `https://img.pokemondb.net/sprites/gold/normal/${mon.slug}.png`;
                  }}
                />
              </div>

              <div className="dex-card-info">
                <h4 className="dex-mon-name">
                  {mon.discovered ? mon.name : "???"}
                </h4>

                <div className="dex-type-pills">
                  {mon.discovered ? (
                    mon.type.map((t) => (
                      <span key={t} className={`type-badge type-${t.toLowerCase()}`}>
                        {t}
                      </span>
                    ))
                  ) : (
                    <span className="type-badge type-unknown">UNKNOWN</span>
                  )}
                </div>

                <div className="dex-status-foot">
                  {mon.discovered ? (
                    <span className="status-discovered">
                      <CheckCircle2 size={12} /> Discovered (Step {mon.stepDiscovered || 0})
                    </span>
                  ) : (
                    <span className="status-undiscovered">
                      <HelpCircle size={12} /> Undiscovered
                    </span>
                  )}
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Pokédex Entry Modal */}
      {selectedMon && (
        <div className="retro-modal-backdrop" onClick={handleCloseModal}>
          <div
            className="retro-modal-content dex-entry-modal gsc-dialog-frame"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="modal-header">
              <div className="modal-title-wrap">
                <span className="modal-badge">
                  #{selectedMon.id.toString().padStart(3, "0")}
                </span>
                <h3 className="modal-title">{selectedMon.name}</h3>
                <span className="category-text">"{selectedMon.category}"</span>
              </div>
              <button
                className="modal-close-btn"
                onClick={handleCloseModal}
                aria-label="Close modal"
              >
                <X size={20} />
              </button>
            </div>

            <div className="dex-modal-body">
              {/* Left Column: Sprite & Physical Specs */}
              <div className="dex-modal-left">
                <div className="dex-modal-sprite-box">
                  <img
                    src={`https://img.pokemondb.net/sprites/crystal/normal/${selectedMon.slug}.png`}
                    alt={selectedMon.name}
                    className="dex-modal-sprite"
                  />
                  <div className="dex-type-row">
                    {selectedMon.type.map((t) => (
                      <span key={t} className={`type-badge type-${t.toLowerCase()}`}>
                        {t}
                      </span>
                    ))}
                  </div>
                </div>

                <div className="dex-physical-grid">
                  <div className="phys-item">
                    <span className="phys-k">HT:</span>
                    <span className="phys-v">{selectedMon.height || "—"}</span>
                  </div>
                  <div className="phys-item">
                    <span className="phys-k">WT:</span>
                    <span className="phys-v">{selectedMon.weight || "—"}</span>
                  </div>
                </div>
              </div>

              {/* Right Column: Lore & Agent Encounter Data */}
              <div className="dex-modal-right">
                <div className="gsc-dialog-box lore-dialog-box">
                  <div className="dialog-title">POKÉDEX DATA</div>
                  <p className="lore-text">{selectedMon.entry}</p>
                </div>

                <div className="agent-encounter-card gsc-panel">
                  <div className="encounter-header">AGENT DISCOVERY TELEMETRY</div>
                  <div className="encounter-row">
                    <span className="enc-k">Discovered Step:</span>
                    <span className="enc-v">Step {selectedMon.stepDiscovered}</span>
                  </div>
                  <div className="encounter-row">
                    <span className="enc-k">Encounter Location:</span>
                    <span className="enc-v">{selectedMon.encounterArea}</span>
                  </div>
                  {selectedMon.agentRole && (
                    <div className="encounter-row full">
                      <span className="enc-k">Agent Notes:</span>
                      <span className="enc-v">{selectedMon.agentRole}</span>
                    </div>
                  )}
                </div>

                {/* Base Stats Bar (8-Bit Segmented XP Bar style) */}
                <div className="dex-stats-section">
                  <div className="stat-row">
                    <span className="stat-name">HP ({selectedMon.hp || 50})</span>
                    <XpBar
                      value={selectedMon.hp || 50}
                      max={140}
                      segments={16}
                      variant="retro"
                      progressColor="green"
                      height={10}
                    />
                  </div>
                  <div className="stat-row">
                    <span className="stat-name">ATK ({selectedMon.attack || 50})</span>
                    <XpBar
                      value={selectedMon.attack || 50}
                      max={140}
                      segments={16}
                      variant="retro"
                      progressColor="red"
                      height={10}
                    />
                  </div>
                  <div className="stat-row">
                    <span className="stat-name">DEF ({selectedMon.defense || 50})</span>
                    <XpBar
                      value={selectedMon.defense || 50}
                      max={140}
                      segments={16}
                      variant="retro"
                      progressColor="blue"
                      height={10}
                    />
                  </div>
                  <div className="stat-row">
                    <span className="stat-name">SPD ({selectedMon.speed || 50})</span>
                    <XpBar
                      value={selectedMon.speed || 50}
                      max={140}
                      segments={16}
                      variant="retro"
                      progressColor="gold"
                      height={10}
                    />
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
