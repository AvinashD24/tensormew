import React from "react";

export default function FoundersTab() {
  const founders = [
    {
      name: "Shreyank Gupta",
      trainerId: "IDNo. 02482",
      money: "¥ 84,200",
      pokedexDiscovered: "15 PKMN",
      partner: "Cyndaquil",
      sprite: "https://img.pokemondb.net/sprites/crystal/normal/cyndaquil.png",
    },
    {
      name: "Arc Kumar",
      trainerId: "IDNo. 05918",
      money: "¥ 96,400",
      pokedexDiscovered: "15 PKMN",
      partner: "Onix",
      sprite: "https://img.pokemondb.net/sprites/crystal/normal/onix.png",
    },
    {
      name: "Avinash Dube",
      trainerId: "IDNo. 07142",
      money: "¥ 78,500",
      pokedexDiscovered: "15 PKMN",
      partner: "Totodile",
      sprite: "https://img.pokemondb.net/sprites/crystal/normal/totodile.png",
    },
    {
      name: "Yash Kulkarni",
      trainerId: "IDNo. 09321",
      money: "¥ 81,100",
      pokedexDiscovered: "15 PKMN",
      partner: "Chikorita",
      sprite: "https://img.pokemondb.net/sprites/crystal/normal/chikorita.png",
    },
  ];

  return (
    <div className="founders-view">
      {/* 4 Founder Trainer Cards (Clean, with no bio/contribution write-ups yet) */}
      <div className="founders-grid">
        {founders.map((founder) => (
          <div key={founder.name} className="trainer-card-wrapper gsc-frame">
            {/* Header */}
            <div className="trainer-card-header">
              <div className="card-badge-gold">TRAINER CARD</div>
              <div className="card-id-text">{founder.trainerId}</div>
            </div>

            {/* Profile Info */}
            <div className="trainer-profile-box">
              <div className="trainer-avatar-col">
                <div className="avatar-frame">
                  <img
                    src={founder.sprite}
                    alt={founder.name}
                    className="trainer-sprite-img"
                    loading="lazy"
                  />
                </div>
                <div className="fav-mon-tag">Partner: {founder.partner}</div>
              </div>

              <div className="trainer-identity-col">
                <h3 className="trainer-name">{founder.name}</h3>

                <div className="trainer-stats-table">
                  <div className="trainer-stat-line">
                    <span className="ts-label">MONEY:</span>
                    <span className="ts-val highlight-gold">{founder.money}</span>
                  </div>
                  <div className="trainer-stat-line">
                    <span className="ts-label">POKéDEX:</span>
                    <span className="ts-val">{founder.pokedexDiscovered}</span>
                  </div>
                  <div className="trainer-stat-line">
                    <span className="ts-label">REGION:</span>
                    <span className="ts-val">JOHTO</span>
                  </div>
                </div>
              </div>
            </div>

            {/* Badge Strip */}
            <div className="badges-case-strip">
              <div className="badge-case-title">BADGES:</div>
              <div className="badge-slot unlocked" title="Zephyr Badge">
                <span className="badge-pixel-symbol">✦</span>
                <span className="badge-slot-label">ZEPHYR</span>
              </div>
              <div className="badge-slot locked" title="Hive Badge">
                <span className="badge-slot-label">HIVE</span>
              </div>
              <div className="badge-slot locked" title="Plain Badge">
                <span className="badge-slot-label">PLAIN</span>
              </div>
              <div className="badge-slot locked" title="Fog Badge">
                <span className="badge-slot-label">FOG</span>
              </div>
            </div>
          </div>
        ))}
      </div>

      {/* Vision & Motivation Story Box */}
      <div className="vision-story-section gsc-dialog-box">
        <div className="dialog-title">FOUNDERS' STATEMENT &amp; VISION</div>
        <p className="vision-text">
          "We started TensorMew because we were dissatisfied with the current state of autonomous gaming agents:
          either brute-forcing pure RL for weeks without grasping game story logic, or burning through API credits
          prompting frontier LLMs on every single overworld step.
          <br /><br />
          Pokémon Crystal is one of the greatest games ever designed. It demands directional planning, resource management,
          stat knowledge, and long-horizon curiosity. By fusing fast recurrent spatial vision policies with deterministic WRAM
          introspection, we created an agent that can conquer Johto without shortcuts."
        </p>
        <div className="dialog-signature">
          — <strong>Shreyank Gupta, Arc Kumar, Avinash Dube &amp; Yash Kulkarni</strong>, Creators of TensorMew
        </div>
        <div className="dialog-prompt-arrow">▼</div>
      </div>
    </div>
  );
}
