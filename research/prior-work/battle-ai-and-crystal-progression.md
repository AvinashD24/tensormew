# Battle AI prior work + Pokémon Crystal full-game progression

Research date: 2026-10-19. Scope: (A) competitive-Pokémon battle AI / RL and what a Crystal battle policy needs; (B) everything needed to beat Pokémon Crystal end-to-end autonomously.

Provenance legend
- **[PRET]** = read directly from a fresh download of `pret/pokecrystal` master (maps/*.asm, data/trainers/*.asm, engine/battle/ai/*.asm, constants/*.asm, ram/wram.asm, docs/bugs_and_glitches.md) during this research. Highest confidence.
- **[SYM]** = checked against this repo's `rom/pokemon_crystal.sym` (US 1.1).
- **[WEB]** = from a fetched web page / paper summary (a small model summarised the page, so numbers are as-reported).
- **[UNVERIFIED]** = from memory, a search snippet, or inference. Do not rely on it without checking.

---------------------------------------------------------------------

## PART A. Battle AI / competitive RL prior work

### A1. Landscape (what exists, what it is good for)

| System | Approach | Relevance to Crystal in-game battles |
|---|---|---|
| **Metamon** (UT Austin RPL; Grigsby, Xie, Sasek, Zheng, Zhu; RLC 2025; arXiv 2504.04395) | Offline RL + imitation of Transformer policies on reconstructed Showdown human replays, then self-play fine-tuning. Gens 1-4 OU (+NU/UU/Ubers data), Gen 9 OU added later. MIT license. [WEB] | Only mature learned agent that covers **Gen 2**. But trained on *player-vs-player Showdown rules* (team preview-less, sleep clause, 6v6 OU sets, max DVs/stat exp). Not directly transferable to a Lv 20-50 in-game party fighting scripted trainers. Useful as architecture/data-format reference. |
| **Foul Play** (pmariglia) | Search bot on `poke-engine` (Rust); README says it plays singles in all gens; dynamax/z-moves unsupported. GPL-3.0. [WEB] | poke-engine's README lists **Gens 4-8 only** [WEB], so the "all generations" README claim conflicts for Gen 2 [UNVERIFIED which is right; check Makefile `GEN=` targets]. Metamon paper reports the agents beat Foul Play in Gens 1-2 and tie in Gens 3-4 (so Foul Play *was* run in Gen 1-2 by the Metamon authors). |
| **PokéChamp** (Karten et al.; ICML 2025; arXiv 2503.04094) | LLM-guided minimax: LLM does action sampling, opponent modelling, value estimate; uses a damage calculator tool. Gen 9 OU. Reported 76% vs best prior LLM bot, 84% vs best rule-based bot, projected Elo 1300-1500 (top 30-10%). Dataset >3M games. [WEB, author-reported] | Gen 9 only. Design pattern (damage-calc tool + shallow minimax + LLM only for tie-breaks) is portable. |
| **PokéLLMon** (Hu, Huang, Liu; Georgia Tech; arXiv 2402.01118) | LLM agent with in-context RL from battle feedback, knowledge-augmented generation (type chart, move effects), consistent action generation (anti "panic switching"). 49% ladder / 56% invited vs humans. [WEB] | Documents LLM failure modes (hallucinated type matchups, panic switching) that apply directly to an LLM-in-the-loop Crystal agent. Metamon paper says it is "uncompetitive in the long horizons of the early generations". |
| **PokéAgent Challenge** (NeurIPS 2025 competition; arXiv 2603.15563) | Battling track (Gen 1 OU, Gen 9 OU; 20M+ trajectory dataset) + Speedrunning track (**Pokémon Emerald** to first gym, 15 milestones). [WEB] | Winners in battling were all RL/search (FoulPlay won Gen 9 OU; PA-Agent won Gen 1 OU); 13/16 qualifiers extended the Metamon baselines. In speedrunning, raw VLMs w/o harness ~0% completion; best pure-LLM harness 1:29:17, winner (scripted-policy distillation) 40:13. CLI-agent harnesses "lose coherence over thousands of decisions". |
| **VGC-Bench** (arXiv 2506.10326) | Doubles / team-diversity benchmark. [WEB, only title seen] | Not relevant (doubles). |
| `poke-env` (hsahovic) | Python client for Showdown servers; MIT. Does not simulate; needs a Showdown server. [WEB] Gen 2 support not stated on README. | Tooling for any Showdown-based training; Metamon wraps Showdown as a gymnasium env. |
| `@smogon/calc` (damage-calc) | JS damage calculator; README: "all generations", credits Gen 1-6 (Honko) + 7/8/9. [WEB] | Gen 2 coverage is plausible but README does not name it. |
| `pokemon-showdown` `data/mods/gen2` | Files: conditions, formats-data, items, learnsets, moves, pokedex, rulesets, scripts, typechart (+README). README notes: no abilities, shared Special, Hidden Power from DVs, crits ignore defensive *boosts* but not drops, sleep 1-5 turns. [WEB] | Best open Gen 2 rules engine in JS; used by Metamon through a local server (`node pokemon-showdown start --no-security`). |

Metamon facts worth knowing [WEB, from its GitHub README]
- Observation: ~87 words of text + 48 numeric features per turn, only opponent's active mon observed; **9 discrete actions (4 moves + 5 switches)** for Gens 1-4 (`MinimalActionSpace`). Rewards: mostly win/loss + light shaping (damage dealt, HP recovered).
- Checkpoints on Hugging Face (needs the `amago` package): 15M/50M/200M paper policies, "Kakuna" 142M (best public), etc. Python 3.10, Linux tested.
- **Gen 2 is under-served**: the repo itself says its `competitive` team set has <30 Gen 2 teams and admits "inattention to Gens 2-4". Kakuna: ~66-71% GXE on local Gen 2 OU tables; paper: "All RL models sit inside the top 500 in Gen2OU".
- Ships a `GymLeader` heuristic baseline playable in `gen2ou`.

### A2. Why none of this drops straight into Crystal

1. **Opponent distribution is different.** Showdown agents face humans/self-play with optimised 6-mon teams. Crystal trainers have fixed parties and a *scripted, non-learning AI* (see A4). Exploiting that AI is easier than playing minimax against a human.
2. **Rule differences.** In-game: no team preview, no Sleep Clause, no item clause, move-selection/PP/accuracy as in cartridge, party can be underlevelled, **badge boosts apply** (A5), Battle Tower not relevant, trainer parties have **class-based DVs and zero stat exp** (A5). Showdown's gen2 mod assumes DV 15 / max stat exp by default (Smogon teambuilder guide [WEB]).
3. **Information.** In-game the player *can* know the entire enemy party in advance from the ROM (`data/trainers/parties.asm`), whereas Metamon must infer hidden sets. A Crystal policy can therefore use **perfect-information search** over a known enemy team.
4. **Objective.** Win the fight with minimal item/PP/time cost and *no whiteout* (a whiteout returns you to the last Pokémon Center with a money penalty [UNVERIFIED amount]); win-rate against a fixed AI, not ladder Elo.

Recommendation (inference): do **not** try to adapt Metamon for the in-game battle policy. Build (a) a deterministic-ish **damage/outcome estimator from RAM** and (b) 1-2 ply search over the known enemy party with a heuristic value function; add an LLM only for strategic choices (who to lead, when to heal/grind). Treat Metamon/PokéChamp as references for state encoding and failure modes.

### A3. Gen 2 mechanics a Crystal battle estimator must get right

Everything below marked [PRET] is from the disassembly; the rest is [WEB]/[UNVERIFIED].

- **Damage clamp**: `damage = min(base, 997) + 2` (cap 999, floor from MIN_DAMAGE 2) [PRET `engine/battle/effect_commands.asm:3025`]. Stat cap 999 [PRET]. Random factor: 217-255 /255 [UNVERIFIED, standard GSC knowledge]. Order: base damage -> crit x2 -> item/badge/weather -> STAB x1.5 -> type -> random [UNVERIFIED order].
- **Crit chance by stage** (out of 256-ish thresholds expressed as fractions) [PRET `data/battle/critical_hit_chances.asm`]: stage 0: 1/15, +1: 1/8, +2: 1/4, +3: 1/3, +4/+5/+6: 1/2. Crits ignore the defender's boosts and the attacker's drops only when favourable (see Smogon note: crits ignore stat changes/screens if attacker's offensive stage <= defender's defensive stage) [WEB].
- **Type chart** is a table with a Foresight/Odd-Egg separator at the end [PRET `data/types/type_matchups.asm`, 118 lines]. Key GSC facts visible there: Ghost->Psychic super effective (fixing Gen 1), Ghost->Steel and ->Dark resisted, Dark->Ghost super effective, Normal/Fighting->Ghost no effect (negated by Foresight), Steel and Dark exist. Use the file directly as the ground-truth chart.
- **Physical/Special by type**: Normal, Fighting, Flying, Poison, Ground, Rock, Bug, Ghost, (Steel) physical; Fire, Water, Grass, Electric, Ice, Psychic, Dark, Dragon special [WEB Smogon]. Single Special stat/DV.
- **Stats**: DVs 0-15 (HP DV derived from the low bits of the other four), stat exp 0-65535 per stat. Trainer mons: DVs come from the **trainer class** [PRET `data/trainers/dvs.asm`; e.g. Falkner 9/10/7/7, Whitney 8/8/8/8, Clair 7/12/13/13, Will/Bruno/Koga/Lance(CHAMPION) 13/12/13/13, Karen 7/15/13/15]; they are created with stat exp 0 (`CalcMonStats` called with `b = FALSE` in `engine/pokemon/move_mon.asm`) [PRET, inference from code reading].
- **Priority** list and speed ties [WEB Smogon GSC mechanics]: +2 Protect/Detect/Endure, +1 Quick Attack/Mach Punch/ExtremeSpeed, -1 Counter/Mirror Coat/Roar/Whirlwind/Vital Throw; Roar/Whirlwind only work if user moves last.
- **Sleep** 1-6 turns (Rest 2); Toxic counter 1/16 growing, resets to 1/8 flat on switch; Spikes 1 layer 1/8; Belly Drum halves HP; Reflect/Light Screen double defence (rollover bug above 1024) [WEB Smogon; PRET bugs doc confirms the wrap-around].
- **Known in-game engine bugs** an estimator should model or avoid [PRET `docs/bugs_and_glitches.md`]: moves with 100% secondary-effect chance fail 1/256 of the time; Glacier Badge may fail to boost Sp. Def depending on Sp. Atk; Thick Club/Light Ball and Reflect/Light Screen wrap above 1024; Belly Drum boosts even under 50% HP (in-game); Present damage differs link vs single-player; Counter/Mirror Coat still work if opponent uses an item; **switching out/against a Pokémon with max HP below 4 freezes the game**; catch-rate formula breaks for max HP > 341; Rival2 has lower DVs than Rival1.
- **Tools to compute damage**: Showdown gen2 mod / `@smogon/calc` (JS) or re-implement from `engine/battle/effect_commands.asm`. For fidelity prefer reading the actual game: PyBoy can run the real battle engine, so a *simulator may be unnecessary* if you only need action selection from RAM-derived features.

### A4. Crystal trainer AI (what you are fighting) [PRET unless stated]

Data: `data/trainers/attributes.asm` (per trainer class: 2 items, base reward, AI layer flags, switch/item flags); logic: `engine/battle/ai/{scoring,items,switch,move,redundant}.asm`; layer constants in `constants/trainer_data_constants.asm`.

**Move selection**: each of the 4 moves gets a score (start equal); each enabled layer adds/subtracts; **the lowest score is chosen, ties random**. Layers (bit order): `AI_BASIC, AI_SETUP, AI_TYPES, AI_OFFENSIVE, AI_SMART, AI_OPPORTUNIST, AI_AGGRESSIVE, AI_CAUTIOUS, AI_STATUS, AI_RISKY`. The AI **does not cheat** (no peeking beyond what the engine exposes; Pastebin guide [WEB]) and in the Battle Tower every trainer uses Falkner's layers [WEB pret wiki].

Layer behaviours (pret wiki summary [WEB] + file headers [PRET]):
- BASIC: avoid redundant/failing moves (e.g. Dream Eater on awake target, status moves when target already statused/Safeguarded/Substitute).
- SETUP: stat-up on its first turn ~50%, otherwise discouraged; stat-down on player's first turn.
- TYPES: prefer super-effective, avoid resisted/immune (ignores base power).
- OFFENSIVE / AGGRESSIVE: AGGRESSIVE prefers the move with highest computed damage (STAB, type, weather); "reckless" multi-hit/rampage/self-destruct exempt.
- CAUTIOUS: avoids stall/residual moves in bad situations (bug: may fail to discourage residual moves).
- OPPORTUNIST: discourages stall moves when its HP is low.
- STATUS: type-immunity checks for status moves (Poison vs Poison/Steel etc.).
- RISKY: strongly prefers moves that can KO this turn; holds back Selfdestruct/OHKO at full HP.
- SMART: per-effect routines (Sleep, Selfdestruct, Bide, screens, Dream Eater, Mean Look, Encore, Counter, Disable, Spite, Heal Bell...). Known bugs: encourages Mean Look when its own mon is badly poisoned; does not encourage Sunny Day with Solar Beam/Moonlight; Future Sight repeat; Nightmare; etc.

**All Johto + Kanto gym leaders, Elite Four (Will, Koga, Bruno, Karen), Lance (class `CHAMPION`), Red, Blue, Rival1/Rival2, Cal use the same full set: `AI_BASIC | AI_SETUP | AI_SMART | AI_AGGRESSIVE | AI_CAUTIOUS | AI_STATUS | AI_RISKY`, switching `SWITCH_SOMETIMES`, item mode `CONTEXT_USE`.** Note they do **not** include AI_TYPES or AI_OPPORTUNIST (those are on Rocket Executives/grunts/mid trainers). Rocket Executives: `BASIC|SETUP|TYPES|SMART|OPPORTUNIST|CAUTIOUS|STATUS|RISKY`. Twins: `NO_AI` (fully random moves). Cheap classes (Youngster `BASIC|STATUS`, Hiker `BASIC|OFFENSIVE|STATUS`, ...) are weak.

**Leader item loadouts** (2 slots) [PRET]: Falkner/Whitney/Bugsy/Morty none; Pryce & Jasmine Hyper Potion; Chuck Full Heal; Clair Full Heal + Hyper Potion; Brock/Surge/Erika/Sabrina Hyper Potion; Misty Full Heal; Janine Dire Hit; Blaine Max Potion + Full Heal; Will Max Potion; Bruno Max Potion; Karen Full Heal + Max Potion; Koga Full Heal + Full Restore; **Lance (CHAMPION) Full Heal + Full Restore**; **Red and Blue Full Restore x2**. Rival1/2 none. Rocket Executives none.

**Item-use logic** (`engine/battle/ai/items.asm`) [PRET]: only if a trainer has items and not in Battle Tower; **only the highest-level Pokémon in its party is eligible** (`.IsHighestLevel`); heal items under `CONTEXT_USE` fire when HP <= 50%, always when <= 25%, and with ~20% chance in between [PRET code, interpretation of carry flags is [UNVERIFIED nuance]]; Full Heal-type items fire for statuses (sleep/freeze always in context mode; Toxic count >= 4 ~50%). Using an item **takes the AI's turn** and makes it move first (`wEnemyGoesFirst = 1`). Practical implication: against Red/Blue/Lance, expect 1-2 free-heal turns on their strongest mon; sustained damage races must account for Full Restore. Known bug: AI Full Heal/Full Restore does not cure confusion, Nightmare, or burn/para stat drops.

**Switch logic** (`switch.asm`) [PRET]: if `SWITCH_SOMETIMES`, AI evaluates type matchups of its active mon vs yours (and its moves vs yours); scores 0x10/0x20/0x30 map to thresholds like 20%/50%/~80% switch probability; won't switch if you are trapping it, it is wrapped, or locked in. So a poor matchup does *not* guarantee a switch - expect at most occasional switches.

**What this means for a policy**: leaders are "greedy best-damage + setup on turn 1 + heal at low HP". They are exploitable by (i) outleveling, (ii) a single sweeper that KOs before heals matter, (iii) status (sleep/para) on the highest-level mon, (iv) abusing the "highest-level only" item rule by killing the non-top mons first.

### A5. Crystal-specific battle facts the policy needs

- **Badge boosts** [PRET `data/types/badge_type_boosts.asm`, `bugs_and_glitches.md`]: each badge boosts one type's damage (Zephyr Flying, Hive Bug, Plain Normal, Fog Ghost, Mineral Steel, Storm Fighting, Glacier Ice, Rising Dragon; Boulder Rock, Cascade Water, Thunder Electric, Rainbow Grass, Soul Poison, Marsh Psychic, Volcano Fire, Earth Ground). Badges also give stat multipliers (Attack/Defense/Speed/Special, ~1/8) [UNVERIFIED which badge -> which stat; the pret `bugs_and_glitches.md` says Glacier boosts Sp. Def].
- **Level cap-ish**: no hard level cap by badges in vanilla; obedience is only for traded Pokémon (Crystal has none for in-game). [UNVERIFIED]
- **Party data in RAM** [SYM]: `wPartyMon1Species` DCDF, Item DCE0, Moves DCE1, StatExp DCEA, DVs DCF4, PP DCF6, Level DCFE, HP DD01; stride 0x30; party count DCD7. Enemy trainer party `wOTPartyCount` D280, `wOTPartyMon1Species` D288; enemy mon species D206, moves D208, level D213, HP D216; `wBattleMode` D22D (0 none, 1 wild, 2 trainer), `wBattleType` D230, `wTrainerClass` D233.
- **Enemy parties are fixed** in `data/trainers/parties.asm`; the policy can pre-load them keyed by `(wTrainerClass, wTrainerID)`.

Boss rosters (verbatim from `data/trainers/parties.asm`, level / species / moves) [PRET]

Johto gyms
- Falkner: Pidgey L7, Pidgeotto L9 (listing not extracted; standard) [UNVERIFIED exact moves]
- Bugsy: Metapod 14, Kakuna 14, Scyther 16 (Quick Attack, Leer, Fury Cutter)
- Whitney: Clefairy 18 (Doubleslap, Mimic, Encore, Metronome), **Miltank 20 (Rollout, Attract, Stomp, Milk Drink)**
- Morty: Gastly 21, Haunter 21 (Hypnosis, Mimic, Curse, Night Shade), Haunter 23, **Gengar 25 (Hypnosis, Shadow Ball, Mean Look, Dream Eater)**
- Chuck: Primeape 27, Poliwrath 30 (Hypnosis, Mind Reader, Surf, DynamicPunch)
- Jasmine: Magnemite 30 x2 (Thunderbolt, Supersonic, Sonicboom, Thunder Wave), Steelix 35 (Screech, Sunny Day, Rock Throw, Iron Tail)
- Pryce: Seel 27, Dewgong 29, Piloswine 31 (Icy Wind, Fury Attack, Mist, Blizzard)
- Clair: Dragonair 37 x3 (Thunder Wave, Slam, DragonBreath + Surf/Thunderbolt/Ice Beam), Kingdra 40 (Smokescreen, Surf, Hyper Beam, DragonBreath)

Elite Four / Champion
- Will: Xatu 40, Jynx 41, Exeggutor 41, Slowbro 41, Xatu 42
- Koga: Ariados 40, Venomoth 41, Forretress 43 (Protect, Swift, Explosion, Spikes), Muk 42, Crobat 44
- Bruno: Hitmontop 42, Hitmonlee 42, Hitmonchan 42, Onix 43, Machamp 46
- Karen: Umbreon 42, Vileplume 42, Gengar 45 (Lick, Spite, Curse, Destiny Bond), Murkrow 44, Houndoom 47
- **Lance (class CHAMPION)**: Gyarados 44, Dragonite 47 (Thunder Wave, Twister, Thunder, Hyper Beam), Dragonite 47 (Blizzard variant), Aerodactyl 46, Charizard 46, Dragonite 50 (Fire Blast, Safeguard, Outrage, Hyper Beam)

Kanto (levels are roughly 40-58)
- Brock: Graveler 41, Rhyhorn 41, Omastar 42, Onix 44, Kabutops 42. Misty: Golduck 42, Quagsire 42, Lapras 44, Starmie 47. Lt. Surge: Raichu 44, Electrode 40 x2, Magneton 40, Electabuzz 46. Erika: Tangela 42, Jumpluff 41, Victreebel 46, Bellossom 46. Janine: Crobat 36, Weezing 36 x2, Ariados 33, Venomoth 39. Sabrina: Espeon 46, Mr. Mime 46, Alakazam 48. Blaine: Magcargo 45, Magmar 45, Rapidash 50. Blue: Pidgeot 56, Alakazam 54, Rhydon 56, Gyarados 58, Exeggutor 58, Arcanine 58 (ExtremeSpeed).

**Red (Mt. Silver)**: Pikachu **81** (Charm, Quick Attack, Thunderbolt, Thunder), Espeon 73 (Mud-Slap, Reflect, Swift, Psychic), Snorlax 75 (Amnesia, Snore, Rest, Body Slam), Venusaur 77 (Sunny Day, Giga Drain, Synthesis, SolarBeam), Charizard 77 (Flamethrower, Wing Attack, Slash, Fire Spin), Blastoise 77 (Rain Dance, Surf, Blizzard, Whirlpool) [PRET]. Two Full Restores. (Third-party guides say Pikachu 81 is highest [WEB]. Note: LLM-run reports Red at "L70-80" [WEB]; disassembly gives 73-81.)

### A6. Team-building and battle-value estimation (for a one-player RPG run)

- There is no team-preview team-building in-game; "team-building" = *which Pokémon to catch/evolve/train* and moveset (TM/HM/move reminder/Tutor). Known strong Crystal choices from the community (Typhlosion/Meganium/Feraligatr starters, Pikachu/Snorlax/Gyarados-style carries, level-leader strategies) are [UNVERIFIED]; the Gemini 3 Pro run beat Red with a **single Lv 75 Typhlosion plus low-level cannon fodder, a Revive loop, Smokescreen and Leftovers stalling**, ~7 real-time hours for the Red fight [WEB, blog.jcz.dev].
- Battle-value estimate for a fight = expected damage race. Inputs available from RAM: own/enemy species, levels, HP, moves/PP, DVs/stat exp (own), type matchup table, badge boosts, stat stages. A minimal reliable estimator: for each (own mon, enemy mon) compute max damage per move, hits-to-KO both ways, speed order; bonus for status moves; value = P(win 1v1) aggregated over the known enemy order (AI order is fixed; leader parties are in listed order, switches are occasional).
- Metamon/PokéChamp-style *learned* value functions need large data; for a 1-shot run the closed-form estimator above is more practical and exact because the opponent roster is known.

---------------------------------------------------------------------

## PART B. Pokémon Crystal: beating the whole game autonomously

### B1. Existing evidence that full-Crystal autonomy is feasible, and where agents fail
- **GPT-5.1** completed Crystal with a "fairly minimal harness" in ~9,454 reasoning steps / ~108 real-time hours (~3x slower than a human) [WEB, LessWrong "Insights into Claude Opus 4.5 from Pokémon"]. **Gemini 3 Pro** beat Crystal including Red on turn 24,178 using ~1.88B tokens; Gemini 2.5 Pro did not [WEB, blog.jcz.dev]. As of the sources found, no Claude run of Crystal was confirmed [WEB; unverified status after early 2026].
- Harness features that mattered [WEB]: RAM-derived text state + screenshot; a fog-of-war "mental map" the model fills in; map markers (Crystal NPCs are only readable from RAM when on screen); notepad; A* pathfinding tool; a battle strategist sub-agent; "training wheels" (split direction/action inputs - Crystal's nickname screen gave "G" instead of "GEMINI"); LLM-powered navigator; minimap.
- Reported stall points in Crystal [WEB blog.jcz.dev]: **Olivine Lighthouse** (2.5 Pro looped for ~16.4k turns; 4th-floor pits looked like traps), **Goldenrod Underground switch puzzle** (hint is in Rocket Grunts' post-battle text; skipped dialogue -> ~2 days), **Pokégear radio** (Left/Right switches Pokégear cards, Up/Down tunes; needed to tune the Pokéflute channel; that station is part of the Goldenrod/Radio Tower storyline in Crystal), **Whitney's Miltank** (2.5 Pro lost then ground for 2+ real days), Blackthorn boulder puzzle misjudged, Mt. Mortar boulder, Sprout Tower (segmented map confuses pathfinding).
- Claude/Red failure modes [WEB LessWrong]: stuck for ~4 days circling Erika's gym; misreading walls as targets; wrong notes poisoning progress; not understanding trainer line-of-sight; wasting PP; hoarding/discarding useful items.

### B2. RAM / event-flag / disassembly resources

Primary
- **`pret/pokecrystal`** (GitHub): `ram/wram.asm` (symbols), `constants/event_flags.asm` (event bit indices, 2048 flags), `constants/engine_flags.asm` (badges, flypoints, daily flags, Radio Card etc.), `maps/*.asm` (every script, NPC, warp, `checkevent`/`setevent`), `data/trainers/{parties,attributes,dvs}.asm`, `engine/battle/ai/*`, `docs/{bugs_and_glitches,design_flaws,event_commands,map_event_scripts}.md`, wiki (183 pages; "Hard-coded logic", "Improve the enemy trainer AI", scripting commands, wild encounter data). Built docs at `pret.github.io/pokecrystal`.
- **This repo's `rom/pokemon_crystal.sym`** (US 1.1) - authoritative addresses for our ROM. Key ones verified present [SYM]:

| Symbol | Addr | Notes |
|---|---|---|
| wPartyCount | DCD7 | species list DCD8.., mon structs from DCDF (stride 0x30) |
| wPartyMon1Level / HP | DCFE / DD01 | |
| wMapGroup / wMapNumber / wYCoord / wXCoord | DCB5 / DCB6 / DCB7 / DCB8 | |
| wMoney | D84E (3 bytes BE) | |
| wJohtoBadges / wKantoBadges | D857 / D858 | bit0..7, order below |
| wTMsHMs | D859 | one byte per TM/HM |
| wNumItems / wItems | D892 / D893 | pairs, terminated 0xFF |
| wKeyItems | D8BD | |
| wEventFlags | DA72 | 2048 bits = 256 bytes |
| wDailyFlags1/2, wSwarmFlags | DC1E, DC1F, DC20 | |
| wPhoneList | DC7C | |
| wBattleMode, wBattleType, wTrainerClass | D22D, D230, D233 | |
| wEnemyMonSpecies / Level / HP / Moves | D206 / D213 / D216 / D208 | |
| wOTPartyCount / wOTPartyMon1Species | D280 / D288 | |
| wTimeOfDay | D269 | (0 morn, 1 day, 2 nite [UNVERIFIED]) |
| wCurDay | D4CB | weekday |
| wBugContestMinsRemaining | D46C | |
| wScriptRunning, wMapStatus | D438, D432 | |
| wPlayerDirection | D4DE | |
| wCurPartyMon | D109 | |
| wRepelEffect | DCA1 | |
| wPokedexCaught | DE99 | |

- **Flag bit addressing**: event index `n` -> byte `DA72 + n//8`, bit `n%8` (LSB first) [PRET `flag_array` convention; computed here]. Indices were computed by replaying `const`/`const_skip`/`const_next` in `event_flags.asm`; the computed index for `EVENT_BEAT_FALKNER` (1213) and `EVENT_GOT_A_POKEMON_FROM_ELM` (26) agree with this repo's `src/story_flags.py`, supporting the method. Selected flags (index -> addr.bit):

| Event | idx | addr.bit |
|---|---|---|
| EVENT_GOT_A_POKEMON_FROM_ELM | 26 | DA75.2 |
| EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON | 30 | DA75.6 |
| EVENT_GOT_HM01_CUT | 16 | DA74.0 |
| EVENT_GOT_HM02_FLY | 17 | DA74.1 |
| EVENT_GOT_HM03_SURF | 18 | DA74.2 |
| EVENT_GOT_HM04_STRENGTH | 19 | DA74.3 |
| EVENT_GOT_HM05_FLASH | 20 | DA74.4 |
| EVENT_GOT_HM06_WHIRLPOOL | 21 | DA74.5 |
| EVENT_GOT_HM07_WATERFALL | 1672 | DB43.0 |
| EVENT_CLEARED_RADIO_TOWER | 33 | DA76.1 |
| EVENT_CLEARED_ROCKET_HIDEOUT | 34 | DA76.2 |
| EVENT_GOT_SECRETPOTION_FROM_PHARMACY | 35 | DA76.3 |
| EVENT_GOT_SS_TICKET_FROM_ELM | 36 | DA76.4 |
| EVENT_HERDED_FARFETCHD | 41 | DA77.1 |
| EVENT_FOUGHT_SUDOWOODO | 42 | DA77.2 |
| EVENT_CLEARED_SLOWPOKE_WELL | 43 | DA77.3 |
| EVENT_BEAT_ELITE_FOUR | 68 | DA7A.4 |
| EVENT_GOT_ITEMFINDER / BICYCLE / SQUIRTBOTTLE | 90 / 91 / 92 | DA7D.2 / .3 / .4 |
| EVENT_GOT_CLEAR_BELL | 120 | DA81.0 |
| EVENT_RETURNED_MACHINE_PART | 201 | DA8B.1 |
| EVENT_RESTORED_POWER_TO_KANTO | 205 | DA8B.5 |
| EVENT_GOT_PASS_FROM_COPYCAT | 209 | DA8C.1 |
| EVENT_TALKED_TO_OAK_IN_KANTO | 225 | DA8E.1 |
| EVENT_FOUGHT_SUICUNE / GOT_RAINBOW_WING | 821 / 822 | DAD8.5 / .6 |
| EVENT_BEAT_FALKNER, BUGSY, WHITNEY, MORTY, **JASMINE**, **CHUCK**, PRYCE, CLAIR | 1213..1220 | DB09.5 .. DB0A.4 |
| EVENT_BEAT_BROCK, MISTY, LTSURGE, ERIKA, JANINE, SABRINA, BLAINE, BLUE | 1221..1228 | DB0A.5 .. DB0B.4 |
| EVENT_BEAT_ELITE_4_WILL/KOGA/BRUNO/KAREN | 1464..1467 | DB29.0-.3 |
| EVENT_BEAT_CHAMPION_LANCE | 1468 | DB29.4 |
| EVENT_OPENED_MT_SILVER | 1871 | DB5B.7 |
| EVENT_TEAM_ROCKET_DISBANDED | 1889 | DB5E.1 |
| EVENT_RED_IN_MT_SILVER (Red's *appear* flag; set = Red absent/removed) | 1890 | DB5E.2 |
| EVENT_BLUE_IN_CINNABAR | 1909 | DB60.5 |

  All indices above are [computed from PRET, not run against a live save]; test against a known save before trusting. **Discrepancies found in this repo's `src/story_flags.py` vs the disassembly**: (a) it lists `EVENT_GOT_HM01_CUT` = 14 .. `HM06` = 19, but the disassembly places them at 16..21 (8 temp flags + 8 Johto-TM flags come first); (b) it lists `EVENT_BEAT_CHUCK` = 1217 / `JASMINE` = 1218, but pokecrystal order is Falkner, Bugsy, Whitney, Morty, **Jasmine, Chuck**, Pryce, Clair (Jasmine 1217, Chuck 1218). Recommend the badge bytes (D857/D858) as the primary progress signal instead of those flags.
- **Badge bit order** (`engine_flags.asm`) [PRET]: Johto D857 bit0..7 = Zephyr, Hive, Plain, Fog, Mineral (Jasmine), Storm, Glacier, Rising. Kanto D858 bit0..7 = Boulder (Brock), Cascade (Misty), Thunder (Surge), Rainbow (Erika), Soul (Janine), Marsh (Sabrina), Volcano (Blaine), Earth (Blue). Datacrystal agrees on a subset (Misty $02, Surge $04, Erika $08, Janine $10, Sabrina $20) [WEB].
- **datacrystal.tcrf.net Pokémon Crystal RAM map** [WEB]: partially matches the sym (money D84F-D850?? vs sym D84E 3 bytes - datacrystal is off by one; trust the sym). Treat datacrystal as a lead, not truth.
- Other: TASVideos GBC Gen 2 resources (speedrun routes; "Pokémon Gold/Silver/Crystal in 2:54:21" any% TAS) [WEB]; Bulbapedia Crystal walkthrough and StrategyWiki (both returned HTTP 403 to my fetcher; not usable here); nationaldex.io guide (fetched; some details unreliable - cross-check against pret).
- Useful community codebases for the Red analogue (PyBoy + RAM + RL): `drubinstein/pokemonred_puffer`, PokeRL (arXiv 2604.10812) [WEB, Red only]. No Crystal RL project was found in searches.

### B3. HM / field-move map (verified from pret maps) [PRET]

| HM | Where obtained | Badge to use outside battle | First *required* use / notes |
|---|---|---|---|
| HM01 Cut | **Ilex Forest** - the Charcoal Kiln master after you herd Farfetch'd (`IlexForest.asm` L348-353; flag `EVENT_HERDED_FARFETCHD`) | **Hive** (Bugsy) | Needed to leave Ilex Forest / many Kanto trees (Vermilion, Route 2, Viridian). Also bike-path trees. |
| HM05 Flash | **Sprout Tower 3F** (Elder Li, `SproutTower3F.asm`) | **Zephyr** (Falkner) | Optional in practice (Dark Cave, Union Cave dim). Not required for the critical path [UNVERIFIED]. |
| HM03 Surf | **Ecruteak Dance Theater**: the Surf guy gives it only after **all five Kimono Girls are beaten** (Naoko, Sayo, Zuki, Kuni, Miki - flags `EVENT_BEAT_KIMONO_GIRL_*`) (`DanceTheater.asm`) | **Fog** (Morty) | Route 40/41 to Cianwood, Dragon's Den, Kanto sea routes, Cinnabar. Crystal-specific (GS gave it differently). |
| HM04 Strength | **Olivine Cafe** (`OlivineCafe.asm`) | **Plain** (Whitney) | Boulders: Ice Path, Burned Tower B1F, Slowpoke Well B1F, Union Cave B1F, Mt. Mortar, Whirl Islands B1F, Blackthorn Gym 2F, Cianwood Gym [PRET maps that contain boulder objects]. Which are *mandatory* is [UNVERIFIED]; since Slowpoke Well/Burned Tower precede Strength canonically, those boulders are optional. |
| HM02 Fly | **Cianwood City** (Chuck's wife, `CianwoodCity.asm`) after Chuck | **Storm** (Chuck) | Huge time saver. Flypoints set on first visit (`ENGINE_FLYPOINT_*`). |
| HM06 Whirlpool | **Team Rocket Base B2F, Mahogany**: Lance gives it after the Electrode trap (`TeamRocketBaseB2F.asm`) | **Glacier** (Pryce) | Required for Whirl Islands (optional Lugia area). Not needed for main line [UNVERIFIED]. |
| HM07 Waterfall | **Ice Path 1F item ball** (`IcePath1F.asm`; flag `EVENT_GOT_HM07_WATERFALL`) | **Rising** (Clair) | Needed for **Mt. Silver/Silver Cave** progress (Surf + Waterfall inside) [UNVERIFIED but widely reported], Mt. Mortar, Whirl Islands. Rising Badge is awarded **in Dragon's Den after the quiz**, not in the gym (see Blackthorn). |
| Rock Smash / Headbutt | TM08 Rock Smash (Route 36 Rock Smash guy / Dept Store 5F?), TM02 Headbutt (Ilex Forest) [PRET partial] | none | Optional, though Rock Smash rocks block some item paths. |

### B4. World-state triggers that change the route (important for an agent) [PRET]

- **Badge-count scripts**: gym scripts call a handler when `VAR_BADGES` reaches 6 or 7. Example `OlivineGym.asm`: after receiving the Mineral Badge, `ifequal 6 -> GoldenrodRocketsScript` (`clearevent EVENT_GOLDENROD_CITY_ROCKET_TAKEOVER`), `ifequal 7 -> RadioTowerRocketsScript` (`setflag ENGINE_ROCKETS_IN_RADIO_TOWER`, civilians out of Goldenrod, black belt blocks stairs, `EVENT_MAHOGANY_TOWN_POKEFAN_M_BLOCKS_EAST`, "weird broadcast" call). **So the Radio Tower raid is driven by your Johto badge count (not by a specific gym), and which gym you beat 6th/7th decides the timing.** Which other gyms contain the same handler is [UNVERIFIED].
- **Radio Tower clear** (`RadioTower5F.asm`) sets `EVENT_CLEARED_RADIO_TOWER`, `EVENT_TEAM_ROCKET_DISBANDED`, gives the **Clear Bell** (`EVENT_GOT_CLEAR_BELL`) and sets `EVENT_BLACKTHORN_CITY_SUPER_NERD_BLOCKS_GYM` (gym access). Tin Tower entrance (Ecruteak) is gated on `EVENT_CLEARED_RADIO_TOWER` + Clear Bell/Suicune/Koji flags (`EcruteakTinTowerEntrance.asm`).
- **Mahogany**: Rocket Hideout under the Mahogany Mart; `EVENT_CLEARED_ROCKET_HIDEOUT` unblocks Route 43 gate and Lake of Rage (`Route43MahoganyGate.asm`, `LakeOfRage.asm`, `MahoganyTown.asm`).
- **Oak's Lab opens Mt. Silver** only when `VAR_BADGES == NUM_BADGES (16)` (`OaksLab.asm`: `ifequal NUM_BADGES, .OpenMtSilver` sets `EVENT_OPENED_MT_SILVER`). Talking to Oak at 8 Johto badges only prompts "get the Kanto badges".
- **Victory Road Gate** (Johto) requires **all 8 Johto badges** (`VAR_BADGES > 7`, `VictoryRoadGate.asm`). The officer turns you back otherwise. There are no boulders in Victory Road (no Strength).
- **S.S. Ticket** comes from Prof. Elm *after* `EVENT_BEAT_ELITE_FOUR` (`ElmsLab.asm` L93-95). Ship schedule is weekday-gated (below).
- **Blue** is not in his gym until you visit **Cinnabar Island** and talk to him there (`CinnabarIsland.asm`: he teleports and clears `EVENT_VIRIDIAN_GYM_BLUE`); `EVENT_BLUE_IN_CINNABAR` otherwise leaves Viridian Gym empty. So Earth Badge requires Cinnabar (Surf).
- **Machine Part** for the Power Plant is a *hidden item* in Cerulean Gym (`CeruleanGym.asm`: `hiddenitem MACHINE_PART`). Returning it sets `EVENT_RETURNED_MACHINE_PART` then `EVENT_RESTORED_POWER_TO_KANTO` (`PowerPlant.asm`), which enables the Magnet Train (also needs the Rail Pass from the Copycat, `EVENT_GOT_PASS_FROM_COPYCAT`) and dozens of NPC state changes (phone/rematch scripts check it). This is a classic agent trap (hidden item without Itemfinder-style search).

### B5. Time gates, weekday events, phone calls

Clock RAM: `wTimeOfDay` D269, weekday `wCurDay` D4CB [SYM]. Time-of-day is derived from the RTC (`checktime MORN/DAY/NITE`). In an emulator the RTC must be controlled or advanced; see "agent notes" below.

Weekday-gated scripts found via `readvar VAR_WEEKDAY` in pret maps [PRET]:

| Day | Event |
|---|---|
| Mon | Route 40 Monica (Sharp Beak); Mt. Moon Square Clefairy dance (Mon night, `checktime NITE`) - Moon Stone; Goldenrod Underground: haircut/shop days |
| Tue | Route 29 Tuscany (Pink Bow); **Bug-Catching Contest** (Tue/Thu/Sat); Dragon's Den rival appearance on Tue/Thu (`DragonsDenB1F.asm`) |
| Wed | Lake of Rage Wesley (Black Belt); Goldenrod Move Tutor (Wed/Sat) |
| Thu | Route 36 Arthur (Hard Stone); Bug-Catching Contest; Dragon's Den rival appears |
| Fri | Route 32 Frieda (Poison Barb); Union Cave B2F Lapras |
| Sat | Blackthorn Santos (Spell Tag); Goldenrod Move Tutor; Bug-Catching Contest; Goldenrod Underground shop open Sat/Sun |
| Sun | Route 37 Sunny (Magnet); Goldenrod Dept Store 5F TM27 Return event |

- **Bug-Catching Contest**: Route 35 National Park Gate scripts: contest runs **Tue, Thu, Sat**; `BUG_CONTEST_BALLS = 20`, `BUG_CONTEST_MINUTES = 20`, seconds 0 (`script_constants.asm`); timer RAM `wBugContestMinsRemaining` D46C. Not required for completion (prize Sunny Day TM11 / items). During the contest the party is forced to 1 mon [UNVERIFIED detail]; it is a time sink / softlock hazard because the contest ends with a warp. Agents should skip it or special-case it.
- **S.S. Aqua** boarding days: Olivine port allows boarding Mon and Fri (script sends Sat/Sun -> "next ship Monday", Tue/Wed/Thu -> "Friday"); Vermilion port Wed and Sun (Mon/Tue -> Wednesday, Thu/Fri/Sat -> Sunday); the first-ever trip (`EVENT_FAST_SHIP_FIRST_TIME`) bypasses the weekday check at Olivine [PRET `OlivinePort.asm`, `VermilionPort.asm`]. After the first trip an agent may need to wait up to ~4 in-game days; alternatives: Magnet Train (needs Power restored + Pass) or Fly.
- **Phone calls**: Pokégear (`ENGINE_POKEGEAR`, `ENGINE_PHONE_CARD`, `wPhoneList` DC7C) calls interrupt movement and open a text box; "ask for phone number" prompts appear on many trainers (`PHONE_CONTACTS_FULL`, `EVENT_*_ASKED_FOR_PHONE_NUMBER` in Route 26/27/32...). Not required for completion; **decline** or handle prompts (a yes/no) deterministically. Rematch scripts depend on `EVENT_RESTORED_POWER_TO_KANTO` and `EVENT_BEAT_ELITE_FOUR`.
- **Time-of-day**: wild encounter tables (morn/day/nite) change; Crystal's Kanto/Johto dark-time effects are mostly cosmetic except Mt. Moon Square (Monday night) and the Goldenrod Radio schedule; Slowpoke Well/Union Cave unaffected. Several Pokémon (e.g., Snubbull/Slowpoke) are time-restricted for catching [UNVERIFIED]. Daily flags reset at midnight (`wDailyFlags1/2` DC1E/DC1F): daily Kurt balls, Day-Care, Sudowoodo-adjacent? etc.
- **Mom's money / Mom saving**: `wMomSavingMoney` flags; Mom auto-saves a percentage of money after battles if enabled, and the "shopping" events deliver items. Not required.
- **RTC in PyBoy**: Crystal uses the MBC3 RTC; PyBoy's RTC advances with emulated time. Time-gated events (weekday ships, Bug Contest) may therefore require *advancing the clock* (frame-count based) or patching RAM (`wCurDay`) - behaviour unverified here [UNVERIFIED]. Start the Crystal "set time" screen deterministically.

### B6. Hard spots and softlock risks (consolidated) 

Order matches the canonical route. Flags: [PRET]=mechanism seen in disassembly; [WEB]=agent post-mortems.

1. **Intro / New Bark** - cannot leave until Mom/Elm scenes done; first rival steal event. Low risk.
2. **Cherrygrove -> Route 30/31, Mr. Pokémon, Mystery Egg**: must return to Elm; Elm's call interrupt; then Elm gives Mystery Egg handling (Egg to Elm after Violet). Standard.
3. **Violet Gym / Sprout Tower** (multi-floor, swaying pillar: trainers inside; pathfinding fails on segmented maps [WEB]). Falkner leads with flying; Chikorita is weak.
4. **Union Cave** (Route 32 -> Route 33): 1F/B1F/B2F; strength boulders and water on B1F are *optional*; the Lapras (Fri) on B2F is optional. Wild Zubat/Geodude/Onix can drain; encounter rate high; main danger is attrition without healing. [partially PRET; whether Flash is needed: no, cave is lit [UNVERIFIED]].
5. **Slowpoke Well** (Azalea): Rocket grunts; EVENT_CLEARED_SLOWPOKE_WELL; Kurt. Boulders optional.
6. **Ilex Forest**: maze; Farfetch'd herding mini-game (`EVENT_HERDED_FARFETCHD`), cut-tree puzzle; Charcoal master gives HM01. **Historically the #1 LLM/RL stall** (Farfetch'd flees). Do not leave without Cut.
7. **Goldenrod City**: Whitney (Miltank Rollout; Attract) - big difficulty spike, plan type/status counter or level 25+ Fighting/Ghost-immune... Radio Tower and Underground exist from the start; **Radio Tower becomes Rocket-occupied after the 7th badge** (B4). Dept Store mazes; Bike voucher; **Squirtbottle** from Floria (flower shop) requires Plain Badge and `EVENT_FOUGHT_SUDOWOODO` flow (`GoldenrodFlowerShop.asm`).
8. **Route 35/36 -> Sudowoodo** blocking Route 36: use Squirtbottle; `EVENT_FOUGHT_SUDOWOODO`. (Rock Smash flower rocks also on Route 36.)
9. **Ecruteak**: Burned Tower (rival fight, Eusine, release of roaming beasts), Dance Theater (five Kimono Girls -> Surf, then Morty's Gym with Ghost trap-maze and Normal-immune Ghosts). Morty's gym maze needs correct path (pits? it is an invisible-floor maze [UNVERIFIED]).
10. **Route 38/39 -> Olivine**; Miltank farm; **Olivine Lighthouse** (multi-floor, pits on upper floors; Jasmine inside until the sick Ampharos is cured with SecretPotion from Cianwood pharmacy, `EVENT_GOT_SECRETPOTION_FROM_PHARMACY`). **Notorious LLM trap** [WEB].
11. **Surf required** from Olivine to Cianwood (Routes 40/41, whirlpools in water on Route 41 can be avoided by routing). Olivine port: the Strength cafe; **Strength obtained here**.
12. **Cianwood**: Chuck (gym needs Strength-free; boulders are part of fight [PRET: `CianwoodGym.asm` boulder objects, `CianwoodGymMovement_ChuckChucksBoulder`]), pharmacy (SecretPotion), Fly (Chuck's wife). Return to Olivine, beat Jasmine (Mineral Badge; badge #6 or so).
13. **Mahogany / Route 42 / Mt. Mortar**: Mt. Mortar is optional (Karate King, Waterfall/Surf/Strength). Route 42 to Mahogany is outdoors [UNVERIFIED].
14. **Lake of Rage / Red Gyarados**: forced shiny Magikarp (Red Scale); Lance fight; `EVENT_DECIDED_TO_HELP_LANCE` choice - **refusing sets `EVENT_REFUSED_TO_HELP_LANCE_AT_LAKE_OF_RAGE`** and affects flow (`LakeOfRage.asm`); help him.
15. **Rocket Hideout (Mahogany)**: B1F-B3F, password/security-camera puzzles, spinner-free but with Electrode trap B2F -> Lance gives Whirlpool; the Executive fight; `EVENT_CLEARED_ROCKET_HIDEOUT`.
16. **Mahogany Gym (Pryce)**: ice-floor sliding puzzle in the gym [UNVERIFIED]; Pryce is the 5th/6th badge depending on order.
17. **Goldenrod Radio Tower** (after 7 badges): floor-by-floor Rocket takeover, Executive fights (Executivem/f AI is `SMART|OPPORTUNIST...`), **Card Key from the Underground via the switch-room puzzle** (`GoldenrodUnderground*.asm`; the correct switch order hint is in a Rocket Grunt's post-battle text [WEB]), Basement Key, Director in the warehouse, Ariana/Archer fight at the top -> Clear Bell (`RadioTower5F.asm`).
18. **Route 44 / Ice Path** (sliding ice, boulders on B1F/B2F [PRET objects]) -> Blackthorn. **Waterfall HM is on Ice Path 1F** (grab it).
19. **Blackthorn Gym**: boulder/cave puzzle (`BlackthornGym2F.asm`, boulders), Clair's team (Dragonair x3 + Kingdra). Clair refuses the badge: after the fight she sends you to **Dragon's Den** (`ClairText_GoToDragonsDen`, `BlackthornGym1F.asm` L32-60). At Dragon Shrine a **5-question quiz** (`DragonShrine.asm`; wrong answers send you back) -> then you receive Rising Badge, a Dratini. Gym also gives TM24 (note: Clair "can give TM24 twice" bug [PRET]).
20. **Dragon's Den** (Surf): Tue/Thu rival appearance; Blackthorn Grandpa blocks Den entrance until Clair sends you.
21. **Victory Road Gate -> Victory Road -> Indigo Plateau**: 8 badges check; Victory Road is long with trainers; Rival Silver appears (`EVENT_RIVAL_VICTORY_ROAD`).
22. **Indigo Plateau**: heal at Pokémon Center, then E4 consecutively **without leaving** (Will -> Koga -> Bruno -> Karen -> Lance); on whiteout during the League you reset to the beginning of the League; **save before**. Prof. Oak/Hall of Fame sequence; credits; `EVENT_BEAT_ELITE_FOUR`; Elm then gives the S.S. Ticket.
23. **Kanto**: S.S. Aqua (weekday gate, B5) -> Vermilion (Lt. Surge: needs Cut for the gym trash cans? In GSC the electric-floor puzzle exists [UNVERIFIED]), Saffron (Magnet Train or foot via guard gates), Celadon (Erika), Fuchsia (Janine), Cerulean (Misty + Machine Part hunt), Pewter (Brock, Mt. Moon), Cinnabar (Blue), Seafoam/Blaine's gym. Power Plant: **Machine Part hidden item in Cerulean Gym**. Pallet Town Oak's Lab for Mt. Silver trigger. Kanto trainers L40-58.
24. **Route 26/27 -> Tohjo Falls -> Route 28 -> Mt. Silver**: Victory Road gate on the way; Route 27 trainers L30s-40s.
25. **Mt. Silver / Silver Cave** (Room 1-3): wild L40-50s (Gen 2 strongest wilds), Hall of Fame Pokémon (Larvitar, etc.), **Red** in Silver Cave Room 3 (`SilverCaveRoom3.asm`, object appears unless `EVENT_RED_IN_MT_SILVER` is set). Needs 16 badges for the entry flag, Waterfall + Surf for the inside route [UNVERIFIED].

Softlock / unrecoverable-state risks worth guarding [mix]
- **Soft-resets aren't free**: Crystal saves only where the player saves (no autosave) and the harness should keep checkpoints/chains before each boss; whiteout does not softlock but returns to last PC and costs money (`wMoney`) [UNVERIFIED amount].
- Losing HM users: HM moves can be forgotten; the Move Deleter is in Blackthorn (only place to delete HMs) [UNVERIFIED], so **never overwrite an HM move**; keep one dedicated HM slave with Cut/Surf/Strength/Fly/Whirlpool/Waterfall/Flash.
- **Selling/tossing key items** is blocked; but wasting Escape Ropes etc. is not a softlock.
- **PC box and party**: don't box the only HM user; the game checks `ENGINE_*` badge flags and the move in the party for field moves.
- **Pit/ledge traps**: Olivine Lighthouse 4F pits, Mt. Mortar, Dragon's Den ponds - recoverable via warps.
- **Whirlpool/Surf**: Surfing onto whirlpool tiles without Whirlpool pushes you around but doesn't softlock.
- **Game freeze bug**: switching against a Pokémon with max HP < 4 freezes the game (rare) [PRET].
- **Sudowoodo / Spinners**: fine.
- **Bug-Catching Contest**: leaving the National Park mid-contest ends it; avoid.
- **Daycare**: depositing the only strong mon loses time; "Pokémon deposited in the Day-Care might lose experience" bug [PRET].
- **Mt. Silver Red**: fight cannot be aborted; keep Revives, Full Restores, PP restores; party order matters because Red's first mon is Pikachu L81 (Speed/Thunder). Having only HM slaves and one carry is a viable strategy (the Gemini run) but takes hours.

### B7. Item, money and level-grinding needs

- **Money sinks that matter**: Poké Balls (Poké/Great), Potions (Super/Hyper/Max), Revives, Full Heals/Antidotes/Parlyz Heals, Escape Ropes, Repels; later Full Restores (Celadon/Kanto marts [UNVERIFIED]), PP restores for Red. Gym leader/E4 prize money = base reward x level (trainer base rewards in `attributes.asm`: gym leaders 25, E4 25) [PRET]. Rival 1 reward 15, Rival 2 25.
- **Reference level curve for "safe" progression** (derived from the rosters above; heuristic): Falkner L9 -> Bugsy L16 -> Whitney L20 -> Morty L25 -> Chuck L30 -> Jasmine L35 -> Pryce L31 -> Clair L40 -> E4 L40-47 / Lance L44-50 -> Kanto gyms L33-50 (Surge 40-46, Erika 41-46, Sabrina 46-48, Blaine 45-50, Blue 54-58) -> Red L73-81. An agent should hold its carry at ~ +3 to +5 over the next boss's top level; AI item healing (A4) makes outleveling matter more than coverage. Wild levels scale with area: Route 22/26/27 (L20s-40s), Mt. Silver L40-55 (grind there for Red).
- **Grinding sources**: E4 rematch is available after the first clear and Kanto trainers (rematches by phone), Mt. Silver wilds, Victory Road trainers (no repeat). Trainer battles don't respawn (except rematches). Exp share-like: Exp. Share exists in Crystal (held); Stat Exp is gained per KO and does **not** share via Exp Share [WEB Smogon].
- **Held items that matter**: Leftovers, Berry heals, type-boost items (Charcoal from the Azalea charcoal kiln, Mystic Water, Black Belt, etc.) giving about +10% in GSC [UNVERIFIED value and sources].
- **Money**: Dept Store, Game Corner coins (Voltorb flip doesn't exist in GSC; slot machines) are optional [UNVERIFIED]; use battles + Mom saving.
- **Elite Four prep**: bring ~20-30 Hyper Potions/Max Potions, Full Heal x10+, Revives x10, at least 5-10 PP restoring items for Lance/Red; Karen/Koga use status (Toxic, Confuse Ray, Hypnosis), Will uses Psychic, Bruno Fighting/Rock, Lance Dragon + Hyper Beam. Type notes [UNVERIFIED, standard type chart]: Ice hits Dragonite/Dragonair/Gyarados-line Dragons hard; Rock hits Charizard (4x) and Aerodactyl.

### B8. Structured milestone checklist (≈58 milestones)

Each row: ID | milestone | key prerequisite(s) | verification signal (RAM / flag). Verification via badge bytes or event flags from B2.

Johto early game
1. M01 | Leave player's house, get Pokégear, deliver Mom/Elm scene | none | `wMapGroup` ≠ 24 (New Bark house) [UNVERIFIED group id]
2. M02 | Receive starter (Cyndaquil/Totodile/Chikorita) from Elm's lab | M01 | `EVENT_GOT_A_POKEMON_FROM_ELM` (DA75.2), `wPartyCount ≥ 1`
3. M03 | Reach Cherrygrove; Mystic Water man; rival encounter | M02 | map = Cherrygrove; Rival event flags [UNVERIFIED]
4. M04 | Get Mystery Egg from Mr. Pokémon | M03 | `EVENT_GOT_MYSTERY_EGG_FROM_MR_POKEMON` (DA75.6)
5. M05 | Return to Elm (egg handed in; Elm call re: stolen Pokémon) | M04 | `EVENT_GAVE_MYSTERY_EGG_TO_ELM` [UNVERIFIED idx]
6. M06 | Violet City: clear Sprout Tower, get HM05 Flash | M05 | `EVENT_GOT_HM05_FLASH` (DA74.4)
7. M07 | Beat Falkner -> Zephyr Badge | M05 | D857 bit0; `EVENT_BEAT_FALKNER` (DB09.5)
8. M08 | Route 32 -> Union Cave -> Route 33 -> Azalea | M07 | map = Azalea
9. M09 | Slowpoke Well cleared | M08 | `EVENT_CLEARED_SLOWPOKE_WELL` (DA77.3)
10. M10 | Beat Bugsy -> Hive Badge | M08 | D857 bit1
11. M11 | Ilex Forest: herd Farfetch'd, get HM01 Cut | M10 | `EVENT_HERDED_FARFETCHD` (DA77.1), `EVENT_GOT_HM01_CUT` (DA74.0)
12. M12 | Route 34 -> Goldenrod; get Bicycle (via Voucher) | M11 | `EVENT_GOT_BICYCLE` (DA7D.3)
13. M13 | Beat Whitney -> Plain Badge | M12 | D857 bit2
14. M14 | Floria's Squirtbottle; clear Sudowoodo (Route 36) | M13 | `EVENT_GOT_SQUIRTBOTTLE` (DA7D.4), `EVENT_FOUGHT_SUDOWOODO` (DA77.2)
15. M15 | Ecruteak: Burned Tower; rival; Eusine event | M14 | Burned Tower flags [UNVERIFIED]
16. M16 | Beat all five Kimono Girls; get HM03 Surf | M15 | `EVENT_GOT_HM03_SURF` (DA74.2)
17. M17 | Beat Morty -> Fog Badge | M15 | D857 bit3
18. M18 | Route 38/39 to Olivine; Strength from Olivine Cafe | M17 | `EVENT_GOT_HM04_STRENGTH` (DA74.3)
19. M19 | Surf to Cianwood (Route 40/41) | M18 + Fog Badge | map = Cianwood
20. M20 | Beat Chuck -> Storm Badge; get HM02 Fly | M19 | D857 bit5, `EVENT_GOT_HM02_FLY` (DA74.1)
21. M21 | Get SecretPotion from Cianwood pharmacy | M19 | `EVENT_GOT_SECRETPOTION_FROM_PHARMACY` (DA76.3)
22. M22 | Olivine Lighthouse: cure Ampharos, beat Jasmine -> Mineral Badge | M21 | D857 bit4, `EVENT_BEAT_JASMINE` (DB0A.1)
23. M23 | Route 42 -> Mahogany Town | M17 | map = Mahogany [UNVERIFIED path details]
24. M24 | Lake of Rage: Red Gyarados + meet Lance (help) | M23 | `EVENT_DECIDED_TO_HELP_LANCE` set; Red Scale
25. M25 | Mahogany Rocket Hideout cleared; Lance gives HM06 Whirlpool | M24 | `EVENT_CLEARED_ROCKET_HIDEOUT` (DA76.2), `EVENT_GOT_HM06_WHIRLPOOL` (DA74.5)
26. M26 | Beat Pryce -> Glacier Badge | M25 | D857 bit6
27. M27 | **Reach 7 Johto badges triggers Radio Tower occupation** | M20/M22/M26 etc. | `ENGINE_ROCKETS_IN_RADIO_TOWER` set (engine flag) 
28. M28 | Goldenrod Underground: solve switch puzzle for Card Key; Basement Key | M27 | Card Key / Basement Key in `wKeyItems`
29. M29 | Clear Radio Tower; get Clear Bell | M28 | `EVENT_CLEARED_RADIO_TOWER` (DA76.1), `EVENT_GOT_CLEAR_BELL` (DA81.0)
30. M30 | Route 44 -> Ice Path; grab HM07 Waterfall | M29 (Blackthorn gate opens) | `EVENT_GOT_HM07_WATERFALL` (DB43.0)
31. M31 | Blackthorn: beat Clair (gym), receive "go to Dragon's Den" | M30 | `EVENT_BEAT_CLAIR` (DB0A.4) is set when she's beaten
32. M32 | Dragon's Den quiz -> Rising Badge | M31 + Surf | D857 bit7
33. M33 | Obtain 8th Johto badge in hand; 8 bits set | M32 | D857 == 0xFF
34. M34 | Victory Road Gate badge check passed | M33 | `wMapGroup/Number` = Victory Road
35. M35 | Victory Road traversed (Rival fight) | M34 | map = Indigo Plateau
36. M36 | Heal, save at Indigo Plateau | M35 | map = Indigo Plateau Pokécenter
37. M37 | Beat Will | M36 | `EVENT_BEAT_ELITE_4_WILL` (DB29.0)
38. M38 | Beat Koga | M37 | DB29.1
39. M39 | Beat Bruno | M38 | DB29.2
40. M40 | Beat Karen | M39 | DB29.3
41. M41 | Beat Lance (Champion) -> Hall of Fame | M40 | `EVENT_BEAT_CHAMPION_LANCE` (DB29.4), `EVENT_BEAT_ELITE_FOUR` (DA7A.4)

Kanto
42. M42 | Return to New Bark/Elm; receive S.S. Ticket | M41 | `EVENT_GOT_SS_TICKET_FROM_ELM` (DA76.4); item S_S_TICKET in key items
43. M43 | Board S.S. Aqua at Olivine (first trip any day; return trips Mon/Fri) -> Vermilion | M42 | map = Vermilion
44. M44 | Beat Lt. Surge -> Thunder Badge | M43 + Cut | D858 bit2, `EVENT_BEAT_LTSURGE` (DB0A.7)
45. M45 | Cerulean City: find Machine Part (hidden, Cerulean Gym); beat Misty -> Cascade | M44 | D858 bit1, MACHINE_PART in key items
46. M46 | Return Machine Part at the Power Plant (Route 10) -> power restored | M45 | `EVENT_RETURNED_MACHINE_PART` (DA8B.1), `EVENT_RESTORED_POWER_TO_KANTO` (DA8B.5)
47. M47 | Rail Pass from Copycat (Kanto) + Magnet Train unlocked | M46 | `EVENT_GOT_PASS_FROM_COPYCAT` (DA8C.1)
48. M48 | Saffron: beat Sabrina -> Marsh Badge | M47 (or foot via gates) | D858 bit5
49. M49 | Celadon: beat Erika -> Rainbow Badge (Cut) | M44 | D858 bit3
50. M50 | Fuchsia: beat Janine -> Soul Badge | M44 | D858 bit4
51. M51 | Pewter: beat Brock -> Boulder Badge (via Mt. Moon/Route 3 or Fly) | M44 | D858 bit0
52. M52 | Cinnabar via Surf; Blaine at Seafoam Islands gym -> Volcano Badge | M44 + Surf (+Strength?) | D858 bit6 [UNVERIFIED gym location]
53. M53 | Cinnabar: talk to Blue so he returns to Viridian | M52 | `EVENT_BLUE_IN_CINNABAR` cleared / Blue in gym
54. M54 | Viridian Gym: beat Blue -> Earth Badge | M53 | D858 bit7, `EVENT_BEAT_BLUE` (DB0B.4)
55. M55 | 16 badges: D857 == 0xFF and D858 == 0xFF | M54 | both bytes 0xFF
56. M56 | Oak's Lab (Pallet): "Mt. Silver opens" | M55 | `EVENT_OPENED_MT_SILVER` (DB5B.7)
57. M57 | Reach Mt. Silver via Route 26/27/Tohjo Falls/28; navigate Silver Cave to Room 3 (Surf/Waterfall) | M56 + Waterfall/Surf | map = Silver Cave Room 3 (`SILVER_CAVE_ROOM_3`)
58. M58 | **Beat Red** -> game complete | M57 + Revives/Full Restores; see A5 roster | trainer class RED defeated; `wBattleMode` returns to 0 with Red removed (`EVENT_RED_IN_MT_SILVER` is Red's visibility flag) [post-battle flag meaning UNVERIFIED]

Optional-but-useful checkpoints (not on the completion path)
- Bug-Catching Contest (Tue/Thu/Sat), Dragon's Den Dratini, Kurt's Apricorn balls, Whirl Islands (Whirlpool+Waterfall; Lugia needs Silver Wing), Tin Tower (Clear Bell; Rainbow Wing -> Ho-Oh; Eusine/Suicune), Pokémon trades, Battle Tower, Safari Zone (Crystal has none; GSC has [UNVERIFIED]), Roaming legendaries (Raikou/Entei/Suicune roam from Burned Tower).

### B9. Autonomy agent notes (design implications)

- Prefer **RAM/flag-driven goal selection** over screen reasoning: compute next milestone from badge bytes + selected event flags; this repo already has `src/story_flags.py` but **its HM and Jasmine/Chuck indices disagree with pret**; fix or rely on badges.
- Handle **forced interruptions**: phone calls, rival ambushes, Mom money, "ask for phone number" prompts, evolution/learn-move prompts, level-up move replacement (never overwrite an HM move).
- **Time**: weekday-dependent steps (S.S. Aqua), Bug Contest, Move Tutor, shop hours. RTC control in the emulator is the largest unknown; verify with `wCurDay`.
- **Pathfinding**: segmented maps (Sprout Tower), NPC-blocked corridors, pits (Olivine Lighthouse 4F), sliding ice (Ice Path), boulders; use RAM collision/map data where possible rather than vision.
- **Battle**: pre-load enemy parties from pret; use 1-2 ply damage-calculator search; keep a Revive/Full Restore reserve for Lance/Blue/Red; gym leaders' item-use is deterministic-ish (top-level mon only).
- **Cheap wins**: Fly (Cianwood) early; Repels vs grind; sell nothing key; buy Revives; keep Nuggets/Big Pearls for cash [UNVERIFIED].

---------------------------------------------------------------------

## Sources

Disassembly [PRET] - https://github.com/pret/pokecrystal (files read: `data/trainers/attributes.asm`, `parties.asm`, `dvs.asm`, `data/types/{type_matchups,badge_type_boosts}.asm`, `data/battle/critical_hit_chances.asm`, `engine/battle/ai/{scoring,items,switch}.asm`, `engine/events/overworld.asm`, `engine/events/std_scripts.asm`, `constants/{event_flags,engine_flags,trainer_data_constants,script_constants}.asm`, `ram/wram.asm`, `docs/bugs_and_glitches.md`, and maps: IlexForest, SproutTower3F, DanceTheater, OlivineCafe, CianwoodCity, TeamRocketBaseB2F, IcePath1F, OlivineGym, BlackthornGym1F, DragonShrine, RadioTower5F, EcruteakTinTowerEntrance, MahoganyTown, LakeOfRage, VictoryRoadGate, OaksLab, ElmsLab, OlivinePort, VermilionPort, CinnabarIsland, CeruleanGym, PowerPlant, SilverCaveRoom3, Route35NationalParkGate, weekday maps).
- pret wiki "Improve the enemy trainer AI": https://github-wiki-see.page/m/pret/pokecrystal/wiki/Improve-the-enemy-trainer-AI
- Metamon: https://github.com/UT-Austin-RPL/metamon ; paper https://arxiv.org/abs/2504.04395 ; https://arxiv.org/html/2504.04395v2
- PokéChamp: https://arxiv.org/abs/2503.04094 ; https://proceedings.mlr.press/v267/karten25a.html
- PokéLLMon: https://arxiv.org/abs/2402.01118
- Foul Play: https://github.com/pmariglia/foul-play ; poke-engine: https://github.com/pmariglia/poke-engine
- PokéAgent Challenge: https://arxiv.org/abs/2603.15563 ; https://pokeagentchallenge.com
- poke-env: https://github.com/hsahovic/poke-env ; Smogon damage-calc: https://github.com/smogon/damage-calc ; Showdown gen2 mod: https://github.com/smogon/pokemon-showdown/tree/master/data/mods/gen2
- Smogon GSC mechanics: https://www.smogon.com/forums/threads/gsc-mechanics.3542417/ ; PS teambuilder guide https://www.smogon.com/player/issue11/past-gens-teambuilder
- Datacrystal Crystal RAM map: https://datacrystal.tcrf.net/wiki/Pok%C3%A9mon_Crystal/RAM_map
- LLM runs: https://blog.jcz.dev/gemini-3-pro-vs-25-pro-in-pokemon-crystal ; https://www.lesswrong.com/posts/u6Lacc7wx4yYkBQ3r/insights-into-claude-opus-4-5-from-pokemon
- Guides (cross-checked, partly unreliable): https://nationaldex.io/guide/pokemon-gold-silver-crystal/walkthrough

Known gaps in this research
- Bulbapedia/StrategyWiki (HTTP 403) not usable; many per-dungeon routing details above rely on pret scripts + memory and are marked [UNVERIFIED].
- Exact Gen 2 damage-formula ordering, stat-boost badge mapping, Ice Path/Silver Cave mandatory-HM requirements, whiteout money penalty, RTC behaviour in PyBoy not verified.
- Event-flag addresses are *computed*, not read from a live save.
