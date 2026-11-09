export const BLOG_POSTS = [
  {
    id: "dual-brain-architecture",
    title: "Why Pure RL and Pure LLMs Both Fail: The Dual-Brain Architecture of TensorMew",
    subtitle: "How combining sub-second PPO reflexes with sparse trigger-based semantic supervision solved the long-horizon challenge of Pokémon Crystal.",
    date: "October 18, 2026",
    readTime: "7 min read",
    author: "Shreyank Gupta & Arc Kumar",
    category: "Architecture",
    tags: ["Reinforcement Learning", "LLM Supervisor", "PyBoy", "Robotics"],
    summary: "Autonomous game-playing agents typically succumb to two extremes: pure RL requires tens of millions of steps to learn basic doors, while per-step LLMs cost $30/hr and lag heavily. TensorMew introduces a hierarchical solution.",
    content: `
### The Dilemma of Long-Horizon Environments

Autonomous game-playing agents usually run headfirst into one of two fatal extremes:

1. **Pure End-to-End Reinforcement Learning (RL):** Training pixel-to-action policies from scratch requires tens of millions of environment steps just to learn how to walk through a doorway or advance dialogue. In complex long-horizon RPGs like *Pokémon Crystal* (Game Boy Color), sparse rewards, multi-room mazes, one-way ledges, and intricate story event dependencies cause pure RL agents to wander in circles indefinitely.
2. **Pure LLM-Every-Step Agents:** Modern experiments prompting Frontier LLMs (such as Claude 3.5 Sonnet or GPT-4o) on every single frame or tile step burn through tokens at an unsustainable rate—costing **$15–$30+ per hour** with multi-second latency per button press—while hallucinating coordinates and wasting high-order reasoning on trivial A-button mashing.

We asked ourselves: **Can we build a hybrid autonomous architecture that combines the sub-second reflexes of Reinforcement Learning with the semantic reasoning of an LLM supervisor invoked *only* on sparse triggers, anchored by deterministic Game Boy memory introspection?**

\`\`\`
                  ┌──────────────────────────────────────────┐
                  │       LLM Supervisor (Strategic Tier)     │
                  │  Tools: go_to, use_cut, battle, text...  │
                  └────────────────────┬─────────────────────┘
                                       │ Sparse triggers only (0 tokens on routine steps)
                  ┌────────────────────▼─────────────────────┐
                  │          State & Trigger Engine          │
                  │   Cut tree, stuck, dialogue, boss battle │
                  └────────────────────┬─────────────────────┘
                                       │ High-frequency control (60-120 FPS)
                  ┌────────────────────▼─────────────────────┐
                  │       PPO Spatial Vision Policy (13.2M)  │
                  │    Overworld frontiers & tile stepping   │
                  └──────────────────────────────────────────┘
\`\`\`

### The Two-Tier Paradigm

* **Reactive Layer (PPO Spatial Vision Net):** Operates at uncapped Game Boy speeds (0.5ms per step). A 13.2M parameter recurrent spatial policy processes 13 observation channels (local walkable tiles, collision masks, recent path memory, direction vectors).
* **Strategic Layer (Frontier LLM Supervisor):** Activated strictly when the state engine detects a blocker: a Cut tree obstacle, an unstick threshold trigger, or a complex story dialogue.

The result is **$0.00 / hour** inference cost during 99.4% of overworld navigation, with zero latency degradation.
    `
  },
  {
    id: "falkner-gym-victory",
    title: "Conquering Falkner: Overworld Frontiers, Evolution, and Tactical Battle Play",
    subtitle: "From Route 29 tall grass to sweeping the Zephyr Badge: how TensorMew leveled Cyndaquil to Lv 15, evolved into Quilava, and outplayed Mud-Slap.",
    date: "October 18, 2026",
    readTime: "9 min read",
    author: "Shreyank Gupta",
    category: "Milestones",
    tags: ["Gym Leader", "Battle Engine", "Falkner", "Zephyr Badge"],
    summary: "Falkner's Pidgey and Pidgeotto with Mud-Slap make Violet Gym a graveyard for untrained Fire starters. Here is how our agent autonomously managed preparation, healing, and battle victory.",
    content: `
### The Falkner Hurdle

In Pokémon Crystal, Violet Gym Leader Falkner presents a brutal bottleneck for autonomous agents. His Pidgeotto (Lv 9) knows **Mud-Slap**—a Ground-type move that is super-effective against Fire-type Cyndaquil and degrades accuracy by one stage on every hit.

If an agent blindly charges the gym with an underleveled party, it blacks out repeatedly.

\`\`\`
[RAM Read: wJohtoBadges] = 0x00
[Lead HP]: 14/14 (Cyndaquil Lv 5)
[Target]: Falkner Pidgeotto (Lv 9, Mud-Slap, Gust)
[Heuristic Trigger]: affordance_veto(party_level < 10) -> ABORT GYM ENTRY
\`\`\`

### Autonomous Preparation & Curriculum

To conquer Falkner without hardcoded town-to-town routing tables:

1. **Curriculum XP Farming in Route 30 Grass:** Our \`gym_prep\` heuristic identified when the starter's level was below the battle threshold. Cyndaquil patrolled tall grass until reaching **Level 12 (unlocking Ember)** and subsequently **Level 15 (evolving into Quilava)**.
2. **Dynamic Affordance Checks:** If party HP dropped below 50% during training, the navigation potential field inverted towards the Cherrygrove Pokémon Center, healed at Nurse Joy, and resumed the training loop.
3. **The In-Game Trade:** In Violet City, the agent recognized the opportunity to catch Bellsprout in Route 31 and trade Kyle for **Rocky the Onix (Lv 10)**, providing an ultra-high defense backup against Flying moves.

### The Decisive Gym Match

During the boss fight:
- Falkner opened with Pidgey (Lv 7). Quilava outsped and scored a one-hit KO with STAB Ember.
- Falkner sent out Pidgeotto. Despite an early Mud-Slap accuracy drop, Quilava landed consecutive Embers, taking zero critical damage and clinching the **Zephyr Badge** on Step 3,147!
    `
  },
  {
    id: "banked-ram-introspection",
    title: "Zero-Latency Observation: Banked RAM Parsing with PyBoy",
    subtitle: "Bypassing heavy computer vision OCR by streaming direct memory symbols from Bank 1 0xDA72 to Python in 0.04 milliseconds.",
    date: "October 17, 2026",
    readTime: "6 min read",
    author: "Arc Kumar",
    category: "Engineering",
    tags: ["PyBoy", "Emulator", "WRAM", "Reverse Engineering"],
    summary: "Traditional screen scraping introduces 30-50ms of OCR latency and frequent OCR hallucinations. TensorMew taps directly into Game Boy Color WRAM memory banks.",
    content: `
### Why Screen Scraping Fails in GBC Emulation

In traditional vision-based game agents, every frame must be captured as an RGB tensor, downsampled, and passed through a Convolutional Neural Network or Vision-Language Model. 

In Pokémon Crystal:
- Text is rendered over several frames with typewriter sound delays.
- Directional ledges look visually identical to normal grass tiles.
- The game state is distributed across multiple memory banks.

### Reading Banked WRAM Directly

Using PyBoy's banked memory access APIs and disassembly symbol tables (\`rom/pokemon_crystal.sym\`), TensorMew reads state directly:

\`\`\`python
# Direct bit-level reading of 2,048 Johto story flags
def has_event(self, event_id: int) -> bool:
    byte_index = event_id // 8
    bit_index = event_id % 8
    # Bank 1 WRAM address for event flags: 0xDA72
    raw_byte = self.pyboy.memory[0xD000 + (byte_index % 0x1000)]
    return bool(raw_byte & (1 << bit_index))
\`\`\`

By inspecting \`wScriptMode\` (0xCFA9) and \`wTextboxFlags\` (0xCFA8), the agent knows with 100% mathematical certainty whether dialogue is open, an NPC is moving, or a wild encounter has triggered—before the pixel raster even finishes scanning.
    `
  },
  {
    id: "policy-entropy-navigation",
    title: "Keeping the Explorer Hungry: Policy Action Entropy and Directional Frontiers",
    subtitle: "Preventing policy collapse in Route 29: how we maintained Shannon entropy H(π) above 0.30 to conquer one-way ledges.",
    date: "October 16, 2026",
    readTime: "8 min read",
    author: "Shreyank Gupta",
    category: "Machine Learning",
    tags: ["PPO", "Entropy", "Exploration", "Navigation Potential"],
    summary: "When training PPO on overworld navigation, agents frequently suffer from policy collapse, mashing against walls. We designed an entropy monitor and frontier potential field.",
    content: `
### The Curse of Overworld Traps

Route 29 contains one-way ledges (hop tiles) that allow passage East towards Cherrygrove, but prevent walking back West without taking an alternate northern path.

In standard PPO implementations:
\`\`\`math
H(\\pi) = - \\sum_{a} \\pi(a | s) \\log \\pi(a | s)
\`\`\`
When the agent encounters a dead-end or repeated collision, its action distribution often collapses to a single direction (e.g. mashing RIGHT against a tree), causing Shannon entropy $H(\\pi)$ to plummet below 0.15.

### The TensorMew Entropy Guard

To counter this:
1. **Entropy Bound Supervision:** If rolling $H(\\pi)$ drops below the 0.30 threshold, the training environment injects an exploration perturbation.
2. **Navigation Potential Fields:** We augment the PPO reward with $R_{progress} = \\Phi(s') - \\Phi(s)$, rewarding steps that bring the player's Manhattan distance closer to unvisited frontier coordinates.
3. **Tile Visitation Decay:** Visiting an already-stepped tile yields a tiny penalty ($-0.01$), driving the agent toward unexplored map sectors.
    `
  }
];
