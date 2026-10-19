export const SCREENSHOTS_DATA = [
  {
    id: 1,
    title: "New Bark Town Start",
    file: "01_start.png",
    step: 0,
    lead: "Cyndaquil Lv.5",
    location: "New Bark Town (24, 4)",
    desc: "Initial boot into Johto. The agent initializes RAM banks, registers collision boundaries, and begins frontier overworld exploration from the player's bedroom.",
    badge: "Genesis"
  },
  {
    id: 2,
    title: "First Partner: Cyndaquil",
    file: "02_starter_party.png",
    step: 0,
    lead: "Cyndaquil Lv.5",
    location: "Elm's Laboratory",
    desc: "Prof. Elm hands over the Fire-type starter Cyndaquil (Lv 5, Tackle & Leer). Party state initialized in WRAM.",
    badge: "Party Initialized"
  },
  {
    id: 3,
    title: "First Wild Encounter & Catch",
    file: "03_first_catch.png",
    step: 54,
    lead: "Cyndaquil Lv.5",
    location: "Route 29 Tall Grass",
    desc: "Agent steps off the main road onto tall grass tiles, encounters a wild Pidgey, and executes its first automated battle routine.",
    badge: "Wild Catch"
  },
  {
    id: 4,
    title: "Arrival at Cherrygrove City",
    file: "04_cherrygrove.png",
    step: 620,
    lead: "Cyndaquil Lv.5",
    location: "Cherrygrove City (26, 1)",
    desc: "Conquered the directional ledge maze on Route 29. Discovered new map warp coordinate, triggering +10.0 map discovery reward.",
    badge: "New Map"
  },
  {
    id: 5,
    title: "Entering Pokémon Center",
    file: "05_pokecenter.png",
    step: 639,
    lead: "Cyndaquil Lv.5",
    location: "Cherrygrove PokéCenter",
    desc: "Autonomous affordance check recognizes damaged party. Navigates through the sliding doors to Joy's counter.",
    badge: "Affordance"
  },
  {
    id: 6,
    title: "Nurse Joy Party Healing",
    file: "06_healed.png",
    step: 646,
    lead: "Cyndaquil Lv.5",
    location: "Cherrygrove PokéCenter",
    desc: "Interacts with Nurse Joy. Party restored to full HP and PP. Respawn checkpoint saved to RAM.",
    badge: "Party Restored"
  },
  {
    id: 7,
    title: "Mr. Pokémon's House",
    file: "07_mr_pokemon.png",
    step: 794,
    lead: "Cyndaquil Lv.6",
    location: "Route 30 (North)",
    desc: "Traversed Route 30's split paths. Arrives at Mr. Pokémon's secluded house for the primary quest objective.",
    badge: "Quest Milestone"
  },
  {
    id: 8,
    title: "Received the Mystery Egg",
    file: "08_got_egg.png",
    step: 796,
    lead: "Cyndaquil Lv.6",
    location: "Mr. Pokémon's House",
    desc: "Acquired the Mystery Egg and Oak's Pokédex! Story bit flag EVENT_GOT_MYSTERY_EGG set in Bank 1 0xDA72.",
    badge: "Key Item"
  },
  {
    id: 9,
    title: "Rival Battle Commences",
    file: "08_rival_fight.png",
    step: 929,
    lead: "Cyndaquil Lv.6",
    location: "Cherrygrove Outskirts",
    desc: "Ambushed by the red-haired Rival holding Totodile. Battle engine activates tactical state evaluation.",
    badge: "Rival Encounter"
  },
  {
    id: 10,
    title: "Rival Defeated!",
    file: "10_rival_beaten.png",
    step: 943,
    lead: "Cyndaquil Lv.6",
    location: "Cherrygrove Outskirts",
    desc: "Exploited turn economy and Leer stat drops. Rival drops prize money and retreats east.",
    badge: "Victory"
  },
  {
    id: 11,
    title: "Delivered Egg to Prof. Elm",
    file: "11_gave_egg.png",
    step: 1088,
    lead: "Cyndaquil Lv.6",
    location: "New Bark Town Lab",
    desc: "Returned to New Bark Town. Elm accepts the egg, hands over 5 Poké Balls, unlocking official Gym qualification.",
    badge: "Gym Qualified"
  },
  {
    id: 12,
    title: "Curriculum XP Farming",
    file: "12_farming_xp.png",
    step: 1133,
    lead: "Cyndaquil Lv.6",
    location: "Route 30 Tall Grass",
    desc: "gym_prep routine activates heuristic level-grinding in Route 30 grass to prepare for Violet Gym's Mud-Slap birds.",
    badge: "Curriculum"
  },
  {
    id: 13,
    title: "Cyndaquil Learns Ember (Lv 12)",
    file: "13_lead_lv12.png",
    step: 2011,
    lead: "Cyndaquil Lv.12",
    location: "Route 30",
    desc: "Major power spike! Cyndaquil unlocks Ember (STAB 40 BP Fire move), drastically shortening wild encounter turns.",
    badge: "Move Unlocked"
  },
  {
    id: 14,
    title: "Evolution & Level 15 Milestone",
    file: "14_lead_lv15.png",
    step: 2645,
    lead: "Quilava Lv.15",
    location: "Route 31",
    desc: "Target level reached! Starter evolves into Quilava with boosted speed and special attack, ready to face Falkner.",
    badge: "Evolution"
  },
  {
    id: 15,
    title: "Talked to Mom / Savings Bank",
    file: "15_talked_mom.png",
    step: 2754,
    lead: "Quilava Lv.15",
    location: "New Bark Town Home",
    desc: "Interacts with Mom to setup autonomous prize-money savings and emergency potion funds.",
    badge: "Economy"
  },
  {
    id: 16,
    title: "Arrival at Violet City",
    file: "16_violet_city.png",
    step: 2820,
    lead: "Quilava Lv.15",
    location: "Violet City (3, 1)",
    desc: "Crossed Route 31 gate into Violet City. Home of Sprout Tower, the Earl's Academy, and Falkner's Flying Gym.",
    badge: "Historic City"
  },
  {
    id: 17,
    title: "Entering Violet Gym",
    file: "17_gym_entered.png",
    step: 2868,
    lead: "Quilava Lv.15",
    location: "Violet Gym Interior",
    desc: "Steps onto the wood catwalk above the abyss. Prepares to face Bird Keepers Abe and Rod.",
    badge: "Gym Infiltration"
  },
  {
    id: 18,
    title: "Gym Leader Falkner Battle",
    file: "18_falkner_fight.png",
    step: 2875,
    lead: "Quilava Lv.15",
    location: "Violet Gym Arena",
    desc: "Falkner sends out Pidgey (Lv 7) followed by ace Pidgeotto (Lv 9) with high-damage Gust and accuracy-reducing Mud-Slap.",
    badge: "Boss Battle"
  },
  {
    id: 19,
    title: "Falkner Swept & Defeated!",
    file: "19_falkner_beaten.png",
    step: 3147,
    lead: "Quilava Lv.16",
    location: "Violet Gym Arena",
    desc: "Quilava weathers Mud-Slap sandstorms and lands consecutive critical Embers. Falkner concedes defeat!",
    badge: "Falkner Beaten"
  },
  {
    id: 20,
    title: "Trainer Card Updated",
    file: "20_trainer_card.png",
    step: 3152,
    lead: "Quilava Lv.16",
    location: "Violet Gym",
    desc: "Trainer Card inspect reveals 1st Gym Badge slot filled with the polished Johto Zephyr Badge.",
    badge: "Trainer Status"
  },
  {
    id: 21,
    title: "Zephyr Badge Officially Conquered!",
    file: "21_badge.png",
    step: 3152,
    lead: "Quilava Lv.16",
    location: "Violet Gym",
    desc: "wJohtoBadges RAM byte updated to 0x01! Falkner presents TM31 (Mud-Slap) and unlocks Flash outside battle.",
    badge: "Zephyr Badge Earned"
  }
];
