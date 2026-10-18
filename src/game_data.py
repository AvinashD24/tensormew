"""Lookup tables and friendly name formatters for Pokémon Crystal.

Includes comprehensive mappings for Gen 2 Pokémon species, moves, and map locations.
"""

from __future__ import annotations

# Gen 2 Species Names (1..251)
SPECIES_NAMES: dict[int, str] = {
    1: "Bulbasaur", 2: "Ivysaur", 3: "Venusaur", 4: "Charmander", 5: "Charmeleon", 6: "Charizard",
    7: "Squirtle", 8: "Wartortle", 9: "Blastoise", 10: "Caterpie", 11: "Metapod", 12: "Butterfree",
    13: "Weedle", 14: "Kakuna", 15: "Beedrill", 16: "Pidgey", 17: "Pidgeotto", 18: "Pidgeot",
    19: "Rattata", 20: "Raticate", 21: "Spearow", 22: "Fearow", 23: "Ekans", 24: "Arbok",
    25: "Pikachu", 26: "Raichu", 27: "Sandshrew", 28: "Sandslash", 29: "Nidoran♀", 30: "Nidorina",
    31: "Nidoqueen", 32: "Nidoran♂", 33: "Nidorino", 34: "Nidoking", 35: "Clefairy", 36: "Clefable",
    37: "Vulpix", 38: "Ninetales", 39: "Jigglypuff", 40: "Wigglytuff", 41: "Zubat", 42: "Golbat",
    43: "Oddish", 44: "Gloom", 45: "Vileplume", 46: "Paras", 47: "Parasect", 48: "Venonat",
    49: "Venomoth", 50: "Diglett", 51: "Dugtrio", 52: "Meowth", 53: "Persian", 54: "Psyduck",
    55: "Golduck", 56: "Mankey", 57: "Primeape", 58: "Growlithe", 59: "Arcanine", 60: "Poliwag",
    61: "Poliwhirl", 62: "Poliwrath", 63: "Abra", 64: "Kadabra", 65: "Alakazam", 66: "Machop",
    67: "Machoke", 68: "Machamp", 69: "Bellsprout", 70: "Weepinbell", 71: "Victreebel", 72: "Tentacool",
    73: "Tentacruel", 74: "Geodude", 75: "Graveler", 76: "Golem", 77: "Ponyta", 78: "Rapidash",
    79: "Slowpoke", 80: "Slowbro", 81: "Magnemite", 82: "Magneton", 83: "Farfetch'd", 84: "Doduo",
    85: "Dodrio", 86: "Seel", 87: "Dewgong", 88: "Grimer", 89: "Muk", 90: "Shellder",
    91: "Cloyster", 92: "Gastly", 93: "Haunter", 94: "Gengar", 95: "Onix", 96: "Drowzee",
    97: "Hypno", 98: "Krabby", 99: "Kingler", 100: "Voltorb", 101: "Electrode", 102: "Exeggcute",
    103: "Exeggutor", 104: "Cubone", 105: "Marowak", 106: "Hitmonlee", 107: "Hitmonchan", 108: "Lickitung",
    109: "Koffing", 110: "Weezing", 111: "Rhyhorn", 112: "Rhydon", 113: "Chansey", 114: "Tangela",
    115: "Kangaskhan", 116: "Horsea", 117: "Seadra", 118: "Goldeen", 119: "Seaking", 120: "Staryu",
    121: "Starmie", 122: "Mr. Mime", 123: "Scyther", 124: "Jynx", 125: "Electabuzz", 126: "Magmar",
    127: "Pinsir", 128: "Tauros", 129: "Magikarp", 130: "Gyarados", 131: "Lapras", 132: "Ditto",
    133: "Eevee", 134: "Vaporeon", 135: "Jolteon", 136: "Flareon", 137: "Porygon", 138: "Omanyte",
    139: "Omastar", 140: "Kabuto", 141: "Kabutops", 142: "Aerodactyl", 143: "Snorlax", 144: "Articuno",
    145: "Zapdos", 146: "Moltres", 147: "Dratini", 148: "Dragonair", 149: "Dragonite", 150: "Mewtwo",
    151: "Mew", 152: "Chikorita", 153: "Bayleef", 154: "Meganium", 155: "Cyndaquil", 156: "Quilava",
    157: "Typhlosion", 158: "Totodile", 159: "Croconaw", 160: "Feraligatr", 161: "Sentret", 162: "Furret",
    163: "Hoothoot", 164: "Noctowl", 165: "Ledyba", 166: "Ledian", 167: "Spinarak", 168: "Ariados",
    169: "Crobat", 170: "Chinchou", 171: "Lanturn", 172: "Pichu", 173: "Cleffa", 174: "Igglybuff",
    175: "Togepi", 176: "Togetic", 177: "Natu", 178: "Xatu", 179: "Mareep", 180: "Flaaffy",
    181: "Ampharos", 182: "Bellossom", 183: "Marill", 184: "Azumarill", 185: "Sudowoodo", 186: "Politoed",
    187: "Hoppip", 188: "Skiploom", 189: "Jumpluff", 190: "Aipom", 191: "Sunkern", 192: "Sunflora",
    193: "Yanma", 194: "Wooper", 195: "Quagsire", 196: "Espeon", 197: "Umbreon", 198: "Murkrow",
    199: "Slowking", 200: "Misdreavus", 201: "Unown", 202: "Wobbuffet", 203: "Girafarig", 204: "Pineco",
    205: "Forretress", 206: "Dunsparce", 207: "Gligar", 208: "Steelix", 209: "Snubbull", 210: "Granbull",
    211: "Qwilfish", 212: "Scizor", 213: "Shuckle", 214: "Heracross", 215: "Sneasel", 216: "Teddiursa",
    217: "Ursaring", 218: "Slugma", 219: "Magcargo", 220: "Swinub", 221: "Piloswine", 222: "Corsola",
    223: "Remoraid", 224: "Octillery", 225: "Delibird", 226: "Mantine", 227: "Skarmory", 228: "Houndour",
    229: "Houndoom", 230: "Kingdra", 231: "Phanpy", 232: "Donphan", 233: "Porygon2", 234: "Stantler",
    235: "Smeargle", 236: "Tyrogue", 237: "Hitmontop", 238: "Smoochum", 239: "Elekid", 240: "Magby",
    241: "Miltank", 242: "Blissey", 243: "Raikou", 244: "Entei", 245: "Suicune", 246: "Larvitar",
    247: "Pupitar", 248: "Tyranitar", 249: "Lugia", 250: "Ho-Oh", 251: "Celebi",
}

# Gen 2 Move Names (1..251)
MOVE_NAMES: dict[int, str] = {
    1: "Pound", 2: "Karate Chop", 3: "Double Slap", 4: "Comet Punch", 5: "Mega Punch",
    6: "Pay Day", 7: "Fire Punch", 8: "Ice Punch", 9: "Thunder Punch", 10: "Scratch",
    11: "Vice Grip", 12: "Guillotine", 13: "Razor Wind", 14: "Swords Dance", 15: "Cut",
    16: "Gust", 17: "Wing Attack", 18: "Whirlwind", 19: "Fly", 20: "Bind",
    21: "Slam", 22: "Vine Whip", 23: "Stomp", 24: "Double Kick", 25: "Mega Kick",
    26: "Jump Kick", 27: "Rolling Kick", 28: "Sand Attack", 29: "Headbutt", 30: "Horn Attack",
    31: "Fury Attack", 32: "Horn Drill", 33: "Tackle", 34: "Body Slam", 35: "Wrap",
    36: "Take Down", 37: "Thrash", 38: "Double-Edge", 39: "Tail Whip", 40: "Poison Sting",
    41: "Twineedle", 42: "Pin Missile", 43: "Leer", 44: "Bite", 45: "Growl",
    46: "Roar", 47: "Sing", 48: "Supersonic", 49: "Sonic Boom", 50: "Disable",
    51: "Acid", 52: "Ember", 53: "Flamethrower", 54: "Mist", 55: "Water Gun",
    56: "Hydro Pump", 57: "Surf", 58: "Ice Beam", 59: "Blizzard", 60: "Psybeam",
    61: "Bubble Beam", 62: "Aurora Beam", 63: "Hyper Beam", 64: "Peck", 65: "Drill Peck",
    66: "Submission", 67: "Low Kick", 68: "Counter", 69: "Seismic Toss", 70: "Strength",
    71: "Absorb", 72: "Mega Drain", 73: "Leech Seed", 74: "Growth", 75: "Razor Leaf",
    76: "Solar Beam", 77: "Poison Powder", 78: "Stun Spore", 79: "Sleep Powder", 80: "Petal Dance",
    81: "String Shot", 82: "Dragon Rage", 83: "Fire Spin", 84: "Thunder Shock", 85: "Thunderbolt",
    86: "Thunder Wave", 87: "Thunder", 88: "Rock Throw", 89: "Earthquake", 90: "Fissure",
    91: "Dig", 92: "Toxic", 93: "Confusion", 94: "Psychic", 95: "Hypnosis",
    96: "Meditate", 97: "Agility", 98: "Quick Attack", 99: "Rage", 100: "Teleport",
    101: "Night Shade", 102: "Mimic", 103: "Screech", 104: "Double Team", 105: "Recover",
    106: "Harden", 107: "Minimize", 108: "Smokescreen", 109: "Confuse Ray", 110: "Withdraw",
    111: "Defense Curl", 112: "Barrier", 113: "Light Screen", 114: "Haze", 115: "Reflect",
    116: "Focus Energy", 117: "Bide", 118: "Metronome", 119: "Mirror Move", 120: "Self-Destruct",
    121: "Egg Bomb", 122: "Lick", 123: "Smog", 124: "Sludge", 125: "Bone Club",
    126: "Fire Blast", 127: "Waterfall", 128: "Clamp", 129: "Swift", 130: "Skull Bash",
    131: "Spike Cannon", 132: "Constrict", 133: "Amnesia", 134: "Kinesis", 135: "Soft-Boiled",
    136: "High Jump Kick", 137: "Glare", 138: "Dream Eater", 139: "Poison Gas", 140: "Barrage",
    141: "Leech Life", 142: "Lovely Kiss", 143: "Sky Attack", 144: "Transform", 145: "Bubble",
    146: "Dizzy Punch", 147: "Spore", 148: "Flash", 149: "Psywave", 150: "Splash",
    151: "Acid Armor", 152: "Crabhammer", 153: "Explosion", 154: "Fury Swipes", 155: "Bonemerang",
    156: "Rest", 157: "Rock Slide", 158: "Hyper Fang", 159: "Sharpen", 160: "Conversion",
    161: "Tri Attack", 162: "Super Fang", 163: "Slash", 164: "Substitute", 165: "Struggle",
    166: "Sketch", 167: "Triple Kick", 168: "Thief", 169: "Spider Web", 170: "Mind Reader",
    171: "Nightmare", 172: "Flame Wheel", 173: "Snore", 174: "Curse", 175: "Flail",
    176: "Conversion 2", 177: "Aeroblast", 178: "Cotton Spore", 179: "Reversal", 180: "Spite",
    181: "Powder Snow", 182: "Protect", 183: "Mach Punch", 184: "Scary Face", 185: "Feint Attack",
    186: "Sweet Kiss", 187: "Belly Drum", 188: "Sludge Bomb", 189: "Mud-Slap", 190: "Octazooka",
    191: "Spikes", 192: "Zap Cannon", 193: "Foresight", 194: "Destiny Bond", 195: "Perish Song",
    196: "Icy Wind", 197: "Detect", 198: "Bone Rush", 199: "Lock-On", 200: "Outrage",
    201: "Sandstorm", 202: "Giga Drain", 203: "Endure", 204: "Charm", 205: "Rollout",
    206: "False Swipe", 207: "Swagger", 208: "Milk Drink", 209: "Spark", 210: "Fury Cutter",
    211: "Steel Wing", 212: "Mean Look", 213: "Attract", 214: "Sleep Talk", 215: "Heal Bell",
    216: "Return", 217: "Present", 218: "Frustration", 219: "Safeguard", 220: "Pain Split",
    221: "Sacred Fire", 222: "Magnitude", 223: "Dynamic Punch", 224: "Megahorn", 225: "Dragon Breath",
    226: "Baton Pass", 227: "Encore", 228: "Pursuit", 229: "Rapid Spin", 230: "Sweet Scent",
    231: "Iron Tail", 232: "Metal Claw", 233: "Vital Throw", 234: "Morning Sun", 235: "Synthesis",
    236: "Moonlight", 237: "Hidden Power", 238: "Cross Chop", 239: "Twister", 240: "Rain Dance",
    241: "Sunny Day", 242: "Crunch", 243: "Mirror Coat", 244: "Psych Up", 245: "Extreme Speed",
    246: "Ancient Power", 247: "Shadow Ball", 248: "Future Sight", 249: "Rock Smash", 250: "Whirlpool",
    251: "Beat Up",
}

# Pokémon Crystal Map Group and Number to Friendly Location Name
MAP_NAMES: dict[tuple[int, int], str] = {
    # Group 24: New Bark & Route 29
    (24, 1): "Player's House 1F",
    (24, 2): "Player's House 2F",
    (24, 3): "Route 29",
    (24, 4): "New Bark Town",
    (24, 5): "Elm's Lab",
    (24, 6): "Route 29 Gate (Route 46)",
    (24, 7): "Route 29 Gate (Cherrygrove)",
    (24, 8): "Elm's House",
    (24, 9): "Neighbor's House",
    (24, 10): "Route 27",

    # Group 26: Cherrygrove, Route 30, Route 31
    (26, 1): "Route 30",
    (26, 2): "Route 31",
    (26, 3): "Cherrygrove City",
    (26, 4): "Cherrygrove Mart",
    (26, 5): "Cherrygrove Poké Center",
    (26, 6): "Cherrygrove Gym Speech House",
    (26, 7): "Guide Gent's House",
    (26, 8): "Cherrygrove Evolution House",
    (26, 9): "Route 30 Berry House",
    (26, 10): "Mr. Pokémon's House",
    (26, 11): "Route 31 Gate (Violet)",

    # Group 10: Violet City & Surrounds
    (10, 1): "Sprout Tower 1F",
    (10, 2): "Sprout Tower 2F",
    (10, 3): "Sprout Tower 3F",
    (10, 4): "Route 32",
    (10, 5): "Violet City",
    (10, 6): "Violet Nickname House",
    (10, 7): "Violet Gym (Falkner)",
    (10, 8): "Earl's Pokémon Academy",
    (10, 9): "Violet Mart",
    (10, 10): "Violet Poké Center",
    (10, 11): "Violet Kyle's House",
    (10, 12): "Route 32 Gate",
    (10, 13): "Route 32 Poké Center",

    # Common shared interiors
    (20, 1): "Poké Center 2F (Colosseum)",

    # Group 21: Azalea Town & Ilex Forest
    (21, 1): "Azalea Town",
    (21, 2): "Slowpoke Well",
    (21, 3): "Slowpoke Well B1F",
    (21, 4): "Azalea Gym (Bugsy)",
    (21, 5): "Charcoal Kiln",
    (21, 6): "Azalea Poké Center",
    (21, 7): "Azalea Mart",
    (21, 8): "Kurt's House",
    (21, 14): "Ilex Forest",

    # Group 22: Route 34 & Daycare
    (22, 1): "Route 34",
    (22, 2): "Day-Care House",
    (22, 3): "Route 34 Gate (Goldenrod)",

    # Group 23: Goldenrod City
    (23, 1): "Goldenrod City",
    (23, 2): "Goldenrod Gym (Whitney)",
    (23, 3): "Goldenrod Bike Shop",
    (23, 7): "Goldenrod Poké Center",
}


def get_species_name(species_id: int) -> str:
    """Return friendly name for Pokémon species ID, or 'Unknown #id'."""
    return SPECIES_NAMES.get(species_id, f"Pokémon #{species_id}")


def get_move_name(move_id: int) -> str:
    """Return friendly name for Move ID, or 'Move #id'."""
    return MOVE_NAMES.get(move_id, f"Move #{move_id}")


def get_map_name(map_group: int, map_number: int) -> str:
    """Return friendly location name for map group & number, or 'Map (g, n)'."""
    return MAP_NAMES.get((map_group, map_number), f"Map ({map_group}, {map_number})")


def get_party_moveset(party_moves: tuple[tuple[int, ...], ...] | list, slot: int = 0) -> list[str]:
    """Return list of move names for active Pokémon or specified party slot."""
    if not party_moves or slot >= len(party_moves):
        return []
    mon_moves = party_moves[slot]
    return [get_move_name(m) for m in mon_moves if m > 0]
