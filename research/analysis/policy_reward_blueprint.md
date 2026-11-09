# Policy and reward blueprint (plain-text extract)

Extracted from `docs/policy_reward_blueprint.html`, the design from the earlier research session.
Each table row reads: signal, when it fires, value, status (new / exists / replace / remove / change).

Policy and Reward Blueprint 
 Policy and Reward Blueprint 
 How the agent would play the early game end to end: who presses the buttons in each situation, and which rewards teach the one part that learns. Values come from src/rl_env.py ; anything not in the code today is marked new. 
 Who acts 
 Stage manager 
 event flags pick the goal 
 HP under 50% swaps in the Pokécenter 
 goal, next step, stage 
 Game 
 PyBoy + RAM 
 Mode? 
 from RAM flags 
 Walking: navigation policy 
 learned · PPO · up, down, left, right, A 
 Battle: tactical controller 
 scripted · best move, item guard, switching 
 Dialogue: text mash 
 scripted · advance text until it closes 
 Buttons 
 press, tick frames 
 next frame, new state 
 state before and after 
 Reward builder 
 dense route + sparse events 
 reward 
 PPO update 
 changes the walking policy only 
 Only the walking policy learns. Battle and dialogue are scripted, so battle, healing and talking rewards reach the policy as credit for the walking choices that led there: entering grass at low HP, reaching the Pokécenter, standing in front of the right NPC. 
 What the walking policy sees 
 5×5 local grid and blocked-tile mask, last action, HP fraction, party size and top level, battle and dialogue flags. New: stage one-hot (or the event flags), the first step of the shortest path to the current goal, and remaining distance to that goal. 
 Why the stage must be in the input 
 Route 30 is walked north to Mr. Pokémon, south with the egg, and north again to Violet. Without the stage, one tile asks for three different directions. 
 Reward sizes 
 Walking to the gym is worth about 350 points today because the hand-coded stage potentials in story_potential climb from 0 to roughly 350. Beating Falkner is worth 50. The agent is paid far more for the commute than for the goal. Scaling route progress to 60 points keeps the dense signal as a guide and lets the milestones decide what matters. 
 Every signal 
 Signal Fires when Value Status 
 Navigation (dense) 
 Route progress Each step along the shortest path to the current goal. Potential is steps remaining on the whole route, so it is continuous across goals. 0.08 × change in Φ ≈ 60 to Falkner new 
 Step cost Every step −0.01 exists 
 Bump Move blocked, outside battle and dialogue −0.05 × min(4, bumps in a row) exists 
 Per-map waypoints First visit to each map on the way to the gym or back with the egg 3 to 20, +15 per map on the egg return remove 
 Hand-coded stage potentials Manhattan distance to hand-placed coordinates sum ≈ 350 replace 
 Story milestones (sparse, once) 
 Starter, egg, rival, egg delivered Event flag flips 15, 20, 15, 20 exists 
 Mom talk, catch tutorial, TM31, Togepi egg Event flag flips 10, 5, 10, 10 exists 
 Falkner beaten, badge, Cut Event flag or badge count rises 50, 30, 20 exists 
 Battling (scripted fighting, learned avoidance) 
 Trainer battle won Battle ends with the lead still up +5 exists 
 Wild battle won Same, by lead level and quest stage +3 under Lv 7, 0 during the egg errand, +0.2 after exists 
 Level up Party level sum rises; full credit to 15, a quarter after 5 × change exists 
 Catch Party gains a species; repeats halve 10, 5, 2.5, floor 0.5 exists 
 Damage taken Party HP drops −2 × fraction of max HP lost exists 
 Lead below 25% HP Every step −0.05 exists 
 White-out Lead faints −25 exists 
 Train to target level Gym prep: walk to a grass patch until the lead reaches the target goal override, no new reward new 
 Healing 
 Pokécenter heal HP rises inside a Pokécenter, once per center per episode 15 + 10 × fraction healed exists 
 Talk to Nurse Joy Dialogue opens while hurt in a Pokécenter +2 exists 
 Heal detour HP under 50%: the goal becomes the Pokécenter door, then the route resumes goal override, same potential new 
 Talking 
 New interaction First A press that opens text from a tile and facing +0.5 exists 
 Goal conversation The goal tile is the one in front of the NPC; the flag flip pays the milestone via milestones change 
 Exploration 
 New tile First visit this episode +1 (+2 on objective maps) → 0.05 with a route graph change 
 Unvisited neighbor Step toward a tile never seen +0.2 → 0 with a route graph change 
 New map, new warp First time per episode +10, +1 exists 
 Frontier shaping Closer to the nearest unvisited tile; used where no route graph exists 4 × (γΦ' − Φ) exists 
 Count-based curiosity Fallback for unmapped areas 0.08 scale exists 
 Episode rules 
 Stepping back into a finished map Ends the episode −2 make stage-aware 
 Truncation 64 steps per episode; resets come from saved states, 30% from the frontier list bootstrap the value, balance start stages change 
 With a route graph, exploration pays almost nothing so it cannot outbid the route. Where the graph has no map yet (later Johto), the full exploration terms come back automatically. 
 Noise and exploration 
 Knob Today Proposed 
 Entropy bonus 0.03, fixed Anneal to 0.005 as route progress per episode rises, so the policy commits once the path is clear. 
 Action mask blocked tiles removed from the softmax Keep. Also mask A while no dialogue target is in front of the player. 
 Start states 30% from the frontier list Sample stages evenly so the return leg is trained as often as the first. 
 Sticky actions none Repeat the last move with probability 0.1 in training; off at evaluation. 
 Start jitter none 0 to 6 random moves in safe maps after a reset. 
 Evaluation sampled Greedy, no noise, from the New Bark save. 
 Build order 
 Merge return_nav and falkner_nav into one route graph with warps, and compute Φ(stage, tile) . 
 Add stage, next-step direction and remaining distance to the observation. 
 Swap the hand-coded potentials for route progress at 0.08 per step, and delete the per-map waypoints. 
 Cut new-tile and neighbor bonuses when a route graph covers the map. 
 Make the finished-map penalty stage-aware so the egg return is allowed. 
 Test: Φ rises along the path, including across each goal; a full circle telescopes to the end value; the same tile gives different features at stage 1 and stage 3. 
 Train from stage-balanced start states with the entropy schedule, and compare steps-to-goal per leg against the scripted navigators.
