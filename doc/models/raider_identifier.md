# raider_identifier.py - Documentation

## 1. Overview
`models/raider_identifier.py` implements **Stage 2** of the pipeline. It distinguishes the *Raider* (the active attacker) from the *Defenders*. Unlike simple systems that might rely on jersey color, this module uses a sophisticated multi-cue logic based on movement patterns, speed, and positioning, making it robust to different jersey colors.

## 2. Technical Overview
- **Type**: Logic / Heuristic Analysis Module
- **Dependencies**: `numpy`, `collections.deque` (for history buffers).
- **Input**: List of player dictionaries (from Stage 1).
- **Output**: `raider_info` (Dictionary with `track_id`, `speed`, `confidence`).

## 3. Detailed Technical
The module employs a "**Permanent Lock**" strategy combined with a multi-cue evaluation window.
- **Evaluation Window**: For the first 90 frames (~3 seconds), it observes all players but does not commit. It accumulates history for position and movement.
- **4-Cue Algorithm**: It calculates a confidence score for each player based on:
    1.  **Directional Commitment (30%)**: Raiders typically move in a consistent direction (attacking), while defenders shuffle.
    2.  **Speed Dominance (30%)**: The raider is often the fastest moving player on the court.
    3.  **Direction Changes (20%)**: Raiders make sharp cuts/dodges.
    4.  *(Deprecated: Role Persistence)*.
- **Locking**: Once a player exceeds the confidence threshold (70%), they are permanently locked as the Raider. This prevents ID switching during the raid.

## 4. Workflow and Function Details

### Workflow
1.  **Accumulate**: Store position history for all players for 90 frames.
2.  **Evaluate**: compute metrics (speed, linearity, zig-zag) for everyone.
3.  **Identify**: Select the player with highest confidence score.
4.  **Lock**: Save that ID and ignore others for the rest of individual processing session.

### Functions

#### `RaiderIdentifier` Class

##### `identify_raider(self, players) -> int`
- **Purpose**: Main logic to find the raider ID.
- **Details**: 
    - If already locked, returns stored ID.
    - If window < 90, returns None (gathering data).
    - At frame 90: loops through all players, calls `calculate_raider_confidence`, selects winner.

##### `calculate_raider_confidence(self, track_id, players) -> float`
- **Purpose**: Computes score (0.0 to 1.0).
- **Details**: Sums up weighted sub-scores:
    - Calls `calculate_directional_commitment`.
    - Calls `calculate_speed`.
    - Calls `calculate_direction_changes`.

##### `calculate_speed(self, track_id)`
- **Purpose**: Computes average pixels/frame velocity.
- **Details**: Total distance traveled / 90 frames.

##### `calculate_directional_commitment(self, track_id)`
- **Purpose**: Measures "purposeful movement".
- **Details**: Analyzes vectors between frames. If a player maintains a similar vector angle for >10% of frames, they score high.

##### `check_midline_crossing(self, track_id)`
- **Purpose**: Court logic helper.
- **Details**: Checks if a player's y-coordinate crosses the defined midline threshold. (Useful as an additional filter, though primary logic relies on motion dynamics).
