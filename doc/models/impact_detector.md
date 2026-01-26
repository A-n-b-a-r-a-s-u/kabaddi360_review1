# impact_detector.py - Documentation

## 1. Overview
`models/impact_detector.py` implements **Stage 6** of the pipeline. In Kabaddi, tackles are the main source of trauma. This module detects collisions between the Raider and Defenders. It models the multi-agent interaction, identifying when defenders converge on the raider and calculating the severity of the impact based on speed and approach angles.

## 2. Technical Overview
- **Type**: Spatial Interaction / Collision Detection Module
- **Dependencies**: `numpy`.
- **Input**: Positions of Raider and All Defenders.
- **Output**: `impact_info` (Severity, Colliding Defenders List).

## 3. Detailed Technical
- **Spatial Tracking**: Tracks the position and velocity of every defender independently.
- **Approach Detection**: Identifies defenders moving *towards* the raider (using dot product of velocity vector and direction vector). This distinguishes an attack from a retreat.
- **Collision Logic**: Checks proximity. If a defender enters the `collision_radius` (100px) with high velocity, it's a tackle.
- **Severity Modeling**: A formula combining:
    - **Velocity**: Faster tackles = higher risk.
    - **Angle**: Direct hits (head-on) are worse than glancing blows.
    - **Count**: "Gang tackles" (3+ defenders) increase severity significantly.

## 4. Workflow and Function Details

### Workflow
1.  **Update Defenders**: Update history for all non-raider players.
2.  **Identify State**: For each defender, check if they are "Approaching" or "Colliding".
3.  **Compute Severity**: Apply weighted formula to the set of interacting defenders.
4.  **Annotate**: Draw collision circles and approach vectors.

### Functions

#### `ImpactDetector` Class

##### `detect_impacts(self, players, raider_id, frame_num) -> (bool, dict)`
- **Purpose**: Main logic.
- **Details**: Separates raider from defenders. Loops through defenders to check distance/velocity. Returns detection status.

##### `calculate_approach_angle(self, def_pos, def_vel, raider_pos)`
- **Purpose**: Vector math.
- **Details**: Returns angle in degrees. 0 deg = moving straight at raider. used to filter out defenders who are close but moving *away*.

##### `_calculate_impact_severity(self, approaching, colliding, ...)`
- **Purpose**: Risk scoring.
- **Details**: Sums contributions from velocity (30%), distance (20%), angle (15%), and defender count (20%).

##### `annotate_impact_detection(frame, ...)`
- **Purpose**: Visualization.
- **Details**: Draws:
    - Orange lines for approaching defenders.
    - Red boxes for colliding defenders.
    - "IMPACT" flash alert.
