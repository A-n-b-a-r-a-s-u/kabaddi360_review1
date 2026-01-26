# risk_fusion.py - Documentation

## 1. Overview
`models/risk_fusion.py` implements **Stage 7** (the Final Stage) of the pipeline. It is the "brain" of the system. It takes the independent signals from the Fall Detector, Impact Detector, Motion Analyzer, and the player's historical data, and fuses them into a single, actionable **Injury Risk Score** (0-100%). It also manages the persistent database of player injuries.

## 2. Technical Overview
- **Type**: Data Fusion & Risk Assessment Engine
- **Dependencies**: 
    - `pandas`: For managing the CSV-based injury database.
    - `numpy`: Numerical operations.
    - `config.config`: Fetches weights (α, β, γ, δ).
- **Input**: Scores from Stages 4, 5, 6 + Player ID.
- **Output**: Final `risk_score`, Risk Level (LOW/MEDIUM/HIGH), and Alert Triggers.

## 3. Detailed Technical
- **Weighted Fusion Algorithm**:
    The core formula is:
    $$ Risk = ( \alpha \times Fall ) + ( \beta \times Impact ) + ( \gamma \times Motion ) + ( \delta \times History ) $$
    Where current weights are approx: Fall (35%), Impact (30%), Motion (25%), History (10%).
- **Temporal Smoothing**: To prevent rapid flickering of risk scores (e.g., 20% -> 90% -> 20%), it uses a sliding window (deque) to smooth the output risk score over 15 frames.
- **Database Management**: It calculates a "Risk Modifier" for each player based on their past injury count and severity. If a high-risk event occurs (Risk > 85%), it automatically updates the CSV database to record this new "injury".

## 4. Workflow and Function Details

### Workflow
1.  **Load History**: On init, loads `data/injury_history.csv`.
2.  **Get Modifier**: Look up the current player's past record.
3.  **Calculate Raw Score**: Apply the weighted formula.
4.  **Smooth**: Average the score with recent history.
5.  **Assess Level**: Map score to Low/Medium/High buckets.
6.  **Auto-Update**: If critical usage, write back to CSV.

### Functions

#### `RiskFusionEngine` Class

##### `calculate_risk_score(self, fall, impact, motion, player_id) -> dict`
- **Purpose**: Main fusion logic.
- **Details**: Performs weighted sum, clipping, smoothing, and level mapping. Returns a rich dictionary with breakdown for UI explanation.

##### `update_injury_history(self, player_id, event)`
- **Purpose**: Persistence.
- **Details**: Updates the player's record (increment count, re-calculate average severity) and saves to CSV.

##### `_load_injury_history(self)`
- **Purpose**: Database connection.
- **Details**: Robust CSV loader that handles multiple encoding types (utf-8, cp1252) to prevent crashes on different systems.

##### `annotate_risk_score(frame, risk_data)`
- **Purpose**: Visualization.
- **Details**: Draws the "Risk Meter" (gauge) and the component breakdown text on the frame.
