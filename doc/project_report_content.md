# Abstract

Kabaddi is a high-intensity contact sport that requires accurate real-time monitoring to prevent player injuries and ensure fair play. Traditional manual monitoring methods are subjective and often fail to detect subtle biomechanical indicators of injury risk. This project proposes a **Real-time Kabaddi Injury Prediction System** utilizing advanced Deep Learning and Computer Vision techniques. The system integrates **YOLOv8** for robust multi-person detection, **Pose Estimation** for skeletal tracking, and a **Kalman Filter** for trajectory smoothing to mitigate motion noise. To assess injury risk, we implement a hybrid analysis pipeline: a **Kinematic Motion Analyzer** extracts physical features such as velocity, acceleration, and jerk to quantify joint stress, while a **Temporal LSTM (Long Short-Term Memory)** network detects anomalous motion patterns indicative of falls or unsafe tackles. Furthermore, **Hough Transform** based court line detection is incorporated to provide spatial context, ensuring actions are evaluated within the game's boundaries. The proposed solution offers a comprehensive, automated tool for coaches and medical staff to monitor player safety and performance in real-time.

---

# Chapter 1 - INTRODUCTION

## 1.1 OVERVIEW
Sports analytics has revolutionized how games are played and monitored. In contact sports like Kabaddi, the risk of injury is exceptionally high due to the nature of the game, which involves rapid movements, sudden tackles, and collisions. Ensuring player safety requires continuous vigilance, which is challenging for human referees and medical staff to maintain consistently throughout a match. This project introduces an automated, vision-based system designed to monitor Kabaddi matches, track player movements, and predict potential injury risks using state-of-the-art artificial intelligence.

## 1.2 PROBLEM STATEMENT
*   **High Injury Rate**: Kabaddi involves frequent physical impact, leading to ligament tears, fractures, and concussions.
*   **Limitations of Human Monitoring**: Referees and coaches cannot track the biomechanical stress on every player simultaneously.
*   **Noisy Data**: Raw video data from sports environments is often noisy due to rapid motion and occlusion, making accurate analysis difficult.
*   **Lack of Context**: existing generic motion capture systems often fail to account for the specific temporal sequences (like a raid) and spatial boundaries (court lines) unique to Kabaddi.

## 1.3 OBJECTIVES
The primary objectives of this project are:
1.  To develop a **real-time player detection and tracking** system capable of handling multiple players in a crowded court.
2.  To implement **Kalman Filtering** for smoothing skeletal, joint trajectories and reducing jitter in pose estimation.
3.  To analyze **kinematic features** (velocity, acceleration, jerk) and **temporal patterns** (using LSTM) to calculate an injury risk score.
4.  To integrate **Court Line Detection** using Hough Transform to spatially contextualize player position (e.g., out-of-bounds events).
5.  To provide a visual interface for coaches to view live risk assessments and alerts.

## 1.4 SOLUTION OVERVIEW
The proposed solution is a multi-stage pipeline:
1.  **Input & Detection**: Video feed is processed using **YOLOv8** to detect and track individual players/raiders.
2.  **Pose Estimation**: Key body joints (shoulders, knees, ankles) are extracted to form a skeletal representation of the player.
3.  **Data Assimilation**: A **Landmark Kalman Filter** is applied to the raw joint coordinates. This filters out detection noise and provides smooth trajectory estimates, ensuring that velocity and acceleration calculations are accurate.
4.  **Feature Extraction**:
    *   **Kinematic Analysis**: The system calculates biomechanical metrics such as joint velocity, acceleration, and "jerk" (rate of change of acceleration) to measure physical stress.
    *   **Spatial Analysis**: **Hough Transformation** is used to detect court lines, defining the active play area.
5.  **Risk Prediction**: A **Temporal LSTM Network** analyzes the sequence of movements over time to detect anomalies (e.g., sudden falls, unnatural twists). This is combined with rule-based kinematic thresholds to generate a final **Injury Risk Score**.

## 1.5 ORGANIZATION OF THE REPORT
The report is organized as follows:
*   **Chapter 2** presents the Literature Survey, reviewing existing work in sports analysis and the specific algorithms selected for this project.
*   **Chapter 3** details the System Analysis and Design.
*   **Chapter 4** describes the Implementation details of the Deep Learning models and signal processing modules.
*   **Chapter 5** discusses the Results and Performance Evaluation.
*   **Chapter 6** concludes the report and discusses future scope.

---

# Chapter 2 - LITERATURE SURVEY

## 2.1 EXISTING WORK
Our research builds upon several fundamental studies in computer vision and sports biomechanics. We have selected specific "base papers" that contribute distinct modules to our final architecture.

### 2.1.1 Main Reference
**"Real-time sports injury monitoring system based on the deep learning algorithm"**
*   **Contribution**: This paper serves as the foundational framework for our project. It demonstrates the feasibility of using deep learning (CNNs and Pose Estimation) to monitor athletes in real-time. We adopt their overall system architecture design, which processes video frames to extract player metrics for health monitoring.

### 2.1.2 Motion Smoothing & Tracking
**"Multi-Person Fall Detection Using Data Assimilation Method With Kalman Filter"**
*   **Feature Adopted**: We utilize the **Kalman Filter** technique described in this paper.
*   **Relevance**: Raw pose estimation output is often jittery. As suggested by this work, we apply a Kalman Filter to "assimilate" noisy measurements into a smooth trajectory. This is critical for calculating accurate higher-order derivatives like acceleration and jerk, which are sensitive to noise.

### 2.1.3 Kinematic Analysis
**"Video-Based Human Motion Analysis for Injury Risk Assessment"**
*   **Feature Adopted**: **Joint Trajectory & Kinematic Feature Extraction**.
*   **Relevance**: This paper emphasizes that simple position tracking is insufficient for injury prediction. We implement their methodology of extracting **velocity, acceleration, and jerk** vectors for key joints (knees, ankles). These kinematic features directly correlate with the biomechanical load and potential tissue damage.

### 2.1.4 Temporal Anomaly Detection
**"Deep Learning for Fall Detection: Three-Dimensional CNN Combined With LSTM on Video Kinematic Data"**
*   **Feature Adopted**: **LSTM (Long Short-Term Memory) Network**.
*   **Relevance**: Injury events are not instantaneous but evolve over a sequence of frames. We adopt the LSTM approach from this paper to model temporal dependencies. Our **MotionLSTM** module analyzes a sliding window of historical joint data to predict if the current motion sequence resembles known injury patterns (like a fall).

### 2.1.5 Context Awareness
**"Use of the Hough Transformation to Detect Lines and Curves in Pictures"**
*   **Feature Adopted**: **Hough Transform for Line Detection**.
*   **Relevance**: To understand the game context (e.g., "raid" vs "out of bounds"), the system must "see" the court. We use the Standard Hough Transform technique to detect straight lines in the video frame, allowing us to map player positions relative to the Kabaddi court boundaries.

## 2.2 OBSERVATIONS FROM THE EXISTING WORK
1.  **Noise Sensitivity**: Purely deep-learning-based pose estimators often produce noisy outputs that make velocity calculation unreliable without post-processing (Kalman Filtering).
2.  **Temporal Importance**: Single-frame analysis (CNN only) fails to capture the momentum and force of impact; temporal sequence models (RNN/LSTM) are necessary for accurate fall/impact detection.
3.  **Context Gap**: Many existing generic systems detect "persons" but lack understanding of the sports field boundaries, leading to false positives (e.g., a player resting on the bench being analyzed).

## 2.3 LIMITATIONS OF EXISTING WORK
*   **Computational Cost**: Some 3D CNN approaches referenced in literature are too heavy for real-time inference on standard hardware.
*   **Lack of Specificity**: Most systems are designed for solitary activities (running/yoga) and struggle with the occlusion and multi-person contact inherent in team sports like Kabaddi.
*   **Static Thresholds**: Older kinematic systems rely solely on fixed thresholds (e.g., "velocity > x"), which fail to adapt to safe high-speed movements versus dangerous uncontrolled falls.

## 2.4 SUMMARY OF LITERATURE SURVEY
The literature suggests that a robust injury prediction system cannot rely on a single algorithm. It requires a **hybrid approach**: **Deep Learning** (YOLO/Pose) for perception, **Signal Processing** (Kalman Filters) for accuracy, **Kinematics** (Physics-based features) for interpretability, and **Sequence Modeling** (LSTM) for capturing complex temporal dynamics. The integration of **Hough Transform** adds necessary spatial awareness. Our proposed system synthesizes these distinct elements into a unified real-time pipeline tailored for Kabaddi.
