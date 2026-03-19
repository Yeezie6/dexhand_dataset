# Dexterous Hand Datasets & Benchmarks: A Decision-Grade Survey Report

**Prepared for:** PI leading a new egocentric dexterous-hand dataset project
**Date:** March 2025
**Scope:** Dexterous / multi-finger / anthropomorphic hand datasets and benchmarks, 2022–2025

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Taxonomy of the Field](#2-taxonomy-of-the-field)
3. [Comprehensive Dataset Table](#3-comprehensive-dataset-table)
4. [Comprehensive Benchmark Table](#4-comprehensive-benchmark-table)
5. [Deep Analysis by Theme](#5-deep-analysis-by-theme)
6. [Critical Gap Analysis](#6-critical-gap-analysis)
7. [Design Recommendations for a New Dataset](#7-design-recommendations-for-a-new-dataset)
8. [Ranked Shortlist](#8-ranked-shortlist)
9. [Final Recommendation](#9-final-recommendation)

---

## 1. Executive Summary

### 1.1 Landscape Overview

The field of dexterous robotic hand data has undergone a rapid transformation between 2022 and 2025. Three distinct waves of work are visible:

**Wave 1 (2020–2022): Grasp-pose and hand-pose datasets.** The dominant paradigm was collecting static or quasi-static grasp configurations — either from human hands (DexYCB, GRAB, ContactPose, OakInk) or synthesized for robot hands (DexGraspNet). These datasets are rich in 3D hand/object geometry but are fundamentally about *pose*, not *manipulation*. They do not provide the sequential action trajectories needed for policy learning.

**Wave 2 (2022–2023): Simulation-heavy dexterous manipulation.** Works like Bi-DexHands, DexArt, UniDexGrasp, and DexMV pushed toward learning dexterous manipulation policies, but almost entirely in simulation. Real-world data remained scarce. The gap between simulation and reality was acknowledged but not closed. Egocentric capture was essentially absent from robot-learning pipelines.

**Wave 3 (2024–2025): Real-world, scalable, and egocentric pipelines.** The most recent work — DexCap, DexWild, HumanPlus, EgoDex, DexCanvas, TACO, OakInk2, DexMimicGen, GR-Dexter — represents a genuine shift toward real-world dexterous manipulation data at scale. Egocentric capture is now appearing (EgoDex, HumanPlus, OpenTouch, OpenEgo). Human-to-robot retargeting pipelines are maturing. However, the combination of *egocentric + high-DoF robot action labels + sequential manipulation + diverse scenes* in a single dataset does not yet exist.

### 1.2 What Is Missing

The critical gaps, in order of severity for your project:

1. **No egocentric dataset with high-DoF robot action labels for sequential manipulation.** EgoDex (2025) is the closest — 829 hours, egocentric, 194 tasks — but it provides human hand tracking, not robot joint states. DexWild provides near-egocentric wrist cameras but only 5 tasks and 9,505 episodes. No dataset combines egocentric RGB with retargeted dexterous robot joint trajectories across diverse sequential tasks.

2. **No benchmark that evaluates cross-scene / cross-task / cross-embodiment generalization from egocentric input.** Existing benchmarks (Bi-DexHands, DexArt, HumanoidBench) are simulation-only and do not use egocentric observations. Real-world benchmarks with standardized splits are absent.

3. **Grasp datasets dominate; sequential manipulation data is thin.** The vast majority of dexterous hand data covers single-grasp events. Long-horizon tasks (pick → regrasp → place → use tool) are almost entirely absent from real-world dexterous datasets.

4. **Tactile and contact data is nearly absent.** OpenTouch (2024) is the only egocentric tactile dataset, and it covers only 5.1 hours. No dataset combines egocentric vision + tactile + robot joint states.

5. **Scene diversity is low.** Most datasets use controlled lab settings with a fixed camera and a small object set. DexWild (93 environments) and DROID (564 scenes, but parallel gripper) are exceptions. No dexterous hand dataset approaches this level of scene diversity.

6. **Language grounding is absent.** No dexterous hand dataset provides language instructions paired with manipulation trajectories at scale.

### 1.3 Strongest Design Recommendations

For a new egocentric dexterous-hand dataset to be genuinely novel and publishable, it should:

1. **Collect egocentric RGB-D from a head-mounted or wrist-mounted camera** paired with full-hand joint tracking (human) and retargeted robot joint states (e.g., LEAP Hand or Inspire Hand).
2. **Cover sequential manipulation tasks** (not just grasps): pick-and-place, regrasp, tool use, articulated object manipulation, and at least one long-horizon task (e.g., table setting, assembly).
3. **Achieve scene diversity** across at least 50+ distinct environments (kitchens, workshops, offices, outdoor).
4. **Include language annotations** (task description + step-level instructions) to enable VLA training.
5. **Design a benchmark** with standardized splits for cross-object, cross-scene, cross-task, and cross-embodiment generalization.
6. **Target 500+ hours** of real egocentric data to be competitive with EgoDex (829 hrs) while differentiating on robot action labels and task diversity.

The clearest differentiation from existing work: **egocentric + high-DoF robot action labels + sequential tasks + scene diversity + language + benchmark**. No existing dataset has all five.

---

## 2. Taxonomy of the Field

### 2.1 Primary Taxonomy Table

| Work | Type | Human/Robot | Egocentric | Grasp-only / Sequential | Static/Temporal | Sim/Real/Mixed | 6DoF/High-DoF | Single/Bimanual | Perception/Action/Policy-Ready |
|------|------|-------------|------------|------------------------|-----------------|----------------|---------------|-----------------|-------------------------------|
| DexGraspNet (2022) | Dataset | Robot | No | Grasp-only | Static | Sim | High-DoF (24) | Single | Action (grasp pose) |
| DexGraspNet 2.0 (2024) | Dataset | Robot | No | Grasp-only | Static | Sim | High-DoF (24) | Single | Action (grasp pose) |
| DexYCB (2021) | Dataset | Human | No | Grasp-only | Temporal | Real | ~6DoF wrist | Single | Perception |
| GRAB (2020) | Dataset | Human | No | Grasp-only | Temporal | Real (MoCap) | High-DoF | Single | Perception |
| OakInk (2022) | Dataset | Human | No | Grasp-only | Temporal | Real | High-DoF | Single | Perception |
| OakInk2 (2024) | Dataset | Human | Partial | Sequential | Temporal | Real | High-DoF | Bimanual | Action |
| HOI4D (2022) | Dataset | Human | Yes | Sequential | Temporal | Real | High-DoF | Single | Perception |
| ARCTIC (2023) | Dataset | Human | Both | Sequential | Temporal | Real | High-DoF | Bimanual | Perception |
| AssemblyHands (2023) | Dataset | Human | Both | Sequential | Temporal | Real | High-DoF | Bimanual | Perception |
| H2O (2021) | Dataset | Human | Both | Sequential | Temporal | Real | High-DoF | Bimanual | Action |
| TACO (2024) | Dataset | Human | Both | Sequential | Temporal | Real | High-DoF | Bimanual | Action |
| ContactPose (2020) | Dataset | Human | No | Grasp-only | Static | Real | High-DoF | Single | Perception |
| EgoDex (2025) | Dataset | Human | Yes | Sequential | Temporal | Real | High-DoF (21) | Single | Action |
| DexWild (2025) | Dataset | Human→Robot | Near-ego | Sequential | Temporal | Real | High-DoF | Single | Action |
| DexCanvas (2025) | Dataset | Human | No | Sequential | Temporal | Mixed | High-DoF | Single | Action |
| DexCap (2024) | Dataset | Human→Robot | No | Sequential | Temporal | Real | High-DoF | Single | Policy-Ready |
| HumanPlus (2024) | Dataset | Robot (humanoid) | Yes | Sequential | Temporal | Real | High-DoF | Bimanual | Policy-Ready |
| OpenTouch (2024) | Dataset | Human | Yes | Sequential | Temporal | Real | High-DoF | Single | Perception |
| OpenEgo (2025) | Dataset | Human | Yes | Sequential | Temporal | Real | High-DoF | Single | Action |
| DexMV (2022) | Dataset | Human→Robot | No | Sequential | Temporal | Mixed | High-DoF (16) | Single | Policy-Ready |
| DIME (2022) | Dataset | Robot | No | Sequential | Temporal | Mixed | High-DoF (16) | Single | Policy-Ready |
| Bi-DexHands (2022) | Benchmark | Robot | No | Sequential | Temporal | Sim | High-DoF (48) | Bimanual | Policy-Ready |
| DexArt (2023) | Benchmark | Robot | No | Sequential | Temporal | Sim | High-DoF | Single | Policy-Ready |
| UniDexGrasp (2023) | Benchmark | Robot | No | Grasp-only | Static | Sim | High-DoF (24) | Single | Action |
| UniDexGrasp++ (2023) | Benchmark | Robot | No | Grasp-only | Static | Sim | High-DoF (24) | Single | Action |
| GenDexGrasp (2023) | Benchmark | Robot | No | Grasp-only | Static | Sim | High-DoF | Single | Action |
| GraspXL (2024) | Dataset/Method | Robot | No | Grasp-only | Temporal | Sim | High-DoF | Single | Policy-Ready |
| DexMimicGen (2024) | Dataset/Method | Robot | No | Sequential | Temporal | Sim | High-DoF | Bimanual | Policy-Ready |
| HumanoidBench (2024) | Benchmark | Robot | No | Sequential | Temporal | Sim | High-DoF | Bimanual | Policy-Ready |
| BimanGrasp (2024) | Dataset | Robot | No | Grasp-only | Static | Sim | High-DoF | Bimanual | Action |
| ManipTrans/DexManipNet (2025) | Dataset | Robot | No | Sequential | Temporal | Mixed | High-DoF | Bimanual | Policy-Ready |
| GR-Dexter (2024) | Dataset | Robot | No | Sequential | Temporal | Real | High-DoF (21) | Bimanual | Policy-Ready |
| DexGrasp Anything (2025) | Dataset | Robot | No | Grasp-only | Static | Sim | High-DoF | Single | Action |
| Get a Grip (2024) | Dataset | Robot | No | Grasp-only | Static | Mixed | High-DoF | Single | Action |
| CyberDemo (2024) | Method/Dataset | Robot | No | Sequential | Temporal | Mixed | High-DoF | Single | Policy-Ready |
| AnyDexGrasp (2025) | Method | Robot | No | Grasp-only | Static | Real | High-DoF | Single | Action |
| WildHands (2024) | Dataset | Human | Yes | N/A | Static | Real | High-DoF (21) | Single | Perception |
| DexSim2Real² (2024) | Method | Robot | No | Sequential | Temporal | Mixed | High-DoF (16) | Single | Policy-Ready |


---

## 3. Comprehensive Dataset Table

> **Legend:** DoF = degrees of freedom of the hand; Ego = egocentric; Seq = sequential manipulation; Lang = language annotations; HiDoF = supports high-DoF dexterous learning; Bench = benchmark included; Rel = relevance to new egocentric dexhand dataset (H=High, M=Medium, L=Low)

| Name | Year | Paper URL | Code URL | Data URL | Type | Human/Robot | Ego | Real/Sim | Tasks | #Tasks | #Scenes | #Objects | #Demos/Seqs | Scale | Modalities | Labels/Annotations | Actions | Hand | DoF | HiDoF | Seq | Lang | Bench | Strengths | Weaknesses | Rel |
|------|------|-----------|----------|----------|------|-------------|-----|----------|-------|--------|---------|----------|-------------|-------|------------|-------------------|---------|------|-----|-------|-----|------|-------|-----------|------------|-----|
| DexGraspNet | 2022 | [arxiv:2210.02697](https://arxiv.org/abs/2210.02697) | [github](https://github.com/PKU-EPIC/DexGraspNet) | [project](https://pku-epic.github.io/DexGraspNet/) | Grasp dataset | Robot | No | Sim | Grasping | 1 | N/A | 5,355 obj / 133 cat | 1.32M grasps | 1.32M grasps | 3D meshes, grasp poses | Grasp pose, physics validity | Yes (grasp pose) | Shadow Hand | 24 | Yes | No | No | No | Largest dexterous grasp dataset; physics-validated; widely used baseline | Simulation only; grasp-pose only; no sequential manipulation; no egocentric | M |
| DexGraspNet 2.0 | 2024 | [arxiv:2410.23004](https://arxiv.org/abs/2410.23004) | [github](https://github.com/PKU-EPIC/DexGraspNet2) | [project](https://pku-epic.github.io/DexGraspNet2.0/) | Grasp dataset | Robot | No | Sim | Grasping (cluttered) | 1 | Cluttered scenes | 1,319 obj | 427M grasps | 427M grasps | Point clouds, grasp poses | Grasp pose, scene context | Yes (grasp pose) | Shadow Hand | 24 | Yes | No | No | No | Massive scale; cluttered scene grasping; real-world point cloud conditioning | Simulation only; grasp-pose only; no manipulation trajectories | M |
| DexYCB | 2021 | [arxiv:2104.04631](https://arxiv.org/abs/2104.04631) | [github](https://github.com/NVlabs/dex-ycb-toolkit) | [dex-ycb.github.io](https://dex-ycb.github.io/) | Grasp dataset | Human | No | Real | Grasping | 1 | Lab | 20 YCB obj | ~1,000 seqs | unclear | RGB-D (multi-view), hand pose, 6D obj pose | Hand pose (MANO), object pose | Partial (grasp seqs) | Human hand | ~21 | Yes | No | No | Yes | Real multi-view RGB-D; YCB objects; benchmark for hand/object pose | Exocentric only; grasp-only; no sequential manipulation; small object set | M |
| GRAB | 2020 | [arxiv:2008.11200](https://arxiv.org/abs/2008.11200) | [github](https://github.com/otaheri/GRAB) | [grab.is.tue.mpg.de](https://grab.is.tue.mpg.de) | Grasp dataset | Human | No | Real (MoCap) | Grasping | 1 | MoCap studio | 51 obj | 10 subj × 51 obj | unclear | MoCap, SMPL-X body, MANO hand, object mesh | Full body+hand pose, object pose | Partial (grasp seqs) | Human hand | ~21 | Yes | No | No | No | Full body+hand MoCap; 51 objects; SMPL-X annotations | MoCap studio only; no egocentric; no sequential manipulation; small scale | L |
| OakInk | 2022 | [arxiv:2203.15709](https://arxiv.org/abs/2203.15709) | [github](https://github.com/oakink/OakInk) | [oakink.net](http://oakink.net/) | HOI dataset | Human | No | Real | Grasping, affordance | 1 | Lab | unclear | unclear | unclear | RGB, depth, MANO hand pose, object pose | Hand pose, object pose, affordance | Partial | Human hand | ~21 | Yes | No | No | No | Affordance annotations; knowledge repository for HOI | Exocentric; grasp-focused; no sequential manipulation | L |
| OakInk2 | 2024 | [arxiv:2403.19417](https://arxiv.org/abs/2403.19417) | [github](https://github.com/oakink/OakInk2) | [oakink.net/v2](https://oakink.net/v2/) | Manipulation dataset | Human | Partial | Real | Bimanual manipulation, tool use | ~10+ | Lab | unclear | unclear | unclear | Multi-view RGB, body/hand/object meshes | Hand pose, object pose, action primitives, task hierarchy | Yes | Human (bimanual) | ~21 | Yes | Yes | No | No | Bimanual; complex task hierarchy (affordance→primitive→task); CVPR 2024 | Mostly exocentric; no robot action labels; lab setting only | H |
| HOI4D | 2022 | [arxiv:2203.01577](https://arxiv.org/abs/2203.01577) | [github](https://github.com/leolyliu/HOI4D-Instructions) | [hoi4d.github.io](https://hoi4d.github.io/) | HOI dataset | Human | Yes | Real | Category-level HOI | 16 cat | 610 rooms | 800 instances / 16 cat | 4,000 seqs | 2.4M frames | RGB-D (egocentric), 3D obj reconstruction, hand pose | Hand pose, object pose, action labels, 3D reconstruction | Yes | Human hand | ~21 | Yes | Yes | No | No | Egocentric; large-scale; diverse rooms; category-level; 4D reconstruction | HOI-focused not manipulation-policy-focused; no robot action labels | H |
| ARCTIC | 2023 | [arxiv:2204.13662](https://arxiv.org/abs/2204.13662) | [github](https://github.com/zc-alexfan/arctic) | [arctic.is.tue.mpg.de](https://arctic.is.tue.mpg.de) | Manipulation dataset | Human | Both | Real | Bimanual articulated obj manipulation | ~10 | Lab | 11 articulated obj | unclear | 2.1M frames | Multi-view RGB, 3D hand meshes, 3D obj meshes, contact maps | Hand pose, object pose, contact maps, articulation state | Yes | Human (bimanual) | ~21 | Yes | Yes | No | No | Bimanual; articulated objects; dynamic contact maps; egocentric+exocentric | Lab setting; no robot action labels; limited object diversity | H |
| AssemblyHands | 2023 | [arxiv:2304.12301](https://arxiv.org/abs/2304.12301) | unclear | unclear | Manipulation dataset | Human | Both | Real | Assembly (LEGO-like) | 1 | Lab | LEGO parts | unclear | 3.0M images | RGB (ego+exo), 3D hand pose | 3D hand pose, assembly step labels | Partial | Human (bimanual) | ~21 | Yes | Yes | No | No | Egocentric; assembly task; large-scale annotations | Single task type; no object pose; no robot action labels | M |
| H2O | 2021 | [arxiv:2104.11181](https://arxiv.org/abs/2104.11181) | unclear | unclear | HOI dataset | Human | Both | Real | Bimanual HOI, action recognition | ~10 | Lab | unclear | unclear | unclear | Multi-view RGB-D, hand poses, 6D obj pose, segmentation | Hand pose, object pose, action labels | Yes | Human (bimanual) | ~21 | Yes | Yes | No | No | Bimanual; egocentric+exocentric; action labels | Small scale; lab setting; no robot action labels | M |
| TACO | 2024 | [arxiv:2401.08399](https://arxiv.org/abs/2401.08399) | [github](https://github.com/LinghaoChan/TACO) | [taco2024.github.io](https://taco2024.github.io) | Tool-use dataset | Human | Both | Real | Bimanual tool use | ~100 combos | Lab | ~50 tools | 2,500 seqs | 2,500 seqs | RGB (ego+exo), 3D hand-obj meshes | Hand pose, object pose, tool-action-object labels | Yes | Human (bimanual) | ~21 | Yes | Yes | No | No | Egocentric+exocentric; tool-action-object taxonomy; bimanual; CVPR 2024 | Lab setting; no robot action labels; limited scene diversity | H |
| ContactPose | 2020 | [arxiv:2007.09545](https://arxiv.org/abs/2007.09545) | [github](https://github.com/facebookresearch/ContactPose) | [contactpose.cc.gatech.edu](https://contactpose.cc.gatech.edu) | Grasp dataset | Human | No | Real | Grasping with contact | 1 | Lab | 25 obj | 2,306 grasps | 2,306 grasps | RGB-D (multi-view), hand pose, contact maps (thermal) | Hand pose, object pose, contact maps, grasp intent | Partial | Human hand | ~21 | Yes | No | No | No | Real contact maps (thermal); functional intent labels | Small scale; exocentric; grasp-only; no sequential manipulation | L |
| EgoDex | 2025 | [arxiv:2505.11709](https://arxiv.org/abs/2505.11709) | unclear | unclear | Manipulation dataset | Human | Yes | Real | Diverse household manipulation | 194 | unclear | unclear | unclear | 829 hrs | Egocentric RGB, 3D hand/finger tracking, SLAM camera pose | Hand pose (full finger), camera pose | Yes (hand tracking) | Human hand | ~21 | Yes | Yes | No | No | Largest egocentric dexterous dataset; 829 hrs; 194 tasks; full finger tracking | Human hand only; no robot action labels; no language; 2025 (very new) | H |
| DexWild | 2025 | [arxiv:2505.07813](https://arxiv.org/abs/2505.07813) | [github](https://github.com/dexwild/dexwild) | [HuggingFace](https://huggingface.co/datasets/boardd/dexwild-dataset) | Manipulation dataset | Human→Robot | Near-ego | Real | Pick, place, grasp | 5 | 93 envs | unclear | 9,505 eps | 9,505 eps | RGB (wrist+external), hand pose | Hand pose, robot action (retargeted) | Yes (retargeted) | Human→LEAP Hand | ~16 | Yes | Yes | No | No | In-the-wild; 93 environments; co-training human+robot; portable rig | Only 5 tasks; no egocentric head-mounted view; limited task diversity | H |
| DexCanvas | 2025 | [arxiv:2510.15786](https://arxiv.org/abs/2510.15786) | unclear | [HuggingFace](https://huggingface.co/datasets/Manggu/DexCanvas) | Manipulation dataset | Human | No | Mixed | Broad skill taxonomy | many | unclear | unclear | unclear | 7,000 hrs | RGB, force/contact, hand pose trajectories | Hand pose, contact/force annotations | Yes | Human (DexRobot) | ~21 | Yes | Yes | No | No | Massive scale (7,000 hrs); physics-validated contact; skill taxonomy | Not egocentric; mostly synthetic expansion; no robot joint states | M |
| DexCap | 2024 | [arxiv:2403.07788](https://arxiv.org/abs/2403.07788) | [github](https://github.com/j96w/DexCap) | [dex-cap.github.io](https://dex-cap.github.io/) | Manipulation dataset | Human→Robot | No | Real | Diverse dexterous manipulation | ~10 | Lab | unclear | unclear | unclear | RGB-D, wrist/finger mocap (EM+SLAM), point clouds | Hand pose, robot joint states (retargeted), point clouds | Yes (retargeted) | Human→LEAP Hand | 16 | Yes | Yes | No | No | Wearable mocap; occlusion-resistant; retargeted to LEAP Hand; point cloud obs | Not egocentric; lab setting; limited scale reported | H |
| HumanPlus | 2024 | [arxiv:2406.10454](https://arxiv.org/abs/2406.10454) | [github](https://github.com/MarkFzp/humanplus) | unclear | Manipulation dataset | Robot (humanoid) | Yes | Real | Whole-body + dexterous manipulation | ~10 | Lab | unclear | unclear | 40 hrs motion | Egocentric RGB, whole-body pose, proprioception | Whole-body pose, robot joint states | Yes | Unitree H1 + dexterous hands | ~21+ | Yes | Yes | No | No | Egocentric; humanoid robot; real-world; shadowing pipeline | Humanoid-specific; limited task diversity; no language | H |
| OpenTouch | 2024 | [arxiv:2512.16842](https://arxiv.org/abs/2512.16842) | unclear | unclear | Tactile dataset | Human | Yes | Real | Diverse manipulation | unclear | In-the-wild | unclear | unclear | 5.1 hrs | Egocentric RGB, full-hand tactile maps, LLM annotations | Tactile maps, video, language descriptions | Partial | Human (tactile glove) | ~21 | Yes | Yes | Yes | No | First egocentric full-hand tactile dataset; LLM annotations; in-the-wild | Very small scale (5.1 hrs); no robot action labels | H |
| OpenEgo | 2025 | [arxiv:2509.05513](https://arxiv.org/abs/2509.05513) | unclear | unclear | Aggregated dataset | Human | Yes | Real | Diverse manipulation | many | many | many | many | 1,107 hrs | Egocentric RGB, hand pose, action primitives | Hand pose, action primitives | Yes | Human hand | ~21 | Yes | Yes | No | No | Aggregates 6 datasets; 1,107 hrs; standardized annotations | Aggregation only; no new data; no robot action labels | M |
| DexMV | 2022 | [arxiv:2108.05877](https://arxiv.org/abs/2108.05877) | [github](https://github.com/yzqin/dexmv-sim) | unclear | Manipulation dataset | Human→Robot | No | Mixed | In-hand manipulation | 5 | Sim | ~10 obj | unclear | unclear | RGB video (human), sim states | Retargeted robot actions | Yes (retargeted) | Allegro Hand | 16 | Yes | Yes | No | No | Human-to-robot retargeting; in-hand manipulation tasks | Simulation-based; limited scale; no egocentric | M |
| DIME | 2022 | [arxiv:2203.13251](https://arxiv.org/abs/2203.13251) | [github](https://github.com/NYU-robot-learning/DIME-Models) | [project](https://nyu-robot-learning.github.io/dime) | Manipulation dataset | Robot | No | Mixed | In-hand manipulation | ~5 | Lab | ~5 obj | unclear | unclear | RGB, proprioception | Robot joint states, actions | Yes | Allegro Hand | 16 | Yes | Yes | No | No | Real Allegro Hand; in-hand manipulation; sim+real | Small scale; limited task diversity; no egocentric | M |
| WildHands | 2024 | [arxiv:2312.06583](https://arxiv.org/abs/2312.06583) | unclear | unclear | Hand pose dataset | Human | Yes | Real | N/A (pose estimation) | N/A | In-the-wild | N/A | unclear | unclear | Egocentric RGB, 3D hand pose | 3D hand pose (MANO) | No | Human hand | ~21 | Yes | No | No | No | In-the-wild egocentric hand pose; diverse backgrounds | Pose estimation only; no manipulation; no action labels | L |
| DexMimicGen | 2024 | [arxiv:2410.24185](https://arxiv.org/abs/2410.24185) | [github](https://github.com/NVlabs/dexmimicgen) | [project](https://dexmimicgen.github.io/) | Manipulation dataset | Robot | No | Sim | Bimanual dexterous manipulation | 9 | Sim | ~20 obj | 21,000 seqs | 21,000 seqs | RGB, sim states, proprioception | Robot joint states, actions | Yes | Humanoid dexterous hands | ~21+ | Yes | Yes | No | No | Large-scale synthetic; bimanual; humanoid; data augmentation pipeline | Simulation only; no egocentric; limited real-world transfer shown | M |
| Bi-DexHands | 2022 | [arxiv:2206.08686](https://arxiv.org/abs/2206.08686) | [github](https://github.com/PKU-MARL/DexterousHands) | [project](https://pku-marl.github.io/DexterousHands/) | Benchmark | Robot | No | Sim | Bimanual dexterous manipulation | 20 | Sim | thousands | N/A (RL) | N/A | Sim states, proprioception | Robot joint states, RL rewards | Yes (RL) | Two Shadow Hands | 48 | Yes | Yes | No | Yes | 20 bimanual tasks; Shadow Hand; RL benchmark; MARL support | Simulation only; no vision; no egocentric; no real-world | M |
| DexArt | 2023 | [arxiv:2305.05706](https://arxiv.org/abs/2305.05706) | [github](https://github.com/Kami-code/dexart-release) | unclear | Benchmark | Robot | No | Sim | Articulated object manipulation | 4 | Sim | 4 categories | N/A (RL) | N/A | Point cloud, proprioception | Robot joint states, RL rewards | Yes (RL) | Multi-finger robot hand | ~16 | Yes | Yes | No | Yes | Articulated objects; point cloud input; RL benchmark | Only 4 task categories; simulation only; no egocentric | M |
| UniDexGrasp | 2023 | [arxiv:2303.00938](https://arxiv.org/abs/2303.00938) | [github](https://github.com/PKU-EPIC/UniDexGrasp) | [project](https://pku-epic.github.io/UniDexGrasp/) | Benchmark | Robot | No | Sim | Universal dexterous grasping | 1 | Sim | 3,200+ obj | N/A | N/A | Point cloud | Grasp pose | Yes | Shadow Hand | 24 | Yes | No | No | Yes | Universal grasping; large object set; point cloud input | Grasp-only; simulation; no sequential manipulation | M |
| GraspXL | 2024 | [arxiv:2403.19649](https://arxiv.org/abs/2403.19649) | unclear | [project](https://eth-ait.github.io/graspxl/) | Dataset/Method | Robot | No | Sim | Grasping | 1 | Sim | 500K+ obj | 500K+ motions | 500K+ motions | Object meshes, grasp trajectories | Grasp motion trajectories | Yes | Multiple (Shadow, Allegro) | 16–24 | Yes | No | No | No | Massive object diversity; multi-hand; motion trajectories | Simulation only; grasp-only; no sequential manipulation | M |
| BimanGrasp | 2024 | [arxiv:2411.15903](https://arxiv.org/abs/2411.15903) | unclear | unclear | Grasp dataset | Robot | No | Sim | Bimanual grasping | 1 | Sim | unclear | 150K+ grasps | 150K+ grasps | Grasp poses, object meshes | Bimanual grasp poses | Yes | Robot dexterous (bimanual) | ~16–24 | Yes | No | No | No | First large-scale bimanual dexterous grasp dataset | Simulation only; grasp-only; no sequential manipulation | L |
| ManipTrans/DexManipNet | 2025 | [arxiv:2503.21860](https://arxiv.org/abs/2503.21860) | unclear | [project](https://maniptrans.github.io/) | Manipulation dataset | Robot | No | Mixed | Fine dexterous manipulation | ~5 | Sim+Lab | ~10 obj | 3,300 eps | 3,300 eps | Sim states, RGB, proprioception | Robot joint states, actions | Yes | Multiple dexterous hands | ~16–24 | Yes | Yes | No | No | Human-to-robot transfer; novel fine tasks (pen capping); multi-embodiment | Small scale; no egocentric; limited scene diversity | M |
| GR-Dexter | 2024 | [arxiv:2512.24210](https://arxiv.org/abs/2512.24210) | unclear | unclear | Manipulation dataset | Robot | No | Real | Long-horizon manipulation | ~10 | Lab | unclear | unclear | unclear | RGB, proprioception, teleoperation data | Robot joint states, actions | Yes | Custom 21-DoF hand (bimanual) | 21 | Yes | Yes | No | No | Real-world; 21-DoF custom hand; VLA training; bimanual | No egocentric; limited public data release; lab setting | H |
| CyberDemo | 2024 | [arxiv:2402.14795](https://arxiv.org/abs/2402.14795) | unclear | unclear | Method/Dataset | Robot | No | Mixed | In-hand manipulation, tool use | ~5 | Sim+Lab | ~10 obj | unclear | unclear | RGB, sim states, proprioception | Robot joint states, actions | Yes | Multi-finger dexterous hand | ~16 | Yes | Yes | No | No | Sim-to-real; data augmentation; CVPR 2024 | No egocentric; limited scale; sim-heavy | M |
| DexGrasp Anything | 2025 | [arxiv:2503.08257](https://arxiv.org/abs/2503.08257) | unclear | unclear | Grasp dataset | Robot | No | Sim | Universal grasping | 1 | Sim | 15,000+ obj | 3.4M grasps | 3.4M grasps | Point clouds, grasp poses | Grasp poses | Yes | Multiple robot hands | ~16–24 | Yes | No | No | No | Largest object diversity for dexterous grasping; physics-aware | Simulation only; grasp-only; no sequential manipulation | L |
| Get a Grip | 2024 | [openreview](https://openreview.net/forum?id=1jc2zA5Z6J) | unclear | [project](https://sites.google.com/view/get-a-grip-dataset) | Grasp dataset | Robot | No | Mixed | Grasping, sim-to-real | 1 | Lab | unclear | 3.5M grasps | 3.5M grasps | RGB-D, grasp labels, point clouds | Grasp labels, success/failure | Yes | Multi-finger robot hand | ~16 | Yes | No | No | No | Perceptual data with grasp labels; sim-to-real evaluation | Grasp-only; limited sequential manipulation | L |
| HumanoidBench | 2024 | [arxiv:2403.10506](https://arxiv.org/abs/2403.10506) | [github](https://github.com/carlosferrazza/humanoid-bench) | [project](https://humanoid-bench.github.io/) | Benchmark | Robot | No | Sim | Whole-body + dexterous manipulation | 27 | Sim | N/A | N/A (RL) | N/A | Sim states, proprioception | Robot joint states, RL rewards | Yes (RL) | Humanoid + dexterous hands | ~21+ | Yes | Yes | No | Yes | Humanoid; dexterous hands; 27 tasks; RL benchmark | Simulation only; no vision; no egocentric; no real-world | M |


---

## 4. Comprehensive Benchmark Table

| Name | Year | Scope | Tasks | Real/Sim | Embodiments | Input Modalities | Output/Control Space | Metrics | High-DoF | Egocentric Input | Long-Horizon | Cross-Embodiment | Main Limitations | Relevance |
|------|------|-------|-------|----------|-------------|-----------------|---------------------|---------|----------|-----------------|--------------|-----------------|-----------------|-----------|
| Bi-DexHands | 2022 | Bimanual dexterous manipulation RL | 20 tasks (catch, lift, scissors, etc.) | Sim (Isaac Gym) | Two Shadow Hands (48 DoF) | Sim state, proprioception | Joint torques (continuous) | Task success rate, reward | Yes (48 DoF) | No | Partial | No | Sim-only; no vision; no real-world transfer; no egocentric | High — defines bimanual dexterous task taxonomy |
| DexArt | 2023 | Dexterous articulated object manipulation | 4 categories (faucet, bucket, laptop, toilet) | Sim (SAPIEN) | Multi-finger robot hand (~16 DoF) | Point cloud, proprioception | Joint positions | Task success rate | Yes | No | No | No | Only 4 task types; sim-only; no egocentric; no language | High — articulated object manipulation benchmark |
| UniDexGrasp | 2023 | Universal dexterous grasping | Grasping 3,200+ objects | Sim (Isaac Gym) | Shadow Hand (24 DoF) | Point cloud | Joint positions | Grasp success rate | Yes (24 DoF) | No | No | No | Grasp-only; sim-only; no sequential manipulation | Medium — grasp benchmark reference |
| UniDexGrasp++ | 2023 | Geometry-aware dexterous grasping | Grasping with curriculum | Sim (Isaac Gym) | Shadow Hand (24 DoF) | Point cloud | Joint positions | Grasp success rate | Yes (24 DoF) | No | No | No | Grasp-only; sim-only | Medium — improved grasp benchmark |
| HumanoidBench | 2024 | Whole-body humanoid control | 27 tasks (locomotion + manipulation) | Sim (MuJoCo) | Humanoid + dexterous hands | Sim state, proprioception | Joint torques | Task success rate, reward | Yes | No | Partial | No | Sim-only; no vision; no egocentric; humanoid-specific | Medium — dexterous manipulation subset relevant |
| DexMimicGen | 2024 | Bimanual dexterous manipulation (data gen) | 9 bimanual tasks | Sim (Isaac Lab) | Humanoid dexterous hands | RGB, sim state | Joint positions | Task success rate | Yes | No | Partial | No | Sim-only; no egocentric; limited real-world transfer | Medium — data augmentation benchmark |
| DexYCB Benchmark | 2021 | Hand/object pose estimation | Grasping 20 YCB objects | Real | Human hand | RGB-D (multi-view) | Hand pose (MANO) | MPJPE, ADD-S | Yes | No | No | No | Pose estimation only; not manipulation policy | Medium — hand pose estimation reference |
| HOI4D Benchmark | 2022 | Category-level HOI recognition | 16 categories | Real | Human hand | RGB-D (egocentric) | Action labels | Action recognition accuracy | Yes | Yes | No | No | Recognition only; not manipulation policy | High — egocentric HOI benchmark reference |
| ARCTIC Benchmark | 2023 | Bimanual articulated object manipulation | ~10 articulated objects | Real | Human (bimanual) | Multi-view RGB | Hand/object pose | MPJPE, contact accuracy | Yes | Partial | No | No | Pose estimation focus; not policy learning | High — articulated object manipulation reference |
| TACO Benchmark | 2024 | Bimanual tool-action-object understanding | ~100 tool-action-object combos | Real | Human (bimanual) | RGB (ego+exo) | Action labels | Generalization accuracy | Yes | Yes | No | No | Recognition/understanding focus; not policy learning | High — tool use + egocentric benchmark |
| DexGraspNet Benchmark | 2022 | Dexterous grasp synthesis | 5,355 objects | Sim | Shadow Hand (24 DoF) | 3D meshes | Grasp poses | Grasp success rate, penetration | Yes (24 DoF) | No | No | No | Grasp-only; sim-only | Medium — grasp synthesis reference |
| DexGraspNet 2.0 Benchmark | 2024 | Cluttered scene dexterous grasping | 1,319 objects in clutter | Sim | Shadow Hand (24 DoF) | Point clouds | Grasp poses | Grasp success rate | Yes (24 DoF) | No | No | No | Grasp-only; sim-only | Medium — cluttered grasp reference |
| GraspXL Benchmark | 2024 | Multi-objective dexterous grasping | 500K+ objects | Sim | Multiple (Shadow, Allegro) | Object meshes | Grasp trajectories | Grasp success rate, diversity | Yes | No | No | Yes (multi-hand) | Sim-only; grasp-only | Medium — cross-embodiment grasp reference |
| AnyDexGrasp | 2025 | Universal dexterous grasping (real) | 150+ novel objects | Real | Multiple robot hands | RGB-D, point clouds | Grasp poses | Grasp success rate (real) | Yes | No | No | Yes (3 hands) | Grasp-only; no sequential manipulation | High — real-world cross-embodiment grasp benchmark |
| DexGrasp Anything | 2025 | Universal dexterous grasping | 15,000+ objects | Sim | Multiple robot hands | Point clouds | Grasp poses | Grasp success rate | Yes | No | No | Yes | Sim-only; grasp-only | Medium — large-scale grasp reference |
| Get a Grip | 2024 | Dexterous grasp evaluation (sim-to-real) | Grasping diverse objects | Mixed | Multi-finger robot hand | RGB-D, point clouds | Grasp labels | Grasp success rate (real) | Yes | No | No | No | Grasp-only | Medium — sim-to-real grasp evaluation |
| OakInk2 Benchmark | 2024 | Bimanual complex task understanding | Complex daily activities | Real | Human (bimanual) | Multi-view RGB | Action primitives | Task completion, generalization | Yes | Partial | Yes | No | Human-only; no robot policy | High — task hierarchy + bimanual reference |
| AssemblyHands Benchmark | 2023 | Egocentric assembly hand pose | Assembly tasks | Real | Human (bimanual) | RGB (ego+exo) | 3D hand pose | MPJPE | Yes | Yes | No | No | Pose estimation only; single task type | Medium — egocentric assembly reference |


---

## 5. Deep Analysis by Theme

### 5.1 Egocentric Dexterous Manipulation Data

**Representative works:** HOI4D (2022), ARCTIC (2023), TACO (2024), EgoDex (2025), HumanPlus (2024), OpenTouch (2024), OpenEgo (2025), DexWild (2025), AssemblyHands (2023)

**What they contribute:**

HOI4D is the most important prior egocentric dexterous dataset. It provides 2.4M RGB-D frames across 610 rooms with 16 object categories, making it the most scene-diverse egocentric hand dataset before 2025. However, it is fundamentally a *recognition* dataset — the annotations are action labels and object poses, not manipulation trajectories suitable for policy learning.

EgoDex (2025) is the most significant recent entry. At 829 hours with 194 tasks and full 3D finger tracking from a calibrated multi-camera egocentric rig, it is the first dataset that genuinely combines egocentric capture with dexterous hand tracking at scale. The SLAM-based camera pose enables 3D scene understanding. However, it provides human hand tracking only — no robot joint states, no language, and no standardized benchmark splits are reported.

TACO (2024) provides paired egocentric + third-person views for bimanual tool use, with precise 3D hand-object meshes. Its tool-action-object taxonomy is the most principled annotation structure in the field. The limitation is that it is lab-only and covers only ~2,500 sequences.

OpenTouch (2024) is a unique outlier: the first egocentric full-hand tactile dataset. At only 5.1 hours it is too small for policy learning, but it establishes the modality and demonstrates feasibility of in-the-wild egocentric tactile capture.

DexWild (2025) uses wrist-mounted cameras (near-egocentric) and collects data across 93 environments — the most scene-diverse dexterous manipulation dataset to date. Its co-training framework (human + robot data) is a strong design pattern. The limitation is only 5 task types.

**Where they fall short:**

No existing egocentric dataset simultaneously provides: (1) head-mounted egocentric RGB-D, (2) retargeted robot joint states, (3) sequential multi-step tasks, (4) diverse scenes (50+), and (5) language annotations. EgoDex is closest on scale and egocentric capture but misses robot action labels. DexWild is closest on scene diversity and robot transfer but misses head-mounted egocentric view and task diversity.

**Why it matters for a new dataset:**

The egocentric viewpoint is the natural perspective for a robot hand mounted on a robot arm — the wrist/head camera sees what the hand is doing. Egocentric data is also more scalable (no external camera setup required) and more natural for human demonstrators. A new dataset that fills the EgoDex + DexWild gap — egocentric + robot actions + diverse scenes + diverse tasks — would be immediately differentiated.

---

### 5.2 Human-to-Robot Transfer Datasets

**Representative works:** DexMV (2022), DexCap (2024), DexWild (2025), ManipTrans/DexManipNet (2025), CyberDemo (2024), HumanPlus (2024), AnyTeleop (2023)

**What they contribute:**

DexMV (2022) established the paradigm of retargeting human video demonstrations to robot hand joint states for imitation learning. It showed that human video is a viable source of dexterous manipulation data, but the retargeting was simulation-based and the scale was small.

DexCap (2024) is the most technically sophisticated human-to-robot transfer system. The wearable EM + SLAM mocap system is occlusion-resistant and portable, and the retargeting to LEAP Hand (16 DoF) produces real robot joint trajectories. The DexIL algorithm uses point cloud observations for policy learning. This is the closest existing work to what a new dataset project should build on.

DexWild (2025) scales the human-to-robot transfer paradigm to 93 environments using a portable wrist-camera rig. The co-training framework (mixing human and robot demonstrations) is a key insight: human data provides diversity, robot data provides precision.

ManipTrans (2025) introduces residual learning for human-to-robot transfer, enabling fine-grained tasks (pen capping, bottle unscrewing) that were previously too precise for direct retargeting. The resulting DexManipNet dataset covers novel task types.

**Where they fall short:**

All existing human-to-robot transfer datasets are either (a) small-scale lab settings, (b) limited to a few task types, or (c) not egocentric. None combines egocentric capture with retargeted robot actions at scale across diverse scenes.

**Why it matters:**

Human-to-robot transfer is the most scalable path to large dexterous manipulation datasets. Human demonstrators are cheap, fast, and naturally dexterous. The key technical challenge is retargeting — mapping human hand kinematics to robot hand kinematics while preserving manipulation intent. A new dataset should include retargeted robot joint states as a first-class annotation.

---

### 5.3 High-DoF Dexterous Hand Datasets

**The 6DoF vs. High-DoF distinction is critical and frequently conflated in the literature.**

A **6DoF dataset** provides only the 6-dimensional wrist pose (position + orientation) of the hand. This is sufficient for parallel-jaw gripper control but completely insufficient for dexterous hand control, which requires 16–24+ joint angles.

A **high-DoF dataset** provides the full joint configuration of the hand fingers — typically 16 DoF (Allegro, LEAP Hand) to 24 DoF (Shadow Hand) for robot hands, or ~21 DoF for human hands (MANO model).

**Datasets that are genuinely high-DoF:**
- DexGraspNet / DexGraspNet 2.0: Shadow Hand (24 DoF) grasp poses — high-DoF but grasp-only
- Bi-DexHands: Two Shadow Hands (48 DoF total) — high-DoF but simulation RL only
- DexCap: LEAP Hand (16 DoF) retargeted trajectories — high-DoF and real-world
- DexWild: LEAP Hand (16 DoF) retargeted — high-DoF and real-world
- EgoDex: Human hand (~21 DoF) tracking — high-DoF but human only
- GR-Dexter: Custom 21-DoF hand — high-DoF and real-world
- HumanPlus: Humanoid dexterous hands — high-DoF and real-world

**Datasets that are effectively 6DoF despite appearing dexterous:**
- DexYCB: Provides MANO hand pose but the benchmark tasks are grasp-only; the hand pose is used for pose estimation, not manipulation control
- HOI4D: Hand pose annotations are for recognition, not control
- ARCTIC: Hand pose for pose estimation / contact prediction, not robot control
- Most HOI datasets: Hand pose is a perception label, not an action label

**Why this matters:**

If you train a policy on 6DoF wrist data, you get a policy that can move a gripper to a location. If you train on high-DoF joint data, you get a policy that can control individual fingers. The gap is enormous for tasks like in-hand manipulation, tool use, and fine assembly. A new dataset must explicitly provide high-DoF joint trajectories — either human MANO parameters retargeted to robot joints, or direct robot joint state recordings.

---

### 5.4 Grasp-Pose Datasets vs. Full Manipulation Trajectory Datasets

**Grasp-pose datasets** (DexGraspNet, DexGraspNet 2.0, UniDexGrasp, ContactPose, Get a Grip, DexGrasp Anything) provide static or quasi-static hand configurations for grasping objects. They are useful for grasp synthesis and grasp planning but do not provide the temporal action sequences needed for manipulation policy learning.

**Full manipulation trajectory datasets** (DexCap, DexWild, TACO, OakInk2, ARCTIC, EgoDex, DexMV, DIME, HumanPlus) provide time-series of hand states and/or actions during manipulation tasks. These are what policy learning requires.

The field has a severe imbalance: grasp-pose datasets are large (millions of grasps) while manipulation trajectory datasets are small (hundreds to thousands of sequences). This is because grasp poses can be synthesized in simulation at scale, while manipulation trajectories require either real-world collection or careful simulation.

**The key insight for a new dataset:** The community does not need more grasp-pose data. It needs manipulation trajectory data — especially egocentric, real-world, diverse-scene, sequential manipulation trajectories with robot action labels.

---

### 5.5 Vision-Only vs. Vision-Tactile Datasets

The vast majority of dexterous hand datasets are vision-only. Tactile sensing is nearly absent from public datasets despite being critical for contact-rich manipulation.

**Existing tactile work:**
- ContactPose (2020): Thermal contact maps during grasping — real but static, exocentric
- OpenTouch (2024): Egocentric full-hand tactile maps — real, in-the-wild, but only 5.1 hours
- DexCanvas (2025): Physics-validated contact annotations — but synthetic, not real tactile sensors

**Why tactile matters:** Many dexterous manipulation tasks (in-hand reorientation, tool use, assembly) require contact feedback that is not visible in RGB images. Policies trained on vision-only data struggle with contact-rich tasks. Tactile data enables contact-aware policies.

**The gap:** No dataset combines egocentric RGB + real tactile sensing + robot joint states + sequential manipulation at meaningful scale. OpenTouch is the proof of concept; a new dataset could scale it up.

**Practical consideration:** Tactile gloves (e.g., SynTouch, Tekscan, custom designs) add hardware complexity and cost. A pragmatic approach is to include tactile as an optional modality in a subset of the dataset rather than requiring it for all data.

---

### 5.6 Simulation-Heavy Pipelines vs. Real-Data Pipelines

**Simulation-heavy:** Bi-DexHands, DexArt, UniDexGrasp, DexGraspNet, DexMimicGen, GraspXL, DexGrasp Anything. These achieve large scale but face the sim-to-real gap. RL policies trained in simulation often fail to transfer to real robots, especially for contact-rich tasks.

**Real-data pipelines:** DexCap, DexWild, TACO, HOI4D, EgoDex, HumanPlus, GR-Dexter. These are harder to scale but produce data that directly reflects real-world physics, lighting, and contact dynamics.

**Mixed/hybrid:** CyberDemo, DexCanvas, ManipTrans. These use simulation for data augmentation or physics validation while grounding in real demonstrations.

**The trend:** The field is clearly moving toward real-data pipelines. The 2024–2025 papers (DexCap, DexWild, EgoDex, GR-Dexter) all emphasize real-world collection. The sim-to-real gap is increasingly recognized as a fundamental barrier for dexterous manipulation.

**Recommendation for a new dataset:** Prioritize real-world data collection. Use simulation only for data augmentation (e.g., domain randomization, object pose perturbation) after establishing a real-world core dataset.

---

### 5.7 Bimanual Dexterous Manipulation

**Representative works:** Bi-DexHands (2022), ARCTIC (2023), OakInk2 (2024), TACO (2024), AssemblyHands (2023), DexMimicGen (2024), BimanGrasp (2024), GR-Dexter (2024), HumanPlus (2024)

**What they contribute:**

Bi-DexHands established the simulation benchmark for bimanual dexterous RL with 20 tasks. ARCTIC provided real bimanual data for articulated object manipulation. OakInk2 introduced a three-level task hierarchy for complex bimanual activities. TACO provided the most principled tool-use taxonomy for bimanual manipulation.

**Where they fall short:**

Real-world bimanual dexterous robot data is extremely scarce. GR-Dexter (2024) is one of the few systems with a real bimanual dexterous robot (21-DoF hands), but the dataset is not fully public. DexMimicGen provides large-scale bimanual data but only in simulation.

**Why it matters:**

Most real-world manipulation tasks are bimanual — cooking, assembly, tool use, packaging. A dataset that covers only single-hand manipulation is limited in its applicability. However, bimanual data collection is significantly more complex (two hands to track, more occlusion, more complex retargeting). A pragmatic approach is to include both single-hand and bimanual tasks, with bimanual as a subset.

---

### 5.8 Benchmarks for Dynamic Interaction / Handover / Articulated Objects / Tool Use

**Articulated objects:** DexArt (2023) is the primary simulation benchmark. ARCTIC (2023) provides real human data. Neither provides a real-world robot manipulation benchmark for articulated objects.

**Tool use:** TACO (2024) is the most comprehensive tool-use dataset. It covers ~100 tool-action-object combinations with egocentric + third-person views. However, it is a human dataset without robot action labels.

**Handover:** No dedicated dexterous hand handover benchmark exists. H2O (2021) includes some handover sequences but is not a dedicated benchmark.

**Dynamic interaction:** Bi-DexHands includes dynamic tasks (catching, throwing) but only in simulation.

**The gap:** There is no real-world benchmark that evaluates dexterous robot manipulation on articulated objects, tool use, or handover with standardized protocols. This is a significant opportunity for a new dataset.


---

## 6. Critical Gap Analysis

### Gap 1: No Egocentric Dataset with High-DoF Robot Action Labels for Sequential Manipulation

**Why it matters:** This is the single most important gap. Policy learning for dexterous robots requires (a) observations that match what the robot sees (egocentric), (b) actions that match what the robot does (high-DoF joint states), and (c) sequential task structure (not just single grasps). No existing dataset provides all three.

**What partially addresses it:**
- EgoDex (2025): Egocentric + sequential + high-DoF human hand tracking. Missing: robot action labels.
- DexCap (2024): High-DoF robot action labels (LEAP Hand) + sequential. Missing: egocentric view.
- DexWild (2025): Near-egocentric + robot action labels + diverse scenes. Missing: head-mounted egocentric, task diversity.
- HumanPlus (2024): Egocentric + robot action labels + sequential. Missing: scale, task diversity, public release.

**What is still missing:** A dataset that combines head-mounted egocentric RGB-D + retargeted high-DoF robot joint states + sequential multi-step tasks + diverse scenes. This is the primary opportunity for a new dataset.

---

### Gap 2: No Real-World Benchmark with Standardized Splits for Generalization

**Why it matters:** Without standardized train/val/test splits and evaluation protocols, it is impossible to compare methods or measure progress on generalization. The field currently has no agreed-upon real-world benchmark for dexterous manipulation policy learning.

**What partially addresses it:**
- Bi-DexHands: Standardized simulation benchmark, but simulation-only.
- DexArt: Standardized simulation benchmark for articulated objects, but simulation-only.
- DexYCB: Standardized benchmark for hand/object pose estimation, but not policy learning.
- HOI4D: Standardized benchmark for action recognition, but not policy learning.

**What is still missing:** A real-world benchmark with standardized splits for cross-object, cross-scene, cross-task, and cross-embodiment generalization of dexterous manipulation policies. This is the second major opportunity.

---

### Gap 3: Lack of Long-Horizon Task Structure

**Why it matters:** Real manipulation tasks are long-horizon — they involve multiple steps, sub-goals, and state transitions. A dataset of single-grasp events cannot train policies for tasks like "set the table" or "assemble a toy." Long-horizon structure is essential for VLA training.

**What partially addresses it:**
- OakInk2 (2024): Three-level task hierarchy (affordance → primitive → complex task). Human data only.
- TACO (2024): Tool-action-object sequences. Human data only.
- HumanoidBench (2024): Some long-horizon tasks in simulation.
- GR-Dexter (2024): Long-horizon manipulation with real robot. Limited public release.

**What is still missing:** A real-world dexterous robot dataset with long-horizon task structure (5+ steps), diverse tasks, and standardized evaluation. The longest existing real-world dexterous sequences are typically 1–3 steps.

---

### Gap 4: Weak Contact / Tactile / Force Supervision

**Why it matters:** Contact is the fundamental physical phenomenon in manipulation. Without contact supervision, policies cannot learn to apply appropriate forces, detect slip, or respond to unexpected contact. This is especially critical for dexterous manipulation where finger-object contact patterns determine grasp stability and manipulation success.

**What partially addresses it:**
- ContactPose (2020): Real thermal contact maps for grasping. Static, exocentric, small scale.
- ARCTIC (2023): Dynamic contact maps for bimanual articulated object manipulation. Human data, no robot actions.
- DexCanvas (2025): Physics-validated contact annotations. Synthetic expansion, not real tactile.
- OpenTouch (2024): Real egocentric full-hand tactile. Only 5.1 hours.

**What is still missing:** A dataset with real tactile sensor data (not just contact maps) paired with robot joint states and egocentric vision at meaningful scale. OpenTouch is the proof of concept; scaling it to 100+ hours with robot action labels would be a significant contribution.

---

### Gap 5: Insufficient Language Grounding

**Why it matters:** VLA (Vision-Language-Action) models are the dominant paradigm for generalist robot policies. Training VLAs requires language instructions paired with manipulation demonstrations. No dexterous hand dataset provides language annotations at scale.

**What partially addresses it:**
- OpenTouch (2024): LLM-generated high-level descriptions. Only 5.1 hours, no robot actions.
- HOI4D: Action labels (not natural language). Recognition-focused.
- TACO: Tool-action-object labels (structured, not natural language).

**What is still missing:** A dexterous manipulation dataset with natural language task descriptions + step-level instructions + robot action labels. This would enable training VLAs for dexterous manipulation — a completely open problem.

---

### Gap 6: Scene Diversity

**Why it matters:** Policies trained on lab-only data fail to generalize to new environments. Scene diversity is essential for learning robust visual representations and generalizable manipulation skills.

**What partially addresses it:**
- HOI4D: 610 rooms — the most scene-diverse egocentric hand dataset.
- DexWild: 93 environments — the most scene-diverse dexterous manipulation dataset.
- DROID: 564 scenes — but parallel gripper, not dexterous hand.

**What is still missing:** A dexterous hand dataset with 100+ diverse scenes (kitchens, workshops, offices, outdoor) and robot action labels. DexWild is the closest but covers only 5 tasks.

---

### Gap 7: Mismatch Between Human-Hand Signals and Robot-Hand Control Requirements

**Why it matters:** Human hands and robot hands have different kinematics, joint limits, and contact properties. Naive retargeting of human hand poses to robot joint states produces physically infeasible configurations. This mismatch is a fundamental challenge for human-to-robot transfer.

**What partially addresses it:**
- DexCap (2024): IK-based retargeting with physics validation.
- ManipTrans (2025): Residual learning for fine-grained retargeting.
- AnyTeleop (2023): Real-time retargeting system for multiple robot hands.
- DexMV (2022): Simulation-based retargeting.

**What is still missing:** A systematic study of retargeting quality across different robot hands and task types, with ground-truth robot execution data to validate retargeting accuracy. Most retargeting papers report success rates on a small set of tasks without systematic evaluation.

---

### Gap 8: Lack of Cross-Embodiment Evaluation

**Why it matters:** A dataset that only works for one robot hand has limited impact. Cross-embodiment generalization — training on one hand and evaluating on another — is essential for the field to converge on shared representations.

**What partially addresses it:**
- GenDexGrasp (2023): Multi-hand grasp synthesis (Shadow, Barrett, Allegro).
- GraspXL (2024): Multi-hand grasping (Shadow, Allegro, custom).
- AnyDexGrasp (2025): Real-world grasping across 3 robot hands.
- ManipTrans (2025): Multi-embodiment manipulation transfer.

**What is still missing:** A manipulation (not just grasping) benchmark with standardized cross-embodiment evaluation protocols. No dataset provides train/test splits across different robot hands for sequential manipulation tasks.


---

## 7. Design Recommendations for a New Dataset

### 7.1 Target Problem Statement

**Recommended framing:** "A large-scale egocentric dexterous manipulation dataset for learning generalizable dexterous robot policies, covering sequential multi-step tasks across diverse real-world scenes, with paired human hand demonstrations, retargeted high-DoF robot joint states, and language annotations."

This framing positions the dataset at the intersection of three underserved needs: (1) egocentric capture for robot-centric observations, (2) high-DoF robot action labels for policy learning, and (3) sequential task structure for long-horizon reasoning.

---

### 7.2 Data Collection Setup

**Recommended hardware stack:**
- **Egocentric camera:** Head-mounted RGB-D camera (e.g., Intel RealSense D435i or Azure Kinect) + fisheye/wide-angle lens for full hand visibility. Alternatively, a wrist-mounted stereo camera pair (as in DexWild) for near-egocentric capture.
- **Hand tracking:** Wearable EM + SLAM mocap (as in DexCap) for occlusion-resistant full-hand tracking. Alternatively, multi-camera hand pose estimation (as in EgoDex) for markerless tracking.
- **Robot hand:** LEAP Hand (16 DoF, ~$2,000, open-source) or Inspire Hand (16 DoF) for retargeted robot data. Shadow Hand (24 DoF) for maximum DoF but higher cost.
- **Optional tactile:** Tactile glove (e.g., custom capacitive or resistive array) for contact supervision on a subset of data.
- **Language annotation:** Post-hoc LLM-assisted annotation (as in OpenTouch) + human verification for task descriptions and step-level instructions.

**Collection protocol:**
- Human demonstrators wear the egocentric camera + hand tracking system.
- Demonstrations are collected in diverse real-world environments (not a fixed lab).
- Each demonstration covers a complete sequential task (not just a single grasp).
- Robot retargeting is performed offline using IK + physics validation.

---

### 7.3 Camera Placement

**Primary recommendation:** Head-mounted egocentric camera (forward-facing, ~30° downward tilt) to capture the natural first-person view of manipulation. This matches the perspective of a robot with a head-mounted camera.

**Secondary recommendation:** Wrist-mounted camera (as in DexWild) as a complementary view. The wrist camera provides a close-up view of the hand-object interaction that is often occluded in the head-mounted view.

**Optional:** Fixed external camera for ground-truth object pose estimation and retargeting validation. This does not need to be part of the released dataset but is useful for annotation.

**Key design principle:** The primary observation modality should match what a real robot would see. If the target robot has a head-mounted camera, use head-mounted. If it has a wrist camera, use wrist-mounted. Avoid exocentric-only capture.

---

### 7.4 Egocentric Capture Design

**Lessons from EgoDex:** Use a calibrated multi-camera rig (not a single camera) for robust 3D hand tracking. SLAM-based camera pose estimation enables 3D scene reconstruction. Full finger tracking (not just wrist pose) is essential.

**Lessons from DexWild:** Wrist-mounted cameras are more portable and less intrusive than head-mounted rigs. They enable data collection in truly diverse environments. The trade-off is a more limited field of view.

**Recommended design:** Head-mounted RGB-D (primary) + wrist-mounted RGB (secondary). The head-mounted camera provides the robot-centric view; the wrist camera provides close-up hand-object contact detail.

**Calibration:** Careful extrinsic calibration between head camera, wrist camera, and hand tracking system is essential for accurate 3D annotations.

---

### 7.5 Tasks to Include

**Tier 1 (must-have):**
- Grasping (diverse objects, diverse grasp types)
- Pick-and-place (with regrasp)
- In-hand reorientation / manipulation
- Tool use (hammer, screwdriver, knife, spoon, pen)
- Articulated object manipulation (drawer, door, bottle cap, scissors)

**Tier 2 (strongly recommended):**
- Long-horizon sequential tasks (e.g., "prepare a sandwich": open bag → take bread → spread → close bag)
- Bimanual coordination (one hand holds, other manipulates)
- Handover (object transfer between hands or to another person)
- Deformable object manipulation (cloth folding, bag opening)

**Tier 3 (optional/future):**
- Fine assembly (LEGO, connector insertion)
- Tactile-dependent tasks (identifying objects by touch)
- Language-conditioned tasks (follow verbal instructions)

**Minimum viable task set for a strong paper:** 20+ distinct task types across Tier 1 and Tier 2, with at least 3 long-horizon tasks (5+ steps each).

---

### 7.6 Scene Diversity

**Target:** 100+ distinct environments across at least 5 scene categories:
- Kitchen (countertop, sink, stove)
- Workshop / garage (workbench, tools)
- Office (desk, computer peripherals)
- Living room (table, shelves)
- Outdoor / unstructured (garden, street)

**Minimum viable:** 50+ environments across 3+ scene categories. This would already exceed all existing dexterous hand datasets except HOI4D (610 rooms, but recognition-only) and DexWild (93 environments, but 5 tasks only).

**Lighting diversity:** Collect in varied lighting conditions (bright, dim, mixed, natural light) to improve robustness.

---

### 7.7 Object Taxonomy

**Recommended taxonomy (based on TACO + OakInk2 + HOI4D):**

| Category | Examples | Grasp type | Manipulation type |
|----------|----------|------------|-------------------|
| Rigid containers | Bottle, cup, box, can | Power grasp | Pour, open, close |
| Tools | Hammer, screwdriver, knife, pen | Precision/power | Strike, turn, cut, write |
| Articulated objects | Scissors, drawer, door, laptop | Pinch/power | Open, close, slide |
| Deformable | Cloth, bag, sponge | Whole-hand | Fold, squeeze, wipe |
| Small precision | Coin, key, bolt, USB | Pinch | Insert, turn, pick |
| Food | Apple, bread, egg | Whole-hand | Pick, peel, cut |
| Electronics | Phone, remote, controller | Power/precision | Press, swipe, hold |

**Target:** 200+ distinct object instances across 20+ categories. This would be competitive with DexGraspNet (5,355 objects) for grasping and far exceed existing sequential manipulation datasets.

---

### 7.8 Grasp Taxonomy

Use the Feix et al. (2016) GRASP taxonomy as the foundation:
- **Power grasps:** Cylindrical, spherical, lateral, hook
- **Precision grasps:** Pinch (2-finger), tripod (3-finger), quadpod (4-finger)
- **Intermediate grasps:** Palmar, writing tripod

Annotate each grasp in the dataset with its grasp type. This enables analysis of grasp diversity and training of grasp-type-conditioned policies.

---

### 7.9 Manipulation Taxonomy

Use a hierarchical taxonomy:
- **Level 1 (primitive actions):** Grasp, release, push, pull, rotate, translate, pinch, squeeze
- **Level 2 (manipulation skills):** Pick-and-place, regrasp, in-hand reorientation, tool use, articulated object manipulation
- **Level 3 (task goals):** Pour liquid, open container, assemble parts, fold cloth, write text

Annotate each demonstration at all three levels. This enables training at multiple levels of abstraction and supports language-conditioned policies.

---

### 7.10 Whether to Include Robot Data, Human Data, or Both

**Recommendation: Both, with human data as the primary source.**

Human data is cheaper to collect, more natural, and more diverse. Robot data provides ground-truth action labels but is expensive and slow to collect.

**Proposed split:**
- 80% human demonstrations (egocentric, with hand tracking)
- 20% robot demonstrations (teleoperated, with direct joint state recording)

The human data provides scale and diversity. The robot data provides ground-truth action labels for policy learning evaluation. The co-training framework from DexWild (mixing human + robot data) should be adopted.

---

### 7.11 Whether to Include Retargeted Actions

**Yes, strongly recommended.** Retargeted robot joint states should be a first-class annotation for all human demonstrations. This is what makes the dataset useful for robot policy learning.

**Recommended retargeting pipeline:**
1. Collect human hand tracking (MANO parameters or joint angles)
2. Apply IK-based retargeting to target robot hand (LEAP Hand or Shadow Hand)
3. Validate retargeted poses with physics simulation (check for penetration, joint limit violations)
4. Store both human hand parameters and retargeted robot joint states

**Multiple robot targets:** Retarget to at least 2 robot hands (e.g., LEAP Hand 16 DoF + Shadow Hand 24 DoF) to enable cross-embodiment evaluation.

---

### 7.12 Annotation Design

**Required annotations (every demonstration):**
- Egocentric RGB-D frames (synchronized)
- 3D hand pose (MANO parameters or joint angles, per frame)
- Retargeted robot joint states (per frame, for each target robot hand)
- Camera pose (SLAM-based, per frame)
- Task label (from manipulation taxonomy Level 2)
- Object category label

**Strongly recommended annotations:**
- 6D object pose (per frame, for key objects)
- Language task description (natural language, per demonstration)
- Language step annotations (per sub-task step)
- Grasp type label (from grasp taxonomy)
- Contact map (per frame, from physics simulation or tactile sensor)

**Optional annotations:**
- Object mesh / part labels
- Affordance labels (where to grasp, where to push)
- Failure annotations (demonstrations that failed and why)

---

### 7.13 Benchmark Protocol Design

**Proposed benchmark structure (4 evaluation suites):**

**Suite 1: Cross-Object Generalization**
- Train on seen objects, test on unseen objects within same category
- Metric: Task success rate on unseen objects

**Suite 2: Cross-Scene Generalization**
- Train on seen environments, test on unseen environments
- Metric: Task success rate in unseen scenes

**Suite 3: Cross-Task Generalization**
- Train on seen task types, test on unseen task types
- Metric: Task success rate on unseen tasks (zero-shot and few-shot)

**Suite 4: Cross-Embodiment Generalization**
- Train on one robot hand, test on another
- Metric: Task success rate with different robot hand

**Baseline methods to include:**
- BC (Behavioral Cloning) with ResNet/ViT backbone
- ACT (Action Chunking with Transformers)
- Diffusion Policy
- VLA fine-tuning (e.g., OpenVLA or π0)
- DexCap's DexIL (point cloud-based IL)

---

### 7.14 Train/Val/Test Split Strategy

**Recommended splits:**

| Split | Criteria | Size |
|-------|----------|------|
| Train | Seen objects, seen scenes, seen tasks | 70% |
| Val | Seen objects, seen scenes, seen tasks (held-out demos) | 10% |
| Test-Object | Unseen object instances (same categories) | 5% |
| Test-Scene | Unseen environments | 5% |
| Test-Task | Unseen task types | 5% |
| Test-Embodiment | Different robot hand | 5% |

**Key principle:** Never leak test environments or objects into training. Use object-level, scene-level, and task-level splits, not random splits.

---

### 7.15 Metrics to Report

**Primary metrics:**
- Task success rate (binary: did the task complete successfully?)
- Subtask success rate (for long-horizon tasks: how many steps completed?)

**Secondary metrics:**
- Grasp success rate (did the grasp succeed?)
- Object displacement error (for placement tasks)
- Trajectory smoothness (joint velocity variance)
- Retargeting fidelity (MPJPE between human and retargeted robot hand)

**Generalization metrics:**
- Cross-object success rate (on unseen objects)
- Cross-scene success rate (in unseen environments)
- Cross-task success rate (on unseen task types)
- Cross-embodiment success rate (with different robot hand)

---

### 7.16 What Would Make the Dataset Genuinely Novel and Publishable

The dataset would be novel if it is the **first** to combine:

1. **Egocentric capture** (head-mounted or wrist-mounted) — differentiates from DexCap, DexCanvas, GR-Dexter
2. **High-DoF robot action labels** (retargeted joint states) — differentiates from EgoDex, HOI4D, TACO
3. **Sequential multi-step tasks** (5+ steps, 20+ task types) — differentiates from DexGraspNet, UniDexGrasp
4. **Scene diversity** (100+ environments) — differentiates from all existing dexterous datasets except HOI4D
5. **Language annotations** — differentiates from all existing dexterous datasets
6. **Standardized benchmark** with cross-object/scene/task/embodiment splits — differentiates from all existing work

No existing dataset has all six. A dataset with even four of these six properties would be a strong CVPR/NeurIPS/RSS paper.


---

## 8. Ranked Shortlist

### Top 10 Most Relevant Existing Datasets

| Rank | Dataset | Year | Why It Matters |
|------|---------|------|----------------|
| 1 | **EgoDex** | 2025 | The closest existing work to your goal: 829 hrs, egocentric, 194 tasks, full finger tracking. Your dataset must differentiate by adding robot action labels and language. |
| 2 | **DexWild** | 2025 | Best scene diversity (93 envs) + robot action labels + co-training framework. Your dataset should match or exceed its scene diversity while adding head-mounted egocentric and more tasks. |
| 3 | **DexCap** | 2024 | Best human-to-robot retargeting pipeline (EM+SLAM mocap → LEAP Hand). The technical foundation for your retargeting pipeline. |
| 4 | **TACO** | 2024 | Best tool-use taxonomy + egocentric+exocentric paired views + bimanual. Defines the annotation standard for tool use tasks. |
| 5 | **OakInk2** | 2024 | Best task hierarchy (affordance→primitive→complex task) + bimanual. Defines the annotation standard for complex sequential tasks. |
| 6 | **HOI4D** | 2022 | Most scene-diverse egocentric dataset (610 rooms). Defines the bar for scene diversity. Your dataset should target similar or greater diversity. |
| 7 | **ARCTIC** | 2023 | Best articulated object manipulation data + dynamic contact maps + egocentric+exocentric. Essential reference for articulated object task design. |
| 8 | **HumanPlus** | 2024 | Only existing egocentric + real robot dexterous manipulation dataset. Proof of concept for your approach; differentiate by scale and task diversity. |
| 9 | **OpenTouch** | 2024 | Only egocentric tactile dataset. If you include tactile, this is the reference. |
| 10 | **DexMimicGen** | 2024 | Best data augmentation pipeline for dexterous manipulation. Use its approach to scale up from a small real-world seed dataset. |

---

### Top 10 Most Relevant Existing Benchmarks

| Rank | Benchmark | Year | Why It Matters |
|------|-----------|------|----------------|
| 1 | **Bi-DexHands** | 2022 | Defines the task taxonomy for bimanual dexterous manipulation. Use its 20 tasks as a reference for your task design. |
| 2 | **DexArt** | 2023 | Best benchmark for articulated object manipulation with dexterous hands. Your benchmark should include articulated objects. |
| 3 | **HOI4D Benchmark** | 2022 | Best egocentric benchmark for category-level HOI. Reference for egocentric evaluation protocol design. |
| 4 | **TACO Benchmark** | 2024 | Best benchmark for tool-use generalization. Reference for tool-use evaluation design. |
| 5 | **OakInk2 Benchmark** | 2024 | Best benchmark for complex task hierarchy evaluation. Reference for long-horizon evaluation design. |
| 6 | **UniDexGrasp** | 2023 | Standard benchmark for universal dexterous grasping. Your dataset should include a grasping evaluation that supersedes this. |
| 7 | **AnyDexGrasp** | 2025 | Best real-world cross-embodiment grasping benchmark. Reference for cross-embodiment evaluation design. |
| 8 | **HumanoidBench** | 2024 | Best benchmark for humanoid dexterous manipulation. Reference for whole-body + dexterous evaluation. |
| 9 | **ARCTIC Benchmark** | 2023 | Best benchmark for bimanual articulated object manipulation. Reference for contact-rich evaluation. |
| 10 | **DexYCB Benchmark** | 2021 | Standard benchmark for hand/object pose estimation. Reference for hand pose evaluation. |

---

### Top 5 Most Important Papers to Read First

| Rank | Paper | Why Read First |
|------|-------|----------------|
| 1 | **EgoDex (2025)** [arxiv:2505.11709](https://arxiv.org/abs/2505.11709) | Your most direct competitor. Read to understand exactly what it does and does not provide, so you can position your dataset clearly. |
| 2 | **DexCap (2024)** [arxiv:2403.07788](https://arxiv.org/abs/2403.07788) | The technical blueprint for your data collection system. Read to understand the mocap + retargeting + IL pipeline in detail. |
| 3 | **DexWild (2025)** [arxiv:2505.07813](https://arxiv.org/abs/2505.07813) | The best existing work on in-the-wild dexterous data collection. Read to understand the co-training framework and scene diversity approach. |
| 4 | **OakInk2 (2024)** [arxiv:2403.19417](https://arxiv.org/abs/2403.19417) | The best annotation design for complex sequential tasks. Read to understand the three-level task hierarchy and how to annotate long-horizon manipulation. |
| 5 | **HumanPlus (2024)** [arxiv:2406.10454](https://arxiv.org/abs/2406.10454) | The only existing egocentric + real robot dexterous manipulation system. Read to understand the shadowing pipeline and egocentric policy learning approach. |

---

## 9. Final Recommendation

### 9.1 Most Promising Positioning

**The clearest gap in the field is:** an egocentric, real-world, sequential dexterous manipulation dataset with high-DoF robot action labels, scene diversity, and language annotations.

**Recommended positioning:** "EgoDexter" or similar — a dataset that is to dexterous robot manipulation what DROID is to parallel-gripper manipulation, but with egocentric capture and high-DoF hand control.

Specifically:
- **Scale target:** 500–1,000 hours of egocentric video (competitive with EgoDex's 829 hrs)
- **Task target:** 30+ sequential task types (far exceeding DexWild's 5 tasks)
- **Scene target:** 100+ environments (competitive with HOI4D's 610 rooms, far exceeding DexWild's 93)
- **Action target:** Retargeted LEAP Hand (16 DoF) + Shadow Hand (24 DoF) joint states
- **Language target:** Natural language task descriptions + step-level instructions for all demonstrations
- **Benchmark target:** 4 evaluation suites (cross-object, cross-scene, cross-task, cross-embodiment)

This positioning is differentiated from every existing dataset and directly addresses the needs of the VLA/policy learning community.

---

### 9.2 What to Avoid (Already Saturated)

**Avoid these directions — they are already well-covered:**

1. **Dexterous grasp-pose datasets.** DexGraspNet 2.0 (427M grasps), DexGrasp Anything (3.4M grasps, 15K objects), Get a Grip (3.5M grasps) have saturated this space. A new grasp dataset would need to be dramatically different (e.g., real-world, egocentric, with contact) to be publishable.

2. **Simulation-only dexterous manipulation benchmarks.** Bi-DexHands, DexArt, UniDexGrasp, HumanoidBench cover this well. The community is moving away from sim-only benchmarks.

3. **Hand pose estimation datasets.** DexYCB, GRAB, OakInk, WildHands cover this. Hand pose estimation is a mature field; a new dataset needs to go beyond pose to manipulation.

4. **Single-task in-hand manipulation.** DIME, DexMV cover in-hand manipulation in simulation. This is not a gap.

5. **Bimanual grasping datasets.** BimanGrasp (150K+ grasps) covers this in simulation.

---

### 9.3 Strongest Combination for a Novel Contribution

**The winning combination:**

| Dimension | Choice | Rationale |
|-----------|--------|-----------|
| Scene | 100+ real-world environments (kitchen, workshop, office, outdoor) | Exceeds all existing dexterous datasets; matches DROID's scene diversity |
| Tasks | 30+ sequential tasks: grasping, pick-place, regrasp, tool use, articulated objects, bimanual, long-horizon | Covers all task families; no existing dataset covers all |
| Annotations | Egocentric RGB-D + MANO hand pose + retargeted LEAP/Shadow joint states + language + contact maps | First dataset with all five annotation types |
| Benchmark | 4 suites: cross-object, cross-scene, cross-task, cross-embodiment | First real-world dexterous benchmark with all four generalization axes |
| Baselines | BC, ACT, Diffusion Policy, VLA fine-tuning, DexIL | Covers the dominant policy learning paradigms |
| Metrics | Task success rate, subtask success rate, cross-generalization rates | Standard + novel metrics |

**The key novelty claim:** "The first egocentric dexterous manipulation dataset with high-DoF robot action labels, sequential task structure, scene diversity, and language annotations, paired with a benchmark that evaluates cross-object, cross-scene, cross-task, and cross-embodiment generalization."

This claim is defensible against all existing work as of March 2025. EgoDex (2025) is the closest competitor but lacks robot action labels, language, and a benchmark. DexWild (2025) is the second closest but lacks head-mounted egocentric, task diversity, and language.

**If you move fast (submit by late 2025), this positioning is open.** If EgoDex or DexWild release follow-up work with robot action labels, you will need to differentiate further on scale, task diversity, or language grounding.

---

*End of Report*

---

**Appendix: Key URLs**

| Dataset | Paper | Project | Data |
|---------|-------|---------|------|
| DexGraspNet | [arxiv:2210.02697](https://arxiv.org/abs/2210.02697) | [pku-epic.github.io/DexGraspNet](https://pku-epic.github.io/DexGraspNet/) | same |
| DexGraspNet 2.0 | [arxiv:2410.23004](https://arxiv.org/abs/2410.23004) | [pku-epic.github.io/DexGraspNet2.0](https://pku-epic.github.io/DexGraspNet2.0/) | same |
| DexYCB | [arxiv:2104.04631](https://arxiv.org/abs/2104.04631) | [dex-ycb.github.io](https://dex-ycb.github.io/) | same |
| OakInk2 | [arxiv:2403.19417](https://arxiv.org/abs/2403.19417) | [oakink.net/v2](https://oakink.net/v2/) | same |
| HOI4D | [arxiv:2203.01577](https://arxiv.org/abs/2203.01577) | [hoi4d.github.io](https://hoi4d.github.io/) | same |
| ARCTIC | [arxiv:2204.13662](https://arxiv.org/abs/2204.13662) | [arctic.is.tue.mpg.de](https://arctic.is.tue.mpg.de) | same |
| TACO | [arxiv:2401.08399](https://arxiv.org/abs/2401.08399) | [taco2024.github.io](https://taco2024.github.io) | same |
| EgoDex | [arxiv:2505.11709](https://arxiv.org/abs/2505.11709) | — | — |
| DexWild | [arxiv:2505.07813](https://arxiv.org/abs/2505.07813) | [dexwild.github.io](https://dexwild.github.io/) | [HuggingFace](https://huggingface.co/datasets/boardd/dexwild-dataset) |
| DexCap | [arxiv:2403.07788](https://arxiv.org/abs/2403.07788) | [dex-cap.github.io](https://dex-cap.github.io/) | same |
| HumanPlus | [arxiv:2406.10454](https://arxiv.org/abs/2406.10454) | — | — |
| DexCanvas | [arxiv:2510.15786](https://arxiv.org/abs/2510.15786) | [dexcanvas.github.io](https://dexcanvas.github.io/) | [HuggingFace](https://huggingface.co/datasets/Manggu/DexCanvas) |
| OpenTouch | [arxiv:2512.16842](https://arxiv.org/abs/2512.16842) | [opentouch-tactile.github.io](https://opentouch-tactile.github.io/) | — |
| Bi-DexHands | [arxiv:2206.08686](https://arxiv.org/abs/2206.08686) | [pku-marl.github.io/DexterousHands](https://pku-marl.github.io/DexterousHands/) | same |
| DexArt | [arxiv:2305.05706](https://arxiv.org/abs/2305.05706) | — | [github](https://github.com/Kami-code/dexart-release) |
| DexMimicGen | [arxiv:2410.24185](https://arxiv.org/abs/2410.24185) | [dexmimicgen.github.io](https://dexmimicgen.github.io/) | same |
| HumanoidBench | [arxiv:2403.10506](https://arxiv.org/abs/2403.10506) | [humanoid-bench.github.io](https://humanoid-bench.github.io/) | same |
| ManipTrans | [arxiv:2503.21860](https://arxiv.org/abs/2503.21860) | [maniptrans.github.io](https://maniptrans.github.io/) | same |
| GR-Dexter | [arxiv:2512.24210](https://arxiv.org/abs/2512.24210) | — | — |
| GraspXL | [arxiv:2403.19649](https://arxiv.org/abs/2403.19649) | [eth-ait.github.io/graspxl](https://eth-ait.github.io/graspxl/) | same |
| AnyDexGrasp | [arxiv:2502.16420](https://arxiv.org/abs/2502.16420) | [graspnet.net/anydexgrasp](https://graspnet.net/anydexgrasp/) | same |
| OpenEgo | [arxiv:2509.05513](https://arxiv.org/abs/2509.05513) | — | — |

