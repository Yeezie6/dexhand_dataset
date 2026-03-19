# Recommended Dataset Design Blueprint
## Egocentric Dexterous Manipulation Dataset (Working Title: "EgoDexter")

**Version:** 1.0 | **Date:** March 2025

---

## 1. One-Line Pitch

> The first egocentric, real-world dexterous manipulation dataset with high-DoF robot action labels, sequential task structure, scene diversity, language annotations, and a four-axis generalization benchmark.

---

## 2. Differentiation from Existing Work

| Property | EgoDex (2025) | DexWild (2025) | DexCap (2024) | **This Dataset** |
|----------|--------------|----------------|---------------|-----------------|
| Egocentric (head-mounted) | ✅ | ❌ (wrist) | ❌ | ✅ |
| High-DoF robot action labels | ❌ | ✅ (partial) | ✅ | ✅ |
| Sequential tasks (20+) | ✅ (194 tasks) | ❌ (5 tasks) | ✅ (~10) | ✅ (30+ tasks) |
| Scene diversity (100+) | unclear | ✅ (93 envs) | ❌ (lab) | ✅ (100+ envs) |
| Language annotations | ❌ | ❌ | ❌ | ✅ |
| Standardized benchmark | ❌ | ❌ | ❌ | ✅ (4 suites) |
| Scale (hours) | 829 hrs | unclear | unclear | 500+ hrs |

---

## 3. Hardware Stack

### 3.1 Egocentric Capture
- **Primary camera:** Intel RealSense D435i (RGB-D, 30fps, 848×480 depth) mounted on lightweight head rig
- **Secondary camera:** Wrist-mounted stereo RGB pair (e.g., ZED Mini or custom) for close-up hand-object view
- **Synchronization:** Hardware trigger sync between head and wrist cameras

### 3.2 Hand Tracking
- **Primary:** Wearable EM + SLAM mocap (as in DexCap) — occlusion-resistant, full finger tracking
- **Backup:** Multi-camera markerless hand pose estimation (MediaPipe or MANO-fitting) for environments where wearable is impractical
- **Output:** MANO parameters (shape β, pose θ) at 30fps

### 3.3 Robot Hand (for retargeting)
- **Primary target:** LEAP Hand (16 DoF, ~$2,000, open-source hardware)
- **Secondary target:** Shadow Hand (24 DoF) — for maximum DoF coverage
- **Retargeting:** IK-based with physics validation (DexCap pipeline)

### 3.4 Optional Tactile
- Tactile glove (capacitive array, 32+ sensors per hand) on 20% of demonstrations
- Synchronized with egocentric camera via hardware trigger

---

## 4. Data Collection Protocol

### 4.1 Demonstrators
- 10–20 human demonstrators (diverse hand sizes, dominant hands)
- Each demonstrator trained on task protocol (30-min training session)
- Each demonstrator collects data across multiple environments

### 4.2 Session Structure
- Each session: 2–4 hours in one environment
- Each session covers 5–10 task types
- Each task type: 10–20 demonstrations per session
- Total target: 500+ hours across 100+ environments

### 4.3 Quality Control
- Real-time monitoring of hand tracking quality (reject frames with tracking loss >10%)
- Post-hoc physics validation of retargeted robot poses
- Human annotation review for task labels and language descriptions

---

## 5. Task Taxonomy (30 Tasks Minimum)

### Tier 1: Grasping (5 tasks)
1. Power grasp of rigid object (bottle, cup, ball)
2. Precision grasp of small object (coin, key, pen)
3. Pinch grasp of flat object (card, paper, lid)
4. Lateral grasp (knife, spatula)
5. Whole-hand grasp of deformable (cloth, bag)

### Tier 2: Pick-and-Place (5 tasks)
6. Pick and place rigid object (cup to shelf)
7. Pick and stack (stacking cups/blocks)
8. Pick and insert (USB, plug, key in lock)
9. Pick and pour (liquid from bottle to cup)
10. Pick and handover (pass object to other hand)

### Tier 3: In-Hand Manipulation (5 tasks)
11. Object reorientation (rotate object in hand)
12. Finger gaiting (walk fingers along object)
13. Pen spinning / object spinning
14. Card flipping
15. Coin rolling

### Tier 4: Tool Use (5 tasks)
16. Hammer (strike nail)
17. Screwdriver (turn screw)
18. Knife (cut food)
19. Spoon (scoop and transfer)
20. Pen/marker (write or draw)

### Tier 5: Articulated Objects (5 tasks)
21. Open/close drawer
22. Open/close door (handle)
23. Open/close bottle cap
24. Open/close scissors
25. Open/close laptop

### Tier 6: Long-Horizon Sequential (5 tasks)
26. Table setting (place plate, cup, utensils)
27. Sandwich preparation (open bag, take bread, spread, close)
28. Tool assembly (pick parts, assemble, tighten)
29. Packing (pick objects, place in box, close box)
30. Cleaning (pick cloth, wipe surface, fold cloth)

---

## 6. Scene Taxonomy (100+ Environments)

| Category | Count | Examples |
|----------|-------|---------|
| Kitchen | 25 | Countertop, sink area, stove area, dining table |
| Workshop/Garage | 20 | Workbench, tool shelf, assembly table |
| Office/Desk | 20 | Computer desk, filing area, meeting table |
| Living Room | 15 | Coffee table, bookshelf, side table |
| Outdoor/Unstructured | 10 | Garden table, outdoor workbench, street |
| Lab/Controlled | 10 | Standard lab table (for baseline comparison) |

---

## 7. Annotation Schema

### 7.1 Per-Frame Annotations
```
frame_id: int
timestamp: float (seconds)
rgb_head: H×W×3 uint8
depth_head: H×W float32 (meters)
rgb_wrist: H×W×3 uint8
camera_pose_head: 4×4 float32 (world-to-camera SE3)
mano_shape: 10-dim float32 (β)
mano_pose: 48-dim float32 (θ, 16 joints × 3 axis-angle)
mano_trans: 3-dim float32 (global translation)
leap_joint_angles: 16-dim float32 (retargeted, radians)
shadow_joint_angles: 24-dim float32 (retargeted, radians)
retargeting_valid: bool (physics validation passed)
tactile_map: 32×32 float32 (optional, normalized 0-1)
```

### 7.2 Per-Demonstration Annotations
```
demo_id: str
scene_id: str
scene_category: str
demonstrator_id: str
task_level1: str (primitive action)
task_level2: str (manipulation skill)
task_level3: str (task goal)
language_description: str (natural language task description)
language_steps: list[str] (step-level instructions)
grasp_type: str (from Feix taxonomy)
object_categories: list[str]
object_ids: list[str]
duration_seconds: float
num_frames: int
success: bool
failure_reason: str (if not success)
```

### 7.3 Per-Object Annotations (key objects)
```
object_id: str
object_category: str
object_mesh_path: str (optional)
6d_pose_per_frame: list[4×4 float32]
affordance_labels: dict (grasp_point, push_point, etc.)
```

---

## 8. Benchmark Design

### Suite 1: Cross-Object Generalization
- **Train:** 80% of object instances per category
- **Test:** 20% held-out object instances (same categories, different instances)
- **Metric:** Task success rate on unseen object instances

### Suite 2: Cross-Scene Generalization
- **Train:** 80% of environments
- **Test:** 20% held-out environments (never seen during training)
- **Metric:** Task success rate in unseen environments

### Suite 3: Cross-Task Generalization
- **Train:** 24 of 30 task types
- **Test:** 6 held-out task types (zero-shot)
- **Few-shot variant:** 5 demonstrations of test task, then evaluate
- **Metric:** Task success rate on unseen task types

### Suite 4: Cross-Embodiment Generalization
- **Train:** LEAP Hand (16 DoF) demonstrations
- **Test:** Shadow Hand (24 DoF) execution (and vice versa)
- **Metric:** Task success rate with different robot hand

### Baseline Methods
| Method | Type | Reference |
|--------|------|-----------|
| BC-ResNet | Behavioral Cloning | Standard |
| BC-ViT | Behavioral Cloning | Standard |
| ACT | Action Chunking Transformer | [arxiv:2304.13705](https://arxiv.org/abs/2304.13705) |
| Diffusion Policy | Diffusion-based IL | [arxiv:2303.04137](https://arxiv.org/abs/2303.04137) |
| DexIL | Point cloud IL | DexCap (2024) |
| VLA fine-tune | Vision-Language-Action | OpenVLA or π0 |

---

## 9. Scale Targets

| Metric | Minimum Viable | Target | Stretch |
|--------|---------------|--------|---------|
| Total hours | 200 hrs | 500 hrs | 1,000 hrs |
| Environments | 50 | 100 | 200 |
| Task types | 20 | 30 | 50 |
| Demonstrators | 5 | 15 | 30 |
| Object instances | 100 | 300 | 500 |
| Demonstrations | 5,000 | 15,000 | 30,000 |
| With tactile | 500 demos | 2,000 demos | 5,000 demos |

---

## 10. Publication Strategy

### Target Venues
- **Primary:** CVPR 2026 or NeurIPS 2025 (dataset track)
- **Secondary:** RSS 2026, CoRL 2025

### Key Claims
1. First egocentric dexterous manipulation dataset with robot action labels
2. First dexterous dataset with language annotations at scale
3. First real-world benchmark with cross-object/scene/task/embodiment evaluation
4. State-of-the-art on all four generalization axes vs. baselines

### Differentiation Narrative
> "Existing egocentric datasets (EgoDex) lack robot action labels. Existing robot action datasets (DexCap, DexWild) lack egocentric capture and task diversity. We close this gap with [Dataset Name]: the first dataset combining egocentric capture, high-DoF robot action labels, 30+ sequential task types, 100+ diverse scenes, and language annotations, paired with a four-axis generalization benchmark."

---

## 11. Timeline (Estimated)

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Hardware setup + pilot | 2 months | Calibrated rig, retargeting pipeline |
| Pilot data collection (10 envs, 5 tasks) | 1 month | 20 hrs pilot data |
| Pilot annotation + benchmark design | 1 month | Annotation schema, benchmark protocol |
| Full data collection | 4 months | 500+ hrs across 100+ envs |
| Full annotation | 2 months | All annotations complete |
| Baseline experiments | 2 months | Benchmark results |
| Paper writing + submission | 1 month | CVPR/NeurIPS submission |

**Total: ~13 months from hardware setup to submission**

---

## 12. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| Hand tracking failures in cluttered scenes | Use EM mocap (occlusion-resistant) as primary; markerless as backup |
| Retargeting quality issues | Physics validation pipeline; reject invalid poses |
| Scene diversity logistics | Partner with multiple institutions; use portable rig |
| Annotation cost | LLM-assisted language annotation + human verification |
| Competitor releases similar dataset | Move fast; differentiate on language + benchmark |
| EgoDex releases robot action labels | Differentiate on scene diversity, language, benchmark design |
