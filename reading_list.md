# Priority Reading List: Dexterous Hand Datasets & Benchmarks

**For:** PI leading a new egocentric dexterous-hand dataset project
**Sorted by:** Priority (read in order)

---

## Tier 1: Read Immediately (Before Any Design Decisions)

### 1. EgoDex (2025) — Your Primary Competitor
- **Paper:** https://arxiv.org/abs/2505.11709
- **Why:** 829 hours of egocentric dexterous manipulation data with full finger tracking. This is the most direct competitor to your project. Read to understand exactly what it provides and what it lacks (robot action labels, language, benchmark).
- **Key questions to answer while reading:** What is their capture setup? How do they handle occlusion? What tasks do they cover? What annotations do they provide? What do they NOT provide?

### 2. DexCap (2024) — Your Technical Blueprint
- **Paper:** https://arxiv.org/abs/2403.07788
- **Project:** https://dex-cap.github.io/
- **Why:** The most complete human-to-robot retargeting pipeline for dexterous manipulation. EM + SLAM mocap → LEAP Hand retargeting → DexIL policy learning. This is the technical foundation you should build on.
- **Key questions:** How does the EM mocap work? What is the retargeting algorithm? How do they validate retargeting quality? What is the DexIL architecture?

### 3. DexWild (2025) — Scene Diversity Reference
- **Paper:** https://arxiv.org/abs/2505.07813
- **Project:** https://dexwild.github.io/
- **Data:** https://huggingface.co/datasets/boardd/dexwild-dataset
- **Why:** Best existing work on in-the-wild dexterous data collection (93 environments). The co-training framework (human + robot data) is a key design pattern. Read to understand how they achieve scene diversity and what their limitations are.
- **Key questions:** How is the portable rig designed? How does co-training work? Why only 5 tasks? What prevents scaling to more tasks?

### 4. HumanPlus (2024) — Egocentric Robot Dexterous Manipulation
- **Paper:** https://arxiv.org/abs/2406.10454
- **Why:** The only existing system that combines egocentric capture with real dexterous robot manipulation. The shadowing pipeline (human motion → robot motion in real time) is directly relevant. Read to understand the egocentric policy learning approach.
- **Key questions:** How does the shadowing pipeline work? What is the egocentric observation space? What tasks do they demonstrate? What are the failure modes?

### 5. OakInk2 (2024) — Task Hierarchy Design
- **Paper:** https://arxiv.org/abs/2403.19417
- **Project:** https://oakink.net/v2/
- **Why:** The best annotation design for complex sequential tasks. The three-level hierarchy (affordance → primitive action → complex task) is the model for your annotation schema. Read to understand how to structure task annotations.
- **Key questions:** How is the three-level hierarchy defined? How are affordances annotated? How are complex tasks decomposed? What is the annotation pipeline?

---

## Tier 2: Read Before Data Collection

### 6. TACO (2024) — Tool Use Taxonomy
- **Paper:** https://arxiv.org/abs/2401.08399
- **Project:** https://taco2024.github.io
- **Why:** The most principled tool-use taxonomy in the field. Covers ~100 tool-action-object combinations with egocentric + third-person views. Essential reference for designing your tool-use task set.

### 7. ARCTIC (2023) — Articulated Object Manipulation
- **Paper:** https://arxiv.org/abs/2204.13662
- **Project:** https://arctic.is.tue.mpg.de
- **Why:** Best dataset for bimanual articulated object manipulation with dynamic contact maps. Essential reference for designing your articulated object task set and contact annotation approach.

### 8. HOI4D (2022) — Scene Diversity Reference
- **Paper:** https://arxiv.org/abs/2203.01577
- **Project:** https://hoi4d.github.io/
- **Why:** Most scene-diverse egocentric hand dataset (610 rooms). Read to understand how they achieved scene diversity and what their annotation pipeline looks like.

### 9. ManipTrans / DexManipNet (2025) — Fine-Grained Retargeting
- **Paper:** https://arxiv.org/abs/2503.21860
- **Project:** https://maniptrans.github.io/
- **Why:** Residual learning approach for fine-grained human-to-robot retargeting. Enables tasks (pen capping, bottle unscrewing) that are too precise for direct IK retargeting. Read before finalizing your retargeting pipeline.

### 10. DexMimicGen (2024) — Data Augmentation
- **Paper:** https://arxiv.org/abs/2410.24185
- **Project:** https://dexmimicgen.github.io/
- **Why:** Best data augmentation pipeline for dexterous manipulation. Shows how to scale from a small real-world seed dataset to large synthetic datasets. Read to understand how to augment your real-world data.

---

## Tier 3: Read Before Benchmark Design

### 11. Bi-DexHands (2022) — Bimanual Task Taxonomy
- **Paper:** https://arxiv.org/abs/2206.08686
- **Project:** https://pku-marl.github.io/DexterousHands/
- **Why:** Defines the task taxonomy for bimanual dexterous manipulation (20 tasks). Use as reference for your bimanual task design.

### 12. DexArt (2023) — Articulated Object Benchmark
- **Paper:** https://arxiv.org/abs/2305.05706
- **Code:** https://github.com/Kami-code/dexart-release
- **Why:** Best simulation benchmark for articulated object manipulation with dexterous hands. Reference for your articulated object benchmark design.

### 13. AnyDexGrasp (2025) — Cross-Embodiment Grasping
- **Paper:** https://arxiv.org/abs/2502.16420
- **Project:** https://graspnet.net/anydexgrasp/
- **Why:** Best real-world cross-embodiment grasping benchmark. Reference for your cross-embodiment evaluation design.

### 14. GR-Dexter (2024) — VLA for Dexterous Manipulation
- **Paper:** https://arxiv.org/abs/2512.24210
- **Why:** Full-stack VLA system for bimanual dexterous manipulation. Shows how to train VLAs on dexterous manipulation data. Read to understand what your dataset needs to support VLA training.

### 15. OpenTouch (2024) — Egocentric Tactile
- **Paper:** https://arxiv.org/abs/2512.16842
- **Project:** https://opentouch-tactile.github.io/
- **Why:** First egocentric full-hand tactile dataset. If you plan to include tactile sensing, this is the reference for hardware design and annotation.

---

## Tier 4: Background Reading (Read When Time Permits)

### 16. DexGraspNet 2.0 (2024)
- **Paper:** https://arxiv.org/abs/2410.23004
- **Why:** Defines the state of the art for dexterous grasp synthesis. Understand what grasp-pose datasets look like so you can clearly differentiate your manipulation dataset.

### 17. UniDexGrasp (2023)
- **Paper:** https://arxiv.org/abs/2303.00938
- **Why:** Standard benchmark for universal dexterous grasping. Your dataset should include a grasping evaluation that supersedes this.

### 18. DexMV (2022)
- **Paper:** https://arxiv.org/abs/2108.05877
- **Why:** Foundational work on human-to-robot retargeting for dexterous manipulation. Historical context for DexCap and DexWild.

### 19. HumanoidBench (2024)
- **Paper:** https://arxiv.org/abs/2403.10506
- **Why:** Best benchmark for humanoid dexterous manipulation. Reference for whole-body + dexterous evaluation design.

### 20. DexCanvas (2025)
- **Paper:** https://arxiv.org/abs/2510.15786
- **Data:** https://huggingface.co/datasets/Manggu/DexCanvas
- **Why:** Large-scale (7,000 hrs) hybrid real-synthetic dataset. Understand their skill taxonomy and physics-validated contact annotation approach.

### 21. OpenEgo (2025)
- **Paper:** https://arxiv.org/abs/2509.05513
- **Why:** Aggregates 6 egocentric datasets with standardized annotations. Understand what standardization looks like across datasets.

### 22. AssemblyHands (2023)
- **Paper:** https://arxiv.org/abs/2304.12301
- **Why:** Egocentric assembly task dataset. Reference for fine assembly task design.

### 23. GraspXL (2024)
- **Paper:** https://arxiv.org/abs/2403.19649
- **Why:** Multi-hand grasping across 500K+ objects. Reference for cross-embodiment grasp evaluation.

### 24. DexSim2Real² (2024)
- **Paper:** https://arxiv.org/abs/2409.08750
- **Why:** Goal-conditioned articulated object manipulation with sim-to-real transfer. Reference for articulated object manipulation pipeline.

### 25. AnyTeleop (2023)
- **Paper:** https://arxiv.org/abs/2307.04577
- **Why:** Multi-robot teleoperation system. Reference for real-time hand tracking and retargeting.

---

## Key Methods Papers (Not Datasets, But Essential Background)

### Policy Learning
- **ACT:** https://arxiv.org/abs/2304.13705 — Action Chunking with Transformers (standard baseline)
- **Diffusion Policy:** https://arxiv.org/abs/2303.04137 — Diffusion-based imitation learning (standard baseline)
- **π0:** https://arxiv.org/abs/2410.24164 — Flow matching VLA (state-of-the-art VLA)
- **OpenVLA:** https://arxiv.org/abs/2406.09246 — Open-source VLA (fine-tuning baseline)

### Hand Pose Estimation
- **MANO:** https://arxiv.org/abs/1612.02450 — Parametric hand model (standard representation)
- **FrankMocap:** https://arxiv.org/abs/2108.06428 — Real-time hand+body mocap

### Retargeting
- **DexPilot:** https://arxiv.org/abs/1910.03135 — Vision-based teleoperation (Allegro Hand)
- **AnyTeleop:** https://arxiv.org/abs/2307.04577 — Multi-robot teleoperation

---

*Total: 25 primary papers + 8 methods papers. Estimated reading time: 3–4 weeks for thorough reading.*
