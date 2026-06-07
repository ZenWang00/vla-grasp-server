# System Diagrams

## 时序图 (Sequence Diagram)

```mermaid
sequenceDiagram
    autonumber
    participant Camera as RealSense Camera
    participant ROS2 as GraspPoseClientNode<br/>(Grasp_Pose_Generation)
    participant Server as VLA Grasp Server<br/>(FastAPI)
    participant VLM as VLM Provider
    participant SAM2 as SAM2
    participant CGN as Contact-GraspNet<br/>Worker (subprocess)
    participant WebUI as Web UI (Browser)
    participant Robot as Robot Controller

    Note over Camera,ROS2: 持续运行：相机数据订阅
    Camera->>ROS2: /camera/color/image_raw<br/>/camera/aligned_depth_to_color/image_raw<br/>/camera/color/camera_info
    ROS2->>ROS2: ApproximateTimeSynchronizer<br/>→ 缓存最新同步帧 (color, depth, K)

    Note over WebUI,Server: Phase 1 — 触发拍照上传
    WebUI->>Server: POST /request_capture
    Server->>Server: capture_requested = True

    loop 2 Hz 轮询
        ROS2->>Server: GET /poll_capture_request
        Server-->>ROS2: {requested: true}
    end

    ROS2->>ROS2: take_snapshot()<br/>检查帧时效 (< max_snapshot_age_s)
    ROS2->>Server: POST /upload_capture<br/>(rgb.png, depth.npy, K_json, frame_id)
    Server->>Server: materialize_capture()<br/>→ 写入 pending_capture/camera_data.npy<br/>→ 生成 color_preview.jpg
    Server-->>ROS2: {status: ok, shape, frame_id}

    Note over WebUI,Server: Phase 2a — Option B: VLM + SAM2 + CGN (主流程)
    WebUI->>Server: POST /run_grasp<br/>(task_spec, top_k, num_candidates)
    Server->>Server: run_vg_pipeline()<br/>→ 加载 camera_data.npy
    Server->>VLM: build_align_prompt_multi() +<br/>run_vg_inference()<br/>→ 输出 num_candidates 个 bounding box
    VLM-->>Server: raw bounding box JSON
    Server->>Server: parse_align_results_multi()
    Server->>SAM2: segment within each bounding box
    SAM2-->>Server: 分割 mask + 点云<br/>→ 生成 contact_graspnet_input_NNN.npz
    loop 每个 candidate npz
        Server->>CGN: predict(input_NNN.npz) → predictions_input_NNN.npz
        CGN-->>Server: 6-DoF grasps (pose_4x4, scores, contact_pts)
    end
    Server->>Server: select_top_k_grasps()<br/>→ normalize_predictions_multi()<br/>→ 跨 candidate/segment 合并排序<br/>→ 取 top_k GraspDict[]
    Server->>Server: save_grasp_viz() + save_grasp_viz_3d()
    Server-->>WebUI: {run_id, grasps[], grasp_viz, elapsed_ms}

    Note over WebUI,Server: Phase 2b — Option B: VLA Alignment (轻量)
    WebUI->>Server: POST /run_align<br/>(task_spec, num_candidates)
    Server->>VLM: build_align_prompt() + run_vg_inference()<br/>→ 输出 2D 对齐点 + 夹角
    VLM-->>Server: point_yx, angle_deg, width_m
    Server->>Server: build_align_grasp()<br/>→ depth 反投影→6-DoF (approach 固定 camera +Z)
    Server-->>WebUI: {run_id, grasps[]}

    Note over WebUI,Server: Phase 3 — 服务端预过滤 (pre-IK filter)
    WebUI->>Server: POST /trigger_ik_check
    Server->>Server: filter_grasps(grasps, depth, K)<br/>① 硬过滤: width ∈ [2cm, 8cm]<br/>② 硬过滤: 碰撞穿透 < max_penetration_m<br/>③ 软评分: clearance_score [0,1]<br/>④ 软评分: collision_score [0,1]<br/>⑤ 软评分: contact_quality_score [0,1]
    Server->>Server: ik_results[trace_id] = {status:pending,<br/>grasps: filtered[]}<br/>pending_publish = {mode: ik_check, ...}
    Server-->>WebUI: {trace_id, num_before_filter, num_after_filter}

    Note over ROS2,Server: Phase 4 — ROS2 IK 可行性检验
    loop 2 Hz 轮询
        ROS2->>Server: GET /poll_publish
        Server-->>ROS2: {mode: ik_check, grasps: filtered[], trace_id}
    end

    ROS2->>ROS2: _check_ik_feasibility(grasps)<br/>① TF lookup: camera_optical → LIO_base_link (T₄ₓ₄)<br/>② _transform_to_base(grasp, T) → base_frame pose<br/>③ Pinocchio SE3 target<br/>④ 两阶段 Gauss-Newton IK:<br/>   Stage1: position-only (3-DOF)<br/>   Stage2: full 6-DOF<br/>⑤ 通过 → passed[], 失败 → 丢弃
    ROS2->>Server: POST /submit_ik_result<br/>{run_id, trace_id, grasps: ik_passed[]}
    Server->>Server: ik_results[trace_id].status = complete<br/>grasps 更新为 IK 通过的（camera frame, 软分数完整保留）
    Server-->>ROS2: {status: ok, num_passing}

    Note over WebUI,Server: Phase 5 — 选择最优抓取姿态
    loop 轮询直到 ready
        WebUI->>Server: GET /ik_result_status?trace_id=...
        Server-->>WebUI: {ready: true/false, count: N}
    end

    WebUI->>Server: POST /select_and_execute {trace_id}
    Server->>Server: rank_grasps(ik_passed)<br/>composite_score = (clearance×0.30 +<br/>collision×0.25 + width_margin×0.25 +<br/>contact_quality×0.20) / active_weights
    Server->>Server: pending_publish = {mode: execute, grasps: [best]}<br/>save_grasp_viz_best() → gold overlay JPEG
    Server-->>WebUI: {best_index, scored_grasps, ranked_indices}

    Note over ROS2,Robot: Phase 6 — 执行
    loop 2 Hz 轮询
        ROS2->>Server: GET /poll_publish
        Server-->>ROS2: {mode: execute, grasps: [best_grasp]}
    end

    ROS2->>ROS2: _transform_to_base(best_grasp, T_camera→base)<br/>+ grasp_offset_base (系统误差补偿)
    ROS2->>Robot: ~/best_grasp (PoseStamped, LIO_robot_base_link)
    ROS2->>Robot: ~/grasps (PoseArray, LIO_robot_base_link)
    ROS2->>ROS2: broadcast TF: grasp_best / grasp_best_cam (RViz 可视化)
```

---

## 流程图 (Flowchart)

```mermaid
flowchart TD
    subgraph CAM["相机 & ROS2 节点 (持续运行)"]
        A([RealSense Camera]) -->|color + depth + camera_info| B[ApproximateTimeSynchronizer]
        B --> C[(缓存最新同步帧)]
    end

    subgraph WEBUI["Web UI 触发流 (主流)"]
        direction TB
        W1([用户打开 Web UI]) --> W2[POST /request_capture]
        W2 --> W3{ROS2 poll_capture_request?}
        W3 -->|Yes| W4[ROS2: take_snapshot\n→ POST /upload_capture]
        W4 --> W5[(Server: pending_capture\ncamera_data.npy)]
        W5 --> W6{选择生成方式}

        W6 -->|Option A VLM+SAM2+CGN| OB[POST /run_grasp]
        W6 -->|Option B VLA对齐| OC[POST /run_align]

        subgraph OPT_B["Option A — VLM + SAM2 + CGN"]
            OB --> B1[build_align_prompt_multi\n→ VLM 推理]
            B1 --> B2[parse_align_results_multi\n→ num_candidates 个 bounding box]
            B2 --> B3[SAM2 分割 → mask + 点云]
            B3 --> B4[导出 contact_graspnet_input_NNN.npz]
            B4 --> B5[CGN Worker subprocess\n→ predictions_input_NNN.npz\n6-DoF grasps + scores]
            B5 --> B6[normalize_predictions_multi\n→ 跨 candidate 合并排序\n→ top_k GraspDict]
        end

        subgraph OPT_C["Option B — VLA 对齐 (轻量)"]
            OC --> C1[build_align_prompt\n→ VLM 推理]
            C1 --> C2[parse: point_yx, angle_deg, width_m]
            C2 --> C3[depth 反投影 5×5 中值\n→ 3D position]
            C3 --> C4[build_align_grasp\napproach 固定 camera +Z\nscore 固定 1.0]
        end

        B6 & C4 --> GEN_OUT([grasps: GraspDict[]\npose_4x4, quaternion_xyzw\nwidth_m, approach_dir_xyz\nscore, source])
        GEN_OUT --> VIZ[save_grasp_viz\nsave_grasp_viz_3d]
    end

    subgraph FILTER["服务端预过滤 /trigger_ik_check"]
        F0[加载 depth + K\nfrom pending_capture] --> F1
        GEN_OUT --> F1

        F1{check_width\nwidth_m ∈ 2–8 cm?}
        F1 -->|No| REJ_W[rejected_width]
        F1 -->|Yes| F2{_score_collision\n穿透 > max_penetration?}
        F2 -->|Yes| REJ_C[rejected_collision]
        F2 -->|No| F3[计算软评分\nclearance_score\ncollision_score\ncontact_quality_score]
        F3 --> F4[(ik_results\ntrace_id → pending\nfiltered_grasps[])]
        F4 --> F5[pending_publish\nmode: ik_check]
    end

    subgraph IK["ROS2 IK 可行性检验 (2 Hz 轮询)"]
        I1[GET /poll_publish\n→ mode: ik_check] --> I2[TF lookup\ncamera_optical → LIO_base_link\nT₄ₓ₄]
        I2 --> I3[_transform_to_base\ngrasp camera frame → base frame]
        I3 --> I4[Pinocchio SE3 target\nq_xyzw → Rotation + translation]
        I4 --> I5{两阶段 Gauss-Newton IK\nStage1: position-only\nStage2: full 6-DOF}
        I5 -->|收敛 q_solution| I6[passed_grasps\n保留 camera-frame dict\n含软评分字段]
        I5 -->|未收敛| I7[丢弃该候选]
        I6 --> I8[POST /submit_ik_result\ntrace_id, grasps: passed]
        I8 --> I9[(ik_results\ntrace_id → complete)]
    end

    subgraph SCORE["评分 & 选择 /select_and_execute"]
        S1[rank_grasps\nIK 通过候选] --> S2[compute_composite_score\nclearance × 0.30\ncollision × 0.25\nwidth_margin × 0.25\ncontact_quality × 0.20]
        S2 --> S3[排序 → best_grasp\nsave_grasp_viz_best\ngold overlay]
        S3 --> S4[pending_publish\nmode: execute\ngrasps: best_grasp]
    end

    subgraph EXEC["执行 (ROS2 2 Hz 轮询)"]
        E1[GET /poll_publish\n→ mode: execute] --> E2[_transform_to_base\nbest_grasp → base frame\n+ grasp_offset_base 误差补偿]
        E2 --> E3[~/best_grasp\nPoseStamped\nLIO_robot_base_link]
        E2 --> E4[~/grasps\nPoseArray]
        E2 --> E5[broadcast TF\ngrasp_best frame\nRViz 可视化]
    end

    subgraph DIRECT["直接 API 流 (旧版 /grasp)"]
        D1[POST /grasp\nrgb + depth + K\ntask_spec + frame_id] --> D2[materialize_request\n写入 capture_dir]
        D2 --> D3[run_vg_pipeline\nVLM+SAM2+CGN]
        D3 --> D4[select_top_k_grasps]
        D4 --> D5[返回 JSONResponse\ngrasps[]]
    end

    C --> W4
    F5 --> I1
    I9 --> S1
    S4 --> E1

    style OPT_B fill:#e3f2fd,stroke:#2196f3
    style OPT_C fill:#fff3e0,stroke:#ff9800
    style FILTER fill:#fce4ec,stroke:#e91e63
    style IK fill:#f3e5f5,stroke:#9c27b0
    style SCORE fill:#e8eaf6,stroke:#3f51b5
    style EXEC fill:#e0f2f1,stroke:#009688
    style DIRECT fill:#f5f5f5,stroke:#9e9e9e,stroke-dasharray: 5 5
```

---

## 关键数据结构流转

```mermaid
flowchart LR
    subgraph GEN["生成阶段"]
        G1["GraspDict (原始)\n─────────────\nscore: float\nmodel_confidence: float|null\npose_4x4: 4×4\nposition_xyz: [x,y,z]\nquaternion_xyzw: [x,y,z,w]\nwidth_m: float|null\napproach_dir_xyz: [x,y,z]\nsource: {candidate_index,\n  segment_id, grasp_index,\n  predictions_npz}"]
    end

    subgraph FILT["filter_grasps 后"]
        F1["GraspDict (annotated)\n─────────────\n(所有原始字段保留)\n+ clearance_score: [0,1]\n+ collision_score: [0,1]\n+ contact_quality_score: [0,1]\n\n注: width_m=null 已在此处被\ncheck_width 过滤掉"]
    end

    subgraph IKP["submit_ik_result 后 (IK通过)"]
        I1["GraspDict (IK-passed)\n─────────────\n(所有 annotated 字段不变)\n仍为 camera frame 坐标\nbase-frame 变换仅在内存中\n不回写 dict"]
    end

    subgraph SCR["select_and_execute 后"]
        S1["GraspDict (scored)\n─────────────\n(所有 IK-passed 字段)\n+ composite_score: float\n  = clearance×0.30\n  + collision×0.25\n  + width_margin×0.25\n  + contact_quality×0.20"]
    end

    subgraph PUB["ROS2 发布 (execute)"]
        P1["base frame 变换\n─────────────\nT = TF(camera→LIO_base_link)\npose_base = T @ pose_4x4_cam\n+ grasp_offset_base\n\n→ PoseStamped (LIO_robot_base_link)\n→ PoseArray"]
    end

    G1 -->|硬过滤 + 软评分| F1
    F1 -->|Pinocchio IK check| I1
    I1 -->|rank_grasps| S1
    S1 -->|best_grasp → poll_publish| P1
```
