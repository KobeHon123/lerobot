# WowSkin Sensor Data Refactoring Guide

## What Changed

### Before (Combined)
```
observation.state: [motor_0, motor_1, ..., motor_5, wowskin_0_x, wowskin_0_y, ..., wowskin_4_z]
                   └─────────── 6 motors ──────────┘ └────────── 15 WowSkin dims ────────────┘
Shape: (21,)
```

### After (Separated) ✅
```
observation.state:  [motor_0, motor_1, ..., motor_5]
                    └──────── 6 motors ────────┘
Shape: (6,)

observation.wowskin: [wowskin_0_x, wowskin_0_y, ..., wowskin_4_z]
                     └───────── 15 WowSkin dims ────────┘
Shape: (15,)
```

---

## Why This Is Better

| Aspect | Combined | Separated ✅ |
|--------|----------|------------|
| **Train with WowSkin** | ✅ Yes (always) | ✅ Yes (optional) |
| **Train without WowSkin** | ❌ Requires separate dataset | ✅ Same dataset, different config |
| **Dataset reusability** | ❌ Limited | ✅ High |
| **Transfer learning** | ❌ Difficult | ✅ Easy |
| **Sensor flexibility** | ❌ Coupled | ✅ Independent |
| **Code clarity** | ❌ Implicit | ✅ Explicit |

---

## Updated Code

### 1. SO100WowSkin Robot Class
**File:** `/home/kobe/lerobot/src/lerobot/robots/so100_wowskin.py`

```python
@cached_property
def observation_features(self) -> dict[str, type | tuple]:
    # Get parent motor state features
    parent_features = super().observation_features
    # Add WowSkin as a separate feature (vector, not named fields)
    wowskin_features = {f"observation.wowskin": (self.config.wowskin_num_mags * 3,)}
    return {**parent_features, **wowskin_features}

def get_observation(self) -> dict[str, Any]:
    obs = super().get_observation()
    # observation.state now contains ONLY motors (6 dims)
    # observation.wowskin contains WowSkin data (15 dims)
    wowskin_values = self._read_wowskin()
    obs["observation.wowskin"] = wowskin_values.astype(np.float32)
    return obs
```

---

## Training Commands

### Option 1: Train WITH WowSkin
```bash
lerobot-train \
  --dataset.repo_id=KHandsome/wow-50-1-refactored \
  --policy.type=act \
  --output_dir=outputs/train/act-with-wowskin \
  --batch_size=2 \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=50000 \
  --policy.repo_id=KHandsome/act-with-wowskin
```

**Policy input features:** `observation.state` (6 dims) + `observation.wowskin` (15 dims) = **21 dims**

### Option 2: Train WITHOUT WowSkin (Same Dataset!)
```bash
lerobot-train \
  --dataset.repo_id=KHandsome/wow-50-1-refactored \
  --policy.type=act \
  --policy.input_features='observation.state' \
  --output_dir=outputs/train/act-without-wowskin \
  --batch_size=2 \
  --policy.device=cuda \
  --wandb.enable=true \
  --steps=50000 \
  --policy.repo_id=KHandsome/act-without-wowskin
```

**Policy input features:** `observation.state` (6 dims) only = **6 dims**

---

## Dataset Processing

### Refactoring Script Status
- **Running:** `KHandsome/wow-50-1` → `KHandsome/wow-50-1-refactored`
- **Processing:** 51 episodes, 23,861 frames total
- **ETA:** 20-40 minutes
- **Script:** `/home/kobe/lerobot/refactor_dataset_separate_wowskin.py`

### What the Script Does
1. Loads your original dataset with combined state
2. Splits each frame's `observation.state` into:
   - Motor-only `observation.state` (6 dims)
   - `observation.wowskin` vector (15 dims)
3. Copies all episodes with videos/actions unchanged
4. Uploads as `KHandsome/wow-50-1-refactored` to Hub

---

## Visualization in Rerun

The Rerun logger automatically handles the new schema:

```
observation.state
├── shoulder_pan.pos
├── shoulder_lift.pos
├── elbow_flex.pos
├── wrist_flex.pos
├── wrist_roll.pos
└── gripper.pos

observation.wowskin
├── mag_0
│  ├── x
│  ├── y
│  └── z
├── mag_1
│  ├── x
│  ├── y
│  └── z
... (repeated for mag_2, mag_3, mag_4)
```

---

## Recording with New Schema

Once refactoring completes, new recordings will automatically use:

```bash
lerobot-record \
  --robot.type=so100_wowskin \
  --dataset.repo_id=KHandsome/wow-50-2 \
  ...
```

**Output dataset will have:**
- `observation.state` (6 dims) - motors only
- `observation.wowskin` (15 dims) - WowSkin only
- Ready for both with/without WowSkin training! ✅

---

## Next Steps

1. **Wait for refactoring script to complete** (20-40 min)
   - Check progress: `ps aux | grep refactor_dataset`
   - Monitor: `/home/kobe/.cache/huggingface/lerobot/KHandsome/wow-50-1-refactored/`

2. **Once done, verify the new dataset:**
   ```bash
   python -c "
   from lerobot.datasets.lerobot_dataset import LeRobotDataset
   ds = LeRobotDataset(repo_id='KHandsome/wow-50-1-refactored')
   print(ds.meta.features.keys())
   print('observation.state shape:', ds.meta.features['observation.state']['shape'])
   print('observation.wowskin shape:', ds.meta.features['observation.wowskin']['shape'])
   "
   ```

3. **Train both versions for comparison:**
   - With WowSkin: `lerobot-train ... --dataset.repo_id=KHandsome/wow-50-1-refactored`
   - Without WowSkin: `lerobot-train ... --policy.input_features='observation.state'`
   - Compare results in W&B dashboard 📊

---

## Benefits Summary

✅ **One dataset, multiple training scenarios**
✅ **Easier ablation studies** (with/without tactile sensor)
✅ **Better transfer learning** (motor-only models work on robots without WowSkin)
✅ **Cleaner code** (explicit about what data is used)
✅ **Standard practice** (matches how LeRobot handles multi-sensor data)
✅ **Future-proof** (can add more sensors without breaking schema)
