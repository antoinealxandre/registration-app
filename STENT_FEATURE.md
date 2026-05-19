# TAVI Pacemaker Dependency Risk Assessment Feature

**Author**: Antoine Alexandre  
**Start Date**: 2026-05-19  
**Status**: Planning Phase  
**Reference Article**: Nai Fovino et al, "Anatomical Predictors of Pacemaker Dependency After TAVR", Circulation: Arrhythmia and Electrophysiology, 2021;14:e009028

---

## 📄 Article Summary (Nai Fovino et al, 2021)

### Study Overview
- **Population**: 112 TAVR patients with pacemaker implantation (30 days post-TAVR)
- **Primary Endpoint**: 30-day pacemaker dependency
- **Key Finding**: ~45% of pacemaker-implanted patients remain pacemaker-dependent at 30 days and 1 year

### Critical Anatomic Predictors (Multivariate Analysis)

**Two independent predictors of 30-day pacemaker dependency:**

1. **ΔMSID ≥ 3 mm** (Odds Ratio 7.58, p=0.002)
   - Definition: **Difference between Implantation Depth and Membranous Septum length**
   - ΔMSID = ID - MS_length
   - Sensitivity 84%, Specificity 69% at 3 mm cutoff

2. **LVOT Calcifications under Left Coronary Cusp (LCC)** (OR 5.69, p=0.013)
   - Presence of calcium deposit in LVOT below LCC
   - Indicates asymmetrical prosthesis expansion risk

**Note**: MS length and ID individually NOT associated with pacemaker dependency (p>0.05)

---

## 🎯 Risk Assessment Model (MVP Version)

### MVP: MS Length Display + Future ΔMSID

**Phase 1–3 (MVP)**: Display **MS length in mm** + visual marker on 3D  
- User sees: "MS Length: 2.8 mm" (automatically calculated, updates in real-time)
- No risk score calculation yet (ID not measured)

**Phase 4+ (Optional)**: Add ΔMSID-based risk stratification (once ID available)
- If user adds ID manually or via stent mesh → calculate ΔMSID
- Apply simplified risk model:
  
| Risk Level | Criteria | Pacemaker Dependency Rate |
|-----------|----------|---------------------------|
| **LOW** | ΔMSID < 3 mm | **2.7%** |
| **HIGH** | ΔMSID ≥ 3 mm | **~70%** (avg of intermediate+high) |

**Note**: LCC calcium screening removed for MVP (can be added later as optional feature)

---

## 🔬 Anatomic Definitions (Key Measurements)

### Membranous Septum (MS) Length
- **Measurement location**: Preprocedural CT, dedicated **coronal view**
- **Definition**: Perpendicular distance from **annular plane** to beginning of **muscular septum**
- **Clinical significance**: Anatomic surrogate for distance between aortic annulus and His bundle exit
- **Typical value**: 2.7–3.6 mm (pacemaker-dependent vs. non-dependent in study)

### Implantation Depth (ID)
- **Measurement location**: Final aortic **angiogram** (intraoperative fluoroscopy)
- **Definition**: Distance between lower end of THV frame and lowest part of **noncoronary cusp**
- **Typical value**: Mean 6.3–8.0 mm (non-dependent vs. dependent)

### ΔMSID (Delta MSid)
- **Formula**: `ΔMSID = ID - MS_length`
- **Critical threshold**: ≥ 3 mm indicates HIGH risk
- **Physical interpretation**: Measure of THV frame overlap/compression on atrioventricular bundle
  - Small positive ΔMSID: THV implanted shallowly, minimal bundle contact → transient conduction blocks
  - Large ΔMSID (≥3 mm): Deep THV implantation relative to short MS → permanent AV dissociation

### LVOT Calcifications
- **Relevant location**: Calcium deposit **under Left Coronary Cusp (LCC)**
- **Mechanism**: Asymmetrical prosthesis expansion toward RCC-NCC commissure, pushing frame further into LVOT
- **Assessment**: Semiquantitative (presence/absence in study; volume quantification recommended)

---

## 💡 Technical Approach (MVP: MS Length Only)

### Overall Strategy: 2D→3D Septum Localization

**MVP Goal**: User clicks MS on 2D CT → automatically compute & display MS length in 3D  
**Future Goal** (Phase 4): Add ID + risk scoring once that workflow is clear

### User Workflow

1. **Load CT** in main window (existing)
2. **Open "Membranous Septum" panel** (new)
3. **Click on axial CT slice** at MS location
   - 2D click coordinates captured
   - Converted to 3D world coords via CT affine
4. **See result**:
   - Blue sphere marker in 3D viewport at MS location
   - Text display: "MS Length: 2.8 mm"
5. **Adjust if needed** (click again to reposition)

### New Code Modules

**1. `registration_app/core/ms_measurement.py`** (NEW)

```python
def extract_annular_plane_from_segmentation(aortic_root_mask: np.ndarray, ct_affine) -> tuple:
    """
    Extract annular plane from aortic root segmentation.
    Returns: (plane_normal, plane_point_3d)
    """
    # Fit plane to mask vertices (least squares)
    
def click_2d_to_3d(click_xy: tuple, slice_idx: int, ct_volume_shape, ct_affine) -> np.ndarray:
    """Convert 2D click on CT slice to 3D world coords."""
    
def measure_ms_length(ms_point_3d: np.ndarray, plane_normal, plane_point) -> float:
    """Perpendicular distance from MS point to annular plane."""
```

**2. `registration_app/ui/ms_panel.py`** (NEW)

```python
class MembranousSeptumPanel(QWidget):
    """Compact panel for MS measurement."""
    
    def __init__(self):
        # Button: "Click on CT to mark MS"
        # Display: "MS Point: (x, y, z) mm"
        # Display: "MS Length: X.X mm"
        # 3D visualization callback
```

**3. Modifications to `main_window.py`**:
- Add MS panel to sidebar or new tab
- Wire CT click detection from AnnotationCanvas
- Update 3D marker + MS length display on callback

---

## 📋 Implementation Phases (MVP)

### Phase 1: Annular Plane Extraction + MS 2D→3D Annotation (1 day)
- [ ] Extract annular plane from aortic root segmentation mesh
- [ ] Implement CT 2D viewer click detection (on existing AnnotationCanvas)
- [ ] Convert 2D click → 3D world coords via CT affine transform
- [ ] Place 3D sphere marker at MS location in 3D viewport
- **Deliverable**: User clicks MS on 2D axial slice → blue sphere appears in 3D at MS location

### Phase 2: MS Length Calculation + Display (1 day)
- [ ] Implement perpendicular distance from MS point to annular plane
- [ ] Wire calculation to update callback
- [ ] Display MS length in mm in UI panel
- [ ] Validate against 2–3 ground-truth cases
- **Deliverable**: "MS Length: 2.8 mm" displayed in real-time, updates when MS point moves

### Phase 3: UI Panel + Main Integration (0.5 day)
- [ ] Create compact `MembranousSeptumPanel` widget
- [ ] Wire to main_window.py (new tab or sidebar section)
- [ ] Test workflow: load CT → click MS → read MS length
- **Deliverable**: Full MVP working in main UI

### Phase 4 (Optional, Future): ID + ΔMSID + Risk Scoring
- Manual ID spinbox or stent mesh centroid depth
- ΔMSID calculation
- Risk level badge (LOW/HIGH)
- *Not in initial release*

**Total Effort (MVP)**: ~2–3 days, 1 feature branch, 1 PR

---

## 🔑 Critical Decisions (RESOLVED)

### Q1: Annular Plane Definition ✅
- **Decision**: Auto-extract from aortic root segmentation (already have: aortic root, 3 leaflets, LV)
- **Implementation**: Fit plane to aortic root mask vertices → extract annular plane normal + point
- **Fallback**: Manual 2-point annotation if auto-extraction fails
- **Status**: Ready to implement in Phase 1

### Q2: CT Coordinate System ✅
- **Decision**: All CTs axial, isotropic → same orientation always
- **Impact**: Affine transform straightforward, no rotation/skew handling needed
- **Status**: Simplifies 2D→3D conversion

### Q3: Implantation Depth (ID) ✅
- **Decision**: NOT implementing ID for MVP (workflow minimal for user)
- **MVP Scope**: Measure **MS length only** → show MS value in UI
- **Future**: Phase 4 (optional) can add manual ID spinner if needed
- **Rationale**: User can measure ID separately or use stent centroid depth approximation later
- **Status**: Phase 1–3 focus on MS measurement + display

### Q4: LVOT Calcium Annotation ✅
- **Decision**: SKIP entirely for MVP
- **Simplified Risk Model**: Based on **MS length alone** (not ΔMSID in first iteration)
  - Display MS length value
  - Optional: Add "annotate calcium" checkbox later (Phase 5)
- **Status**: Removes checkbox complexity, cleaner initial release

### Q5: Validation Dataset ✅
- **Decision**: Build dataset from existing data (you have all measurements)
- **Goal**: 3–5 TAVR cases with ground-truth MS length + actual PM dependency outcomes
- **Timing**: Gather during Phase 1–2, use in Phase 5 validation
- **Status**: Plan to create after code phases

---

## 🛠️ Technical Requirements

### Dependencies (already in requirements.txt)
- `numpy`, `scipy`, `nibabel` (3D math, CT loading)
- `trimesh` (stent mesh operations)
- `PyQt5` (UI components)
- `scikit-image` (optional: morphological ops for annulus extraction)

### New Utilities Needed
- 2D CT viewer with click detection (use existing AnnotationCanvas)
- Affine transform utilities (nibabel has these)
- Perpendicular distance functions (scipy.spatial or numpy)
- Risk model serialization (pickle or JSON)

---

## 📊 Output & Reporting

### Per-Patient Report Structure
```json
{
  "patient_id": "TAV_001",
  "examination_date": "2026-05-19",
  "ct_info": {
    "modality": "Cardiac CT",
    "spacing_mm": [0.8, 0.8, 1.0]
  },
  "measurements": {
    "ms_point_3d_mm": [12.5, 8.3, 45.2],
    "ms_length_mm": 3.1,
    "implantation_depth_mm": 7.2,
    "delta_msid_mm": 4.1,
    "lvot_lcc_calcium": true
  },
  "risk_assessment": {
    "risk_level": "high",
    "pacemaker_dependency_rate": 0.812,
    "reasoning": "ΔMSID ≥3 mm and LVOT calcification under LCC present"
  },
  "recommendation": "CRT device recommended if EF<40%; expedited PM implantation; no RV pacing minimization"
}
```

---

## 🚀 Git Workflow (MVP)

**Branch**: `feature/ms-measurement-v1`

**Commits** (1 per phase):
```
1. core: ms_measurement module (annular plane extraction + distance calc)
2. ui: ms_panel widget + main_window integration
3. testing: validation on 3 TAVR cases, end-to-end workflow
```

**PR Checklist** (MVP):
- [ ] MS point marked on 2D axial, rendered as 3D sphere ✓
- [ ] MS length matches manual measurement on ≥3 ground-truth CTs
- [ ] Annular plane auto-extracted from aortic root segmentation
- [ ] Fallback: manual plane annotation if needed
- [ ] UI responsive, no freezing on CT/marker updates
- [ ] Docstrings complete, French comments on medical logic
- [ ] Live demo: user clicks MS → length updates in real-time

---

## 🎓 Knowledge Base for Claude

### Key Clinical Context
- **Why ΔMSID matters**: Captures relative overlap of THV frame with atrioventricular bundle, not absolute measurements
- **Why LCC calcium matters**: Predicts asymmetrical expansion, pushing frame deeper
- **Why persistence matters**: ~45% of pacemaker-implanted patients remain dependent at 1 year; NOT transient
- **Clinical impact**: Device selection (CRT vs. leadless), programming strategy, hospital stay planning

### Anatomic Anchors
- **His bundle**: Located just below MS, emerges on LV surface
- **Conduction system path**: AVN → His bundle → LBB (left) + RBB (right)
- **Annulus**: Fibrous ring, ~440 mm² in study population
- **Muscular septum**: Thicker region distal to MS

### Study Limitations (Context)
- Single-center, retrospective (external validation needed)
- ID measured on fluoroscopy, not CT (adds measurement noise)
- 1.8% of cohort had post-TAVR CT (mostly none)
- Small sample (N=112), advanced age (median 81)
- Cutoff 3 mm may not generalize to other populations

---

## 📞 Next Steps (MVP)

### Immediate (Today)
- [x] Answer Q1–Q5 → **DONE** ✓
- [ ] Gather 3–5 TAVR cases with ground-truth MS length
  - You have CT + segmentation + annotations
  - Manually measure MS length on each (reference value)
  - Store in simple CSV: `case_id, ms_length_mm_manual, ct_file`

### Before Code (Opus Planning Session)
- [ ] Create test dataset (3 cases with known MS lengths)
- [ ] Verify aortic root segmentation quality in each case
- [ ] Outline AnnotationCanvas click-capture mechanism (existing code)

### Code Phase (Opus Session 1, 1 day)
1. Phase 1 (0.5 day): `ms_measurement.py` + annular plane extraction
2. Phase 2 (0.5 day): `ms_panel.py` + main_window integration
3. Test on 3 cases, validate accuracy

### Validation (After Code)
- Compare measured MS length vs. manual ground truth
- Iterate if >5% error
- Document assumptions (annular plane definition, click tolerance)

---

**Last Updated**: 2026-05-19  
**Status**: MVP Approved – Ready for Opus coding session
