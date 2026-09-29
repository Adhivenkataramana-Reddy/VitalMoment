# VitalMoment — Final Workflow (Cuts / External Bleeding Prototype) — v6

This document locks the final architecture, module-by-module, exactly as it maps onto the
approved proposal (CV → Observation Generator → Observation Validator → State Builder →
Adaptive Workflow Engine), scoped to what is achievable in a 6-month timeline with ~2 months
of intensive build time.

**This revision adds research defensibility back on top of the lean build** (see Section 0):
every simplification made for buildability is paired with a lightweight mechanism that keeps
the design auditable, ablatable, and extensible — without reintroducing the full model-training
overhead of the earlier architecture draft.

**v6 changelog (algorithm/model review over v5 — not new bugs, but real gaps found while
checking whether each technique is actually the best fit, not just a working one):**
1. The State Builder's buffer was defined as a fixed frame count ("~30 observations, ~3–5
   seconds") which doesn't match its own stated 10–15 fps effective sampling rate (30 frames is
   2–3s at that rate, not 3–5s) — and every specific basis-window example elsewhere in the doc
   was already time-based, not frame-count-based. Redefined the buffer as a rolling time window
   so it stays correct even if effective fps drifts (Section 3.5).
2. Made the `pressure_applied` fusion rule explicit (`hand_near_wound OR cloth_on_wound`) —
   previously only implied by a field comment, with the one worked example showing hand alone
   and never showing how cloth combines. Left ambiguous, this is exactly the kind of spec gap
   that produces an implementation bug (Section 3.5).
3. `bleeding_severity`'s severe/moderate boundary used a single threshold (>6%), which is prone
   to flicker for readings that hover near the line — directly risking the demo's core
   "reducing" trend moment (Section 7, step 5), where blood-area shrinks gradually through
   exactly that boundary. Added a small hysteresis band (enter at >6%, exit only below 4.5%)
   (Section 3.3).
4. The gauze/cloth detector used a white/beige color heuristic, which is the one detector most
   likely to collide with skin tones and generic backgrounds — the most fragile of the four.
   Recommended a distinctly-colored prop (e.g. blue/green cloth) instead, which is free and
   removes the ambiguity outright (Section 3.2).
5. `gauze_present` had no basis string, unlike every other derived Facts field, despite the
   section explicitly claiming "`basis` on each derived field is the audit trail" — and it was
   never stated what raw observation it actually traces to. Clarified that it reuses the same
   `cloth_on_wound` signal as `pressure_applied`'s cloth input, and added the missing
   `gauze_present_basis` field (Section 3.5).

**v5 changelog (over v4):**
1. Split the reversal rule: `R-97` now only covers `bleeding == "severe"` (where it must be
   gated by `pressure_recently_removed` to avoid shadowing `R-95`, per the v3 fix). A new rule,
   `R-96`, covers `bleeding == "reducing"` unconditionally. This closes a gap where
   `bleeding == "reducing"` with `pressure_applied == false` and `pressure_recently_removed ==
   false` (pressure removed more than the recency window ago, or the dwell-time never reached)
   matched no rule at all — confirmed by exhaustively enumerating every Facts combination
   against the v4 table.
2. Moved the low-confidence fallback from `R-05` (priority 5, the bottom of the table) to
   `R-99` (priority 99, just below the wound-visibility check). At priority 5, `R-05` was
   shadowed by a domain-specific rule in 95% of Facts combinations — i.e. a low-confidence
   reading almost always still produced specific, confident-sounding guidance instead of the
   "unable to assess" fallback it was meant to trigger. Confidence is now checked before any
   bleeding-specific rule, so an unreliable read never drives specific medical guidance.

**v4 changelog (over v3):**
1. Added `R-10`, a catch-all rule for `bleeding in ["none","moderate"]` with `wound_visible ==
   true` — previously no rule matched this combination, so those two states in the Section 3.5
   enum produced no guidance at all. Not triggered by the demo script, but a real gap if the
   table is tested off-script or ablated for the results section.

**v3 changelog (correctness fixes over v2):**
1. Fixed a rule-table bug where the "reversal" rule (previously `R-20`) could never actually
   fire while `bleeding == "severe"`, because the higher-priority `R-95` fully shadowed it —
   contradicting the demo script, which relies on that exact rule firing in that exact state.
   Replaced with a new higher-priority rule, `R-97`, driven by a new derived Facts field,
   `pressure_recently_removed` (Section 3.5, Section 3.6).
2. Defined `Facts.confidence` explicitly as buffer coverage (fraction of expected frames that
   survived validation), not an average of per-frame `detection_confidence` — the latter can
   never fall below the Validator's 0.4 cutoff by construction, which would have made `R-05`
   unreachable (Section 3.5).

---

## 0. Why this is still a defensible design, despite being simplified

The earlier architecture draft (trained CV models, per-entity confidence/provenance, a
versioned state ontology, an evidence-scoring inference engine) is the **target architecture**.
This prototype is a **deliberate, documented scoping-down of it for a 6-month build**, not an
abandonment of it. That distinction is what you say explicitly in the report, and it's true if
the following four properties hold — which is what the rest of this revision adds:

1. **Interface stability** — the *shape* of the Observation Object and Facts object doesn't
   depend on which detection technique fills it in. HSV thresholding today, YOLOv11 tomorrow,
   both produce the same `blood_detected` / `bleeding_severity` fields downstream. This is
   what lets you claim the model-agnostic swap experiment as future work rather than a rewrite.
2. **Rule-level auditability** — every guidance decision traces to a named, numbered rule and
   the specific Facts values that triggered it (added in Section 3.6). No rule fires silently.
3. **Ablatability** — because rules are independent, priority-ordered conditions (not a trained
   classifier), you can disable any single rule or threshold and report what demo behavior it
   was responsible for. That's your experiments section even at prototype scale.
4. **Versioned decision surface** — the rule table and the threshold constants are versioned
   artifacts (Section 3.6, Section 4), so "what did the system believe and why" is answerable
   for any recorded demo run, and changes across the 6 months are traceable.

None of this requires trained models, per-entity confidence, or a state memory manager — it's
achievable with logging discipline and small schema additions on top of the plain-Python design
already planned.

---

## 1. Scope Lock

- **Injury type:** external bleeding from a cut/wound on a visible body part (arm/hand), only.
- **No** multi-injury handling, no burns/CPR/fractures, no environment-material branching, no
  cloud backend, no production database. These are explicitly "Future Work" — see Section 9 for
  how each maps back onto the original research-grade architecture.
- **Goal of the prototype:** demonstrate that the system can (a) detect the relevant visual
  evidence, (b) infer a medical state from it over time, and (c) give guidance that
  **dynamically corrects itself** when the real-world situation changes or reverses —
  this last point is the entire thesis and the main demo moment.

---

## 2. End-to-End Architecture

```
┌──────────┐   ┌──────────────┐   ┌───────────────────┐   ┌───────────────┐   ┌────────────────────────┐
│  Camera  │──▶│  CV Models   │──▶│ Observation        │──▶│ Observation   │──▶│ State Builder           │
│ (webcam) │   │ (detection)  │   │ Generator          │   │ Validator     │   │ (temporal + evidence    │
└──────────┘   └──────────────┘   └───────────────────┘   └───────────────┘   │  fusion + state estimate)│
                                                                                └────────────┬────────────┘
                                                                                             ▼
                                                                                 ┌────────────────────────┐
                                                                                 │ Adaptive Workflow Engine│
                                                                                 │ (rule-based inference)  │
                                                                                 └────────────┬────────────┘
                                                                                             ▼
                                                                                 ┌────────────────────────┐
                                                                                 │   Guidance Output       │
                                                                                 │ (on-screen + optional   │
                                                                                 │  voice)                 │
                                                                                 └────────────────────────┘
```

Every box above corresponds 1:1 to a box in your approved proposal slide. Nothing is removed;
"Adaptive Workflow Engine" is *implemented* as a rule engine rather than a fixed graph — that
is an internal design decision, not a scope change.

---

## 3. Module-by-Module Detail

### 3.1 Camera Input

- Standard webcam feed via OpenCV `cv2.VideoCapture(0)`.
- Frames sampled at a fixed rate (e.g. every 2nd–3rd frame, ~10–15 fps effective) — no need to
  process every frame; first-aid situations evolve over seconds, not milliseconds.

### 3.2 CV Models (Detection Layer)

This layer's **only job is sensing** — it never makes decisions, never remembers past frames,
and outputs raw detections per frame. It is also the layer most likely to be swapped later
(HSV → trained segmentation model), which is why its output contract is fixed independently of
technique (see `detector_id` field below).

| Detection target | Technique | Why this technique |
|---|---|---|
| Blood / bleeding region | HSV color-space thresholding (red/dark-red hue+saturation range), contour area | Real-time, no training data needed, literature-established for bleeding detection (used in capsule-endoscopy and surgical bleeding-detection research) |
| Wound presence | Skin-tone segmentation (YCrCb) combined with the blood-region contour | Cheap, doesn't need a trained wound classifier, which research shows is still an open, hard problem even for large vision-language models |
| Hand position / hand-near-wound | MediaPipe Hands (pretrained) | Free, robust, gives per-landmark coordinates so "hand touching wound area" is a simple distance check |
| Gauze / cloth on wound | Color/contour heuristic, using a **distinctly-colored prop** (e.g. a bright blue or green cloth) rather than realistic white/beige gauze | No training needed; a saturated, non-skin-tone color is far more reliable to threshold than white/beige, which can overlap with skin tones and many backgrounds — this is the most fragile of the four detectors, so it's worth trading a bit of visual realism for reliability in a live demo |
| (Optional, if time allows in months 3-4) Object detection upgrade | YOLOv11 (as listed in your requirements slide) fine-tuned on a small custom set (gauze, hand, cut) | Only attempt this once the HSV/MediaPipe pipeline is working end-to-end — treat it as an upgrade, not the initial path, since it needs data collection + training time you may not have |

Output per frame: a small dict of raw detections, tagged with which detector produced it —
this one field is what makes the later model-swap claim honest rather than aspirational:
```json
{
  "detector_id": "hsv_v1",
  "blood_pixels_pct": 4.2,
  "wound_contour_present": true,
  "hand_distance_to_wound_cm_est": 3.1,
  "cloth_overlap_pct": 0.0
}
```

### 3.3 Observation Generator

Converts raw CV numbers into **standardized, named observations** (boolean/categorical facts),
decoupling the decision logic from CV internals. This is the layer your proposal names
explicitly — its job is only translation, not judgment. Because this layer's output schema is
fixed regardless of `detector_id`, swapping the detection technique later (HSV → YOLOv11) never
requires touching anything downstream — that's the generalization claim made concrete.

Example mapping rules (simple thresholds, tuned during testing, versioned as `threshold_set_v1`
so later recalibration is a tracked change, not a silent edit). The severe/moderate boundary
uses two thresholds rather than one — a small hysteresis band — so a reading hovering right at
the line doesn't flicker `bleeding_severity` back and forth between frames; this matters
because the demo's key "reducing" trend (Section 7, step 5) is exactly the kind of gradual
change that would otherwise sit near that boundary for several seconds:
```
blood_pixels_pct > 1.5%              → "blood_detected": true
blood_pixels_pct > 6.0%              → "bleeding_severity": "severe"    (enter-severe threshold)
blood_pixels_pct < 4.5% (once severe) → "bleeding_severity": "moderate"  (exit-severe threshold — stays "severe" until it drops below this, not just below 6%)
blood_pixels_pct 1.5–6.0% (otherwise) → "bleeding_severity": "moderate"
hand_distance_to_wound < 5cm         → "hand_near_wound": true
cloth_overlap_pct > 40%              → "cloth_on_wound": true
```

Output: a structured **Observation Object**, one per processed frame:
```json
{
  "timestamp": 172.4,
  "wound_visible": true,
  "blood_detected": true,
  "bleeding_severity": "severe",
  "hand_near_wound": true,
  "cloth_on_wound": false,
  "detection_confidence": 0.83,
  "detector_id": "hsv_v1",
  "threshold_set": "v1"
}
```

### 3.4 Observation Validator

Filters out unreliable single-frame observations before they can affect the medical state.
Checks (all lightweight, rule-based — no ML needed here):

- **Confidence check** — discard observations below a confidence threshold (e.g. 0.4).
- **Consistency check** — if `blood_detected: false` but `bleeding_severity: severe` was just
  reported in the same frame, that's an internal contradiction → discard the frame.
- **Sanity check** — e.g. `cloth_on_wound: true` while `wound_visible: false` makes no sense
  spatially → discard.

Frames that fail validation are simply **not passed forward** — the State Builder keeps using
its last-known-good evidence rather than reacting to a garbage frame. This is what prevents a
single bad frame from causing a wrong guidance flicker. Each discard is logged with which check
rejected it (`reason: "confidence" | "consistency" | "sanity"`) — cheap to add, and it's what
lets you report a false-positive/discard rate in results rather than just asserting the
validator "works."

### 3.5 State Builder

This is where **temporal analysis + evidence fusion + medical state estimation** happens —
exactly as named in your proposal.

**Temporal analysis:** maintain a rolling buffer of validated observations covering the last
~5 seconds — sized by elapsed time, not a fixed frame count, since the effective sampling rate
(Section 3.1, ~10–15 fps) can drift with system load or the "every 2nd–3rd frame" sampling
choice. A fact is only considered "true" for decision-making once it has held for a minimum
duration (e.g. 2 seconds) or a majority of the buffer — this removes flicker from momentary CV
noise, and is a lightweight stand-in for the hysteresis concept from the full architecture
(min-dwell-time before a fact is trusted), scoped down to a single buffer instead of a full
state-memory manager. Individual derived facts can use shorter sub-windows within this buffer
where useful (e.g. the 3–4 second windows used by `pressure_basis` and
`pressure_recently_removed_basis` below) without needing a separate buffer per fact.

**Evidence fusion:** combine multiple observation fields into a smoothed, higher-level
**Facts** object — this is the single source of truth the decision engine reads. `basis` on
each derived field is the audit trail: it says which raw observations over which window
produced this fact, so any Facts snapshot is explainable after the fact, not just in the
moment.

```json
{
  "bleeding": "severe",        // "none" | "moderate" | "severe" | "reducing" | "controlled"
  "bleeding_basis": "avg blood_pixels_pct over last 5.0s buffer, trend=flat",
  "wound_visible": true,
  "pressure_applied": false,   // hand or cloth held on wound continuously
  "pressure_basis": "hand_near_wound held true for 0.0s of last 3.0s window",
  "pressure_recently_removed": true,
  "pressure_recently_removed_basis": "pressure_applied was true for >=2.0s within the last 4.0s window, now false",
  "gauze_present": false,
  "gauze_present_basis": "cloth_on_wound held true for 0.0s of last 3.0s window",
  "time_since_last_change_s": 6.0,
  "confidence": 0.81,
  "facts_schema_version": "1.1"
}
```

**Medical state estimation:** derive `bleeding` trend (`severe` → `reducing` → `controlled`)
by comparing blood-area percentage across the buffer window (e.g. shrinking average over the
last 5 seconds → "reducing").

**`pressure_applied` fusion (clarified):** this is `hand_near_wound OR cloth_on_wound`,
sustained continuously for the buffer's minimum-dwell duration — either input is sufficient,
matching the field comment ("hand or cloth held on wound"). The example `pressure_basis` above
shows the hand-only case; when cloth is what's actually holding pressure, the basis string
instead cites `cloth_on_wound`, e.g. `"cloth_on_wound held true for 4.2s of last 5.0s window"`.
Making this OR explicit here (rather than leaving it implied only by the field comment) is
worth doing because the correctness of `pressure_recently_removed` and every downstream rule
in Section 3.6 depends on `pressure_applied` being computed the same way regardless of which
input triggered it. `gauze_present` reuses this same `cloth_on_wound` observation — there's no
separate "gauze detector" in the Observation Object (Section 3.2); it's the identical raw
signal, just read at a different point in the sequence (once `bleeding == "controlled"`, a
sustained `cloth_on_wound` reads as a dressing rather than pressure). This is why
`gauze_present_basis` above cites `cloth_on_wound`, matching `pressure_basis`'s pattern.

**`pressure_recently_removed` (new in v3):** true when `pressure_applied` was continuously
true for at least a minimum dwell time within a short trailing window, and is now false. This
is what lets the rule engine distinguish "pressure was never applied yet" from "pressure was
just taken away" — the two situations that were conflated in v2 and need different guidance
text. It's computed with the same buffer-and-basis mechanism already used for every other
derived field, so it adds no new machinery, only one more derived boolean.

**`confidence` (clarified in v3):** this is **buffer coverage**, not an average of per-frame
`detection_confidence`. Per-frame scores that reach the buffer have already cleared the
Observation Validator's 0.4 threshold by definition, so averaging them could never fall below
0.4 — that would make the low-confidence fallback rule unreachable. Instead, `confidence` is
the fraction of the last N expected frames (given the sampling rate) that actually survived
validation and landed in the buffer. A stretch of discards — from bad lighting, motion blur,
or repeated consistency/sanity failures — pulls this value down even when every frame that did
arrive individually passed its own check. That drop is what the fallback rule below is meant
to catch.

This module produces **Facts**, not a "current state ID." There is no pointer to a graph node —
just the current best estimate of the world, always overwritable by new evidence, with a
`basis` string attached to each field so "why does the system believe this" is always
answerable from a logged Facts object alone.

### 3.6 Adaptive Workflow Engine (Rule-Based Decision Core)

This is the heart of the "dynamic decision-making" requirement. It is a **forward-chaining
rule engine**: a list of `(priority, condition, action)` rules, all re-evaluated from scratch
every cycle against the current Facts. The highest-priority matching rule wins. No history,
no "current node" — so any change in Facts can immediately change the decision, including
reversals.

Every rule now carries a stable ID and a version tag, and every firing is logged with the exact
Facts values that satisfied it — this is the direct, low-cost analog of the "evidence trail"
in the full architecture's State Inference Engine, and it's what makes the rule table
**ablatable**: disable rule `R-97` and rerun the recorded demo footage, and you can report
exactly which guidance moments depended on it.

**Full rule table for the cuts/bleeding scenario (`rule_table_v4`):**

| ID | Priority | Condition (on Facts) | Action / Guidance |
|---|---|---|---|
| R-100 | 100 | `wound_visible == false` | "No wound detected — position camera on the injury." |
| R-99 | 99 | `confidence < 0.4` | "Unable to clearly assess the situation — please check manually or call for help." |
| R-97 | 97 | `pressure_applied == false and pressure_recently_removed == true and bleeding == "severe"` | "Pressure was released — reapply pressure immediately." (this rule is what fires the *reversal* demo moment) |
| R-96 | 96 | `pressure_applied == false and bleeding == "reducing"` | "Pressure was released — reapply pressure immediately." |
| R-95 | 95 | `bleeding == "severe" and pressure_applied == false and pressure_recently_removed == false` | "Apply firm, direct pressure to the wound now." |
| R-90 | 90 | `bleeding == "severe" and pressure_applied == true and time_since_last_change_s > 15` | "Pressure isn't reducing bleeding — seek emergency help immediately." |
| R-85 | 85 | `pressure_applied == true and bleeding == "severe"` | "Keep applying firm pressure. Do not remove yet." |
| R-80 | 80 | `pressure_applied == true and bleeding == "reducing"` | "Bleeding is reducing — continue holding pressure." |
| R-75 | 75 | `bleeding == "controlled" and gauze_present == false` | "Bleeding is controlled. Apply a clean dressing/gauze now." |
| R-70 | 70 | `bleeding == "controlled" and gauze_present == true` | "Dressing applied. Monitor the wound and seek further care if needed." |
| R-10 | 10 | `wound_visible == true and bleeding in ["none","moderate"]` | "No significant active bleeding detected. Clean the area and monitor; apply a dressing if needed." |

`R-99` is checked immediately after wound visibility and before every bleeding-specific rule:
if the read isn't reliable, nothing downstream should be trusted enough to drive a specific
instruction, so confidence has to be resolved first, not last.

`R-97` only needs the `pressure_recently_removed` gate for `bleeding == "severe"`, because
that's the one case where a competing rule (`R-95`) is *logically identical* on `bleeding` and
`pressure_applied` alone — the recency flag is the only thing that can tell "pressure never
applied yet" apart from "pressure just taken away." For `bleeding == "reducing"`, no such
competing rule exists (`R-80` only matches when `pressure_applied == true`), so `R-96` doesn't
need the recency gate and fires any time pressure is off during a reducing trend — closing the
gap the gate would otherwise have left open.

`R-10` exists so the table is genuinely exhaustive over the `bleeding` enum from Section 3.5:
without it, a wound reading as `"none"` or `"moderate"` matched no rule at all and produced no
guidance — a silent gap that wouldn't surface in the staged demo script (which only walks
severe → reducing → controlled) but would surface under any off-script testing or ablation.

Every `(wound_visible, confidence, bleeding, pressure_applied, pressure_recently_removed,
gauze_present)` combination now matches exactly one top-priority rule — verified by
exhaustively enumerating the Facts space against this table rather than spot-checking it.

Example firing log entry (one line per guidance change, cheap to emit, valuable in results):
```json
{ "t": 184.2, "rule_fired": "R-97", "rule_table_version": "v4",
  "facts_snapshot": { "bleeding": "severe", "pressure_applied": false, "pressure_recently_removed": true },
  "prev_rule_fired": "R-85" }
```

**How reversal handling works (the core thesis, made explicit):** there is no dedicated
"went backward" logic anywhere. If pressure is removed mid-treatment, Facts simply update
(`pressure_applied → false`, `pressure_recently_removed → true`), and rule `R-97` (while
bleeding is `"severe"`) or `R-96` (while bleeding is `"reducing"`) naturally fires on the very
next cycle — because *all* rules are checked every time, not just the ones reachable from a
previous state. This is the mechanism, and it should be described exactly this way in your
report and defense. The firing log above is what turns "it should work this way" into a
citable trace from an actual recorded run.

### 3.7 Guidance Output

- **Primary:** on-screen text overlay (rendered on the video feed via OpenCV `putText` or a
  simple UI layer) — most reliable for a live demo.
- **Optional (if time allows):** `pyttsx3` offline text-to-speech, so guidance is audible
  without looking at the screen — nice to have, not required for the core thesis.

---

## 4. Protocol / Knowledge Sourcing (feeds the rule table above)

1. Manually collect first-aid steps for cuts/external bleeding from 2–3 reputable sources
   (Red Cross, Mayo Clinic, WHO/St. John Ambulance).
2. Feed the gathered text to an LLM and ask it to draft the rule table (condition → action,
   in the schema above) — this is the legitimate "LLM-assisted protocol engineering" piece of
   your original idea, scaled to prototype size.
3. Manually review the drafted rules against the source material before finalizing — this is
   your "medical validation" step, done by your team rather than an automated pipeline. Record
   which source justified each rule ID (a one-line mapping table is enough) so the protocol
   basis is auditable alongside the code.

---

## 5. Tech Stack (mapped to modules, prototype-feasible subset of your requirements slide)

| Module | Tool |
|---|---|
| Camera / CV | Python 3.11, OpenCV |
| Hand tracking | MediaPipe |
| Object detection upgrade (optional, months 3–4) | YOLOv11 + PyTorch, only if time allows |
| Observation Generator / Validator / State Builder / Rule Engine | Plain Python (dicts, simple classes) — no framework needed for a single-scenario prototype |
| Guidance output | OpenCV overlay; `pyttsx3` optional |
| Data/Numerics | NumPy, Pandas (for buffer/statistics, minimal use) |
| Backend/Frontend/DB (FastAPI, React, PostgreSQL) | Keep in the report as the **production-scale architecture**; not required for the working prototype demo itself. If a demo UI is wanted, a simple local script + OpenCV window is sufficient — build the FastAPI/React/DB layer only if time genuinely remains after the core pipeline works. |

---

## 6. Six-Month Timeline

**Months 1–2 (intensive build — core pipeline):**
- Weeks 1–2: HSV blood detection + MediaPipe hand tracking; validate Observation Generator
  output against real test footage (fake blood/red dye).
- Weeks 3–4: Manual protocol collection + LLM-assisted rule table drafting + team review.
- Weeks 5–6: Build Observation Validator + State Builder (temporal buffer, evidence fusion).
- Weeks 7–8: Build the Adaptive Workflow Engine (rule evaluation) and connect the full chain
  end-to-end with guidance overlay. First working prototype milestone.

**Months 3–4 (moderate effort — hardening):**
- Tune thresholds across lighting/angle variation (log each change against `threshold_set`
  version so calibration drift is traceable).
- Add/refine the low-confidence fallback and reversal-handling rules.
- Optional: YOLOv11 upgrade for detection, voice output.
- Begin report writing (background, related work, architecture sections).

**Months 5–6 (moderate effort — polish & documentation):**
- Run and record structured test scenarios (see Section 7).
- Ablate 2–3 individual rules against recorded footage and report the guidance moments each
  one was responsible for — this becomes your results/experiments section.
- Finalize report with results, limitations, and future work.
- Prepare and rehearse the live/recorded demo.

---

## 7. Test & Demo Script

A single staged sequence that proves the full pipeline and the dynamic-correction thesis:

1. Show clean arm → system says "No wound detected."
2. Reveal cut/red mark → system detects wound + severe bleeding → "Apply direct pressure."
3. Place hand/cloth on wound → system detects pressure applied → "Keep applying pressure."
4. **Hold pressure for at least ~3 seconds, then remove the hand/cloth mid-demo** → system
   immediately reverts to "Pressure was released — reapply pressure immediately." The few
   seconds of hold in step 3 is what satisfies `pressure_recently_removed`'s dwell-time check
   (Section 3.5) — a shorter hold could fail to trigger this exact rule. *(this is the key
   moment — record it clearly)*
5. Reapply pressure, reduce visible red (wipe/swap prop) → system detects "reducing" →
   "Continue holding pressure."
6. Further reduce blood visibility → "controlled" → "Apply a clean dressing."
7. Place gauze → system confirms → "Dressing applied, monitor the wound."

Record this as a backup video in case live camera/lighting fails during the actual
presentation. Keep the firing-log JSON (Section 3.6) alongside the video — it's your evidence
that the reversal in step 4 was rule-driven, not scripted.

---

## 8. Failure Modes (lightweight analog of the full architecture's failure-mode table)

| Failure | Where handled | Behavior |
|---|---|---|
| Momentary false blood/hand reading | Observation Validator (consistency/sanity checks) + State Builder buffer minimum-duration | Discarded or outvoted by buffer majority; logged with discard reason |
| Camera/lighting degrades mid-session | `detection_confidence` drop → Facts `confidence` drop | R-99 fires: "Unable to clearly assess" |
| Pressure removed mid-treatment | Facts update immediately, no dedicated "reversal" code path | R-97/R-96 fires next cycle (Section 3.6) |
| Detector swapped later (HSV → YOLOv11) | Fixed Observation Object schema (Section 3.3) | Downstream modules unaffected — this is the generalization claim, testable once the swap happens |

---

## 9. Future Work — and how it maps back to the full research architecture

- Multi-injury and multi-region support → requires reintroducing `track_id`/body-region
  concepts from the original architecture draft.
- Additional emergency types (burns, fractures, CPR) using the same architecture.
- YOLOv11/deep-learning-based wound and severity classification → the detector-swap experiment
  this revision's `detector_id` field was designed to support. If pursued (months 3–4, only
  after the HSV/MediaPipe pipeline works end-to-end):
  - **Data collection:** extract frames from your own recorded demo/test footage (Section 7),
    plus a handful of additional staged sessions across different lighting, skin tones, and
    fake-blood/prop variations to avoid overfitting to one recording setup. A few hundred
    labeled frames is a realistic target for a college-scale fine-tune, not thousands.
  - **Annotation:** bounding boxes for three classes — `wound`, `hand`, `gauze` — using a free
    tool (e.g. Roboflow's annotator or CVAT). Roboflow is the more convenient path since it also
    handles the train/val split and exports directly in YOLO format.
  - **Model choice:** fine-tune the smallest YOLOv11 checkpoint (`yolov11n`) rather than a
    larger variant — it trains faster on limited compute (a single consumer GPU or even
    Colab's free tier is enough for a 3-class fine-tune) and inference speed matters more than
    marginal accuracy for a real-time webcam demo.
  - **Training:** standard Ultralytics YOLO fine-tuning workflow (`pip install ultralytics`,
    then `yolo train model=yolov11n.pt data=<your-data.yaml> epochs=50-100`). Track a
    `model_version` string alongside `detector_id` so a specific checkpoint is reproducible in
    the results section, the same way `threshold_set` is tracked for the HSV thresholds.
  - **Evaluation:** report mAP@0.5 per class on a held-out validation split, plus a qualitative
    side-by-side against the HSV/MediaPipe baseline on the same recorded footage — this is what
    turns the "detector-swap" claim into an actual reported experiment rather than an assertion.
  - **Integration:** YOLOv11 replaces only the detection calls inside `3.2 CV Models` — it must
    still populate the same Observation Object fields (`blood_pixels_pct`,
    `wound_contour_present`, `cloth_overlap_pct`, etc., or their closest equivalents) with
    `detector_id` set to something like `"yolov11n_v1"`, so nothing in the Observation
    Generator, Validator, State Builder, or rule engine needs to change. That schema stability
    is the whole point of keeping this as a swap rather than a rewrite.
  - **Fallback:** keep the HSV/MediaPipe pipeline in the codebase and switchable via a config
    flag (`detector_id`) rather than deleting it — if the fine-tuned model underperforms on
    demo day (lighting, unseen background, etc.), you can fall back to the working baseline
    without a last-minute rewrite.
- Per-entity confidence and provenance, a full state ontology with a Memory Manager and
  hysteresis, and an evidence-scored State Inference Engine → the target architecture this
  prototype scopes down from; the rule-firing log and Facts `basis` fields in this revision are
  the minimum viable precursor to that evidence trail.
- Full production backend (FastAPI + PostgreSQL) and mobile/AR frontend (React).
- Formal clinical validation and regulatory considerations (e.g., Software as a Medical
  Device classification) before any real-world deployment.
- Automated multi-source research agent and provenance-tracked knowledge base for scaling
  protocol authoring beyond manual collection.
