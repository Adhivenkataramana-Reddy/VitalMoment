# VitalMoment — Adaptive CV-Based Emergency Care Prototype

VitalMoment is an **offline-first research prototype** for staged cuts/external-bleeding first-aid assistance. It continuously observes a webcam, converts visual measurements into validated observations, builds a time-windowed Facts state, and re-evaluates a priority-ordered adaptive rule table every cycle.

> **Safety:** this is a college/research prototype, not a medical device or substitute for professional emergency care. The visual detector uses staged red/blue props and can make mistakes. In a real emergency, follow local emergency guidance and seek professional help.

## Architecture

`Webcam → OpenCV HSV detector + MediaPipe Hands → Observation Generator → Validator → State Builder → Adaptive Rule Engine → On-screen Guidance`

The baseline intentionally follows the project's locked v6 workflow. YOLO is **not required to run the prototype**; the supplied archive did not contain the checkpoint referenced by the previous code.

## Requirements

- Windows/macOS/Linux
- Python 3.10–3.12 recommended
- Webcam
- Internet only for the initial `pip install` (runtime inference is local/offline)

## Setup — Windows PowerShell

```powershell
cd VitalMoment
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

## Run

```powershell
python main.py
```

If your webcam is not camera 0:

```powershell
python main.py --camera 1
```

Press **Q** to stop. A `firing_log.json` file is written at shutdown. This log records rule transitions and the Facts snapshot that caused each transition, which is useful for the project demo/report.

## Test without a webcam

```powershell
python -m unittest discover -s tests -v
```

The tests cover the observation validator, time-window state builder, pressure reversal behavior, and the documented rule table including the key `R-97` reversal case.

## Demo sequence

1. Show a clean arm/background → `R-100` asks you to position the camera.
2. Show a staged red wound/mark → severe bleeding eventually produces `R-95`.
3. Hold the index fingertip over the wound for about 2 seconds → `R-85` keeps pressure on.
4. Remove pressure → `R-97` detects the reversal when the bleeding state remains severe.
5. Reduce the visible red area over time → the state can move toward `reducing`/`controlled`.
6. Place the blue/green staged cloth over the wound → `gauze_present` becomes true after sustained evidence and `R-70` can fire once bleeding is controlled.

For a reliable presentation, use a clearly visible staged red prop and a saturated blue/green cloth. Lighting, camera angle, skin/background color, and hand occlusion can affect the heuristic detector.

## Optional YOLO training

The archive contains a small YOLO-format wound dataset. It is an **optional experiment**, not a runtime dependency. The baseline remains the fallback if training is unavailable.

```powershell
python -m pip install ultralytics
python scripts/train_yolo.py
```

The training script uses the dataset's actual one-class ontology (`wound`) and saves a checkpoint under `models/wound_detector/weights/best.pt`.

## Important project status

The original archive had a mismatch between `state_builder.py` and the available files: it tried to load `runs/detect/models/wound_detector/weights/best.pt`, but no such checkpoint was included. It also imported LangGraph in `main.py` even though LangGraph was not required by the locked workflow. This version removes those runtime blockers and implements the documented adaptive rule engine directly.


## MediaPipe compatibility

VitalMoment pins MediaPipe `0.10.21` because this prototype uses the classic `mp.solutions.hands` API. Newer MediaPipe releases may not expose `mp.solutions`. The application also includes an OpenCV fallback tracker, so the app can start even if MediaPipe is unavailable.

If you previously installed a newer MediaPipe version, reinstall the pinned dependencies from a clean virtual environment:

```powershell
python -m pip uninstall -y mediapipe numpy
python -m pip install -r requirements.txt
```

The OpenCV fallback is suitable for prototype/demo contact estimation; MediaPipe is preferred for more stable hand landmarks.

## Runtime notes

- Python 3.13 is supported by the project runtime configuration.
- The bleeding detector uses HSV + red-dominance + local-contrast checks and a skin-region guard. This prevents broad normal skin/facial regions from being interpreted as a wound.
- MediaPipe is optional. If `mediapipe` is unavailable or its API does not expose `mp.solutions`, the application uses the built-in OpenCV fallback for hand/contact estimation.
- Run the automated tests with `PYTHONPATH=. python -m unittest discover -s tests -p "test_*.py" -v`.
- The CV detector is a prototype and must be validated against the actual camera/video conditions before treating its output as reliable medical detection.
