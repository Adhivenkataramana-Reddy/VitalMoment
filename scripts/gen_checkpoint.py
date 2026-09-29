from fpdf import FPDF
import os

class PDF(FPDF):
    def header(self):
        self.set_font("helvetica", "B", 16)
        self.cell(0, 10, "VitalMoment - Project Checkpoint & Codebase", align="C", new_x="LMARGIN", new_y="NEXT")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("helvetica", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}", align="C")

def main():
    pdf = PDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=10)

    # The project summary and file map
    document_text = """
1. PROJECT OVERVIEW
VitalMoment is an offline-first AI platform for emergency first-aid.
- Perception Layer: YOLOv8 (wound detection) + MediaPipe (3D hand tracking) -> outputs Semantic JSON.
- Reasoning Layer: Medical Protocol Engine that verifies user actions via mathematical intersection.

2. ENVIRONMENT SETUP
- OS: Windows
- Python: 3.10.11 (Virtual Environment: venv)
- Dependencies: mediapipe==0.10.9, ultralytics, opencv-python

3. DIRECTORY STRUCTURE
D:\\VitalMoment\\
|-- data\\
|    |-- yolo_formatted\\ (Roboflow dataset)
|-- models\\
|    |-- wound_detector\\weights\\best.pt
|-- scripts\\
|    |-- test_live_feed.py
|    |-- train_yolo.py
|-- src\\
  |-- cv_engine\\
  |    |-- anatomy_tracker.py
  |    |-- object_detector.py
  |    |-- state_builder.py
  |    |-- action_verifier.py
  |-- protocol_engine\\
       |-- engine.py

4. NEXT DEVELOPMENT STEPS (PHASE 2 & 3)
- Occlusion Problem: Add spatial memory so the AI remembers the wound when the hand covers it.
- Multi-Class Ontology: Add "bandage" and "gauze" to the Roboflow dataset and retrain YOLO.
- 3D Depth Checking: Use MediaPipe's Z-axis data to ensure the hand is touching the body.
    """

    # Write text to PDF
    pdf.multi_cell(0, 6, document_text)
    
    # Save the file to the project root
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    output_path = os.path.join(project_root, "VitalMoment_Progress.pdf")
    
    pdf.output(output_path)
    print(f"✅ Success! PDF generated at: {output_path}")

if __name__ == "__main__":
    main()