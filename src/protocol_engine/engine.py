from src.cv_engine.action_verifier import ActionVerifier

class FirstAidProtocolEngine:
    def __init__(self):
        self.verifier = ActionVerifier()
        self.steps = ["DETECT_WOUND", "APPLY_PRESSURE", "PROTOCOL_COMPLETE"]
        self.current_step_index = 0
        self.pressure_start_time = None
        self.required_duration = 5.0 

    def process_state(self, state_payload):
        current_step = self.steps[self.current_step_index]
        instruction = ""
        timestamp = state_payload.get("timestamp", 0)

        if current_step == "DETECT_WOUND":
            # FIX: Read from scene_understanding.injuries
            injuries = state_payload.get("scene_understanding", {}).get("injuries", [])
            wound_detected = any(
                obs["object_type"] in ["wound", "cut", "bleeding"] 
                for obs in injuries
            )
            
            if wound_detected:
                self.current_step_index += 1
                instruction = "Wound detected. Apply continuous pressure."
            else:
                instruction = "Scanning for injuries... Please show the wound."

        elif current_step == "APPLY_PRESSURE":
            is_pressing, _ = self.verifier.is_applying_pressure(state_payload)
            
            if is_pressing:
                if self.pressure_start_time is None:
                    self.pressure_start_time = timestamp
                
                elapsed_time = timestamp - self.pressure_start_time
                remaining_time = max(0, self.required_duration - elapsed_time)
                
                if remaining_time == 0:
                    self.current_step_index += 1
                    instruction = "Excellent. Pressure verified and sustained."
                else:
                    instruction = f"Keep holding... {remaining_time:.1f} seconds left."
            else:
                if self.pressure_start_time is not None:
                    self.pressure_start_time = None
                    instruction = "Pressure lost! Reapply pressure immediately."
                else:
                    instruction = "Place your hand directly over the red wound box."

        elif current_step == "PROTOCOL_COMPLETE":
            instruction = "First aid completed. Seek medical attention if needed."

        return {
            "current_step": current_step,
            "system_instruction": instruction
        }