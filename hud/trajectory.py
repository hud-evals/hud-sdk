import datetime
from typing import Optional
from pydantic import BaseModel, Field

from IPython.display import display, Markdown, HTML

class TrajectoryStep(BaseModel):
    """Model representing a single task run's trajectory information."""

    observation_url: Optional[str] = None
    observation_text: Optional[str] = None
    actions: list[dict]
    start_timestamp: Optional[str] = None
    end_timestamp: Optional[str] = None


class Trajectory(BaseModel):
    """Model representing a single task run's trajectory information."""

    id: str
    reward: Optional[float] = None
    logs: Optional[str] = None
    error: Optional[str] = None
    trajectory: list[TrajectoryStep] = Field(default_factory=list)

    def display(self):
        trajectory_start_timestamp = self.trajectory[0].start_timestamp
        t_start_dt = datetime.datetime.fromisoformat(trajectory_start_timestamp.replace('Z', '+00:00')) if trajectory_start_timestamp else None
        for i, step in enumerate(self.trajectory):
            # Use Markdown for better step separation in Jupyter
            display(Markdown(f"### Step {i + 1}"))

            # Observation Image
            if step.observation_url:
                try:
                    # Display in Jupyter/IPython environment using HTML
                    display(Markdown(f"**Observation Image:**"))
                    display(HTML(f'<img src="{step.observation_url}" style="max-width:100%;"/>'))
                    display(Markdown(f"[Image Link]({step.observation_url})"))
                except Exception as e:
                    print(f"    [Error processing image: {e}]")
            elif not step.observation_text: # Only print if no image AND no text
                 print("    No visual or text observation provided.")


            # Observation Text
            if step.observation_text:
                print(f"    Observation Text: {step.observation_text}")

            # Actions
            print(f"\n    Actions: {step.actions}") # Added newline for spacing

            # Duration
            duration_str = "N/A"
            step_start_timestamp = self.trajectory[i].start_timestamp
            step_end_timestamp = self.trajectory[i].end_timestamp
            if step_start_timestamp and step_end_timestamp and t_start_dt:
                try:
                    # Attempt to parse timestamps (assuming ISO format)
                    start_dt = datetime.datetime.fromisoformat(step_start_timestamp.replace('Z', '+00:00'))
                    end_dt = datetime.datetime.fromisoformat(step_end_timestamp.replace('Z', '+00:00'))
                    duration = end_dt - start_dt
                    total_seconds = duration.total_seconds()
                    minutes = int(total_seconds // 60)
                    seconds = total_seconds % 60
                    duration_str = f"{minutes}m {seconds:.2f}s"

                    # Calculate the total duration up to this step
                    total_duration = end_dt - t_start_dt
                    total_minutes = int(total_duration.total_seconds() // 60)
                    total_seconds = total_duration.total_seconds() % 60
                    total_duration_str = f"{total_minutes}m {total_seconds:.2f}s"
                except ValueError:
                    duration_str = "Error parsing timestamps" # Handle potential format issues
            print(f"    Step Duration: {duration_str}")
            print(f"    Total Duration: {total_duration_str}")
            display(Markdown("---")) # Use Markdown horizontal rule