import tensorflow as tf
from typing import List, Dict

# Define the steps for each algorithm as TensorFlow operations
def create_step(description: str) -> tf.Tensor:
    return tf.constant(description)

# Define the algorithms using TensorFlow functions
def mean_firing_rate_calculation() -> Dict[str, tf.Tensor]:
    steps = [
        "Collect spike data over a defined period",
        "Count the total number of spikes",
        "Divide the total spike count by the time period"
    ]
    return {step: create_step(step) for step in steps}

def instantaneous_firing_rate_calculation() -> Dict[str, tf.Tensor]:
    steps = [
        "Define a smoothing window or kernel",
        "Slide the window across the spike data",
        "Count spikes within each window",
        "Normalize by the window size"
    ]
    return {step: create_step(step) for step in steps}

# Define the main structure for common metrics and choosing the right metric
class CommonMetrics:
    def __init__(self):
        self.firing_rate: Dict[str, Dict[str, tf.Tensor]] = {
            "MeanFiringRate": mean_firing_rate_calculation(),
            "InstantaneousFiringRate": instantaneous_firing_rate_calculation()
        }
        self.interspike_interval: Dict[str, Dict[str, tf.Tensor]] = {}
        self.burst_metrics: Dict[str, Dict[str, tf.Tensor]] = {}
        self.synchrony_metrics: Dict[str, Dict[str, tf.Tensor]] = {}
        self.spike_train_metrics: Dict[str, Dict[str, tf.Tensor]] = {}

class ChoosingTheRightMetric:
    def __init__(self):
        self.interest_in_overall_firing_rate: str = "MeanFiringRate"
        self.interest_in_temporal_patterns_and_variability: List[str] = ["ISICV", "BurstMetrics"]

# Define the main structure for pulse spike train speed metrics
class PulseSpikeTrainSpeedMetrics:
    def __init__(self):
        self.common_metrics: CommonMetrics = CommonMetrics()
        self.choosing_the_right_metric: ChoosingTheRightMetric = ChoosingTheRightMetric()
        self.biological_algorithms: Dict[str, Dict[str, tf.Tensor]] = {
            "MeanFiringRateCalculation": mean_firing_rate_calculation(),
            "InstantaneousFiringRateCalculation": instantaneous_firing_rate_calculation()
        }

# Function to print the details of an algorithm
def print_algorithm(name: str, steps: Dict[str, tf.Tensor]):
    print(f"Algorithm: {name}")
    for step, description in steps.items():
        print(f"- {description.numpy().decode('utf-8')}")
    print()

# Main function to create and display the pulse spike train speed metrics
def main():
    pulse_metrics = PulseSpikeTrainSpeedMetrics()
    
    print("Biological Algorithms:")
    for name, steps in pulse_metrics.biological_algorithms.items():
        print_algorithm(name, steps)

if __name__ == "__main__":
    main()
