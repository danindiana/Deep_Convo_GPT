Here is a list of Arduino Uno 328P IO mappings to the elements found within the `pulse_spike_train_speed_metrics_arduino_io.mmd` Mermaid diagram. Each element from the diagram is mapped to specific Arduino pins and functionalities:

### Arduino Uno 328P IO Mappings

1. **Common Metrics IO Board**
   - **Firing Rate IO Board**
     - **Mean Firing Rate Calculation IO**
       - Collect spike data: **Digital Input (D2)**
       - Count total spikes: **Digital Input (D3)**
       - Divide spike count by time period: **Analog Input (A0)**
     - **Instantaneous Firing Rate Calculation IO**
       - Define smoothing window: **Analog Input (A1)**
       - Slide window across data: **Digital Output (D4)**
       - Count spikes in window: **Digital Input (D5)**
       - Normalize by window size: **Analog Output (A2)**

   - **Interspike Interval IO Board**
     - (Reserved for future use)
     
   - **Burst Metrics IO Board**
     - (Reserved for future use)

   - **Synchrony Metrics IO Board**
     - (Reserved for future use)

   - **Spike Train Metrics IO Board**
     - (Reserved for future use)

2. **Choosing the Right Metric IO Board**
   - **Interest in Overall Firing Rate IO**
     - Mean Firing Rate: **Digital Input (D6)**
   - **Interest in Temporal Patterns and Variability IO**
     - ISICV: **Analog Input (A3)**
     - Burst Metrics: **Digital Input (D7)**

3. **Biological Algorithms IO Board**
   - **Mean Firing Rate Calculation IO**
     - Collect spike data: **Digital Input (D8)**
     - Count total spikes: **Digital Input (D9)**
     - Divide spike count by time period: **Analog Input (A4)**
   - **Instantaneous Firing Rate Calculation IO**
     - Define smoothing window: **Analog Input (A5)**
     - Slide window across data: **Digital Output (D10)**
     - Count spikes in window: **Digital Input (D11)**
     - Normalize by window size: **Analog Output (A6)**

### Pin Mapping Summary
- **Digital Inputs**: D2, D3, D5, D6, D7, D8, D9, D11
- **Digital Outputs**: D4, D10
- **Analog Inputs**: A0, A1, A3, A4, A5
- **Analog Outputs**: A2, A6

By mapping the elements to specific Arduino pins, we can see how each part of the pulse spike train speed metrics system could be implemented on an Arduino Uno board. This mapping helps to visualize the hardware setup and the corresponding IO operations for each component.
