#!/bin/bash

# Array of IP addresses or interface names to test
interfaces=("127.0.0.1" "enp3s0f0" "enp3s0f1")
data_size="1000M"  # Amount of data to transfer in megabytes
log_file="network_test.log"

# Calculate timeout based on data size
# Assuming a conservative transfer speed of 100 MB/s, adjust if needed
transfer_speed=100  # MB per second
timeout=$(echo "scale=0; $data_size / ${transfer_speed} + 5" | bc)

# Initialize log file
echo "Starting network interface diagnostics..." | tee "$log_file"
echo "Timestamp: $(date)" | tee -a "$log_file"

# Function to display progress bar
show_progress() {
    local progress=0
    while [ $progress -le 100 ]; do
        printf "\rTesting... [%-50s] %d%%" $(head -c $((progress / 2)) < /dev/zero | tr '\0' '=') "$progress"
        sleep 0.1
        ((progress+=10))
    done
    echo ""
}

# Function to test connectivity
test_interface() {
    local iface=$1
    echo "Testing interface $iface..." | tee -a "$log_file"

    # Show progress bar
    show_progress

    if [ "$iface" == "127.0.0.1" ]; then
        ping -c 4 $iface > /dev/null 2>&1
    else
        # Use ip to get the IP of the interface
        ip_addr=$(ip -4 addr show $iface | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
        
        if [ -n "$ip_addr" ]; then
            echo "Pinging $iface at IP $ip_addr..." | tee -a "$log_file"
            ping -c 4 $ip_addr > /dev/null 2>&1
        else
            echo "No IP found for interface $iface." | tee -a "$log_file"
            return 1
        fi
    fi

    if [ $? -eq 0 ]; then
        echo "Interface $iface is reachable." | tee -a "$log_file"
    else
        echo "Interface $iface is not reachable." | tee -a "$log_file"
    fi
}

# Function to test data transfer throughput between detected interfaces
test_data_throughput() {
    local interfaces=("$@")
    if [ ${#interfaces[@]} -le 1 ]; then
        echo "Not enough interfaces for data transfer test." | tee -a "$log_file"
        return 1
    fi
    
    echo "Testing data throughput between interfaces..." | tee -a "$log_file"

    for ((i = 0; i < ${#interfaces[@]} - 1; i++)); do
        iface1=${interfaces[$i]}
        iface2=${interfaces[$i + 1]}
        
        # Log interfaces being tested
        echo "Setting up data transfer from $iface1 to $iface2" | tee -a "$log_file"

        # Start listener with calculated timeout in the background
        echo "$(date): Starting listener on $iface2 for data transfer with $timeout-second timeout" | tee -a "$log_file"
        (timeout "$timeout" nc -l 12345 > /dev/null &)

        sleep 1  # Ensure listener is ready

        # Log data transfer initiation
        echo "$(date): Transferring $data_size from $iface1 to $iface2 with throughput display..." | tee -a "$log_file"
        dd if=/dev/zero bs=1M count=$(( ${data_size%M} )) 2>/dev/null | pv -s "$data_size" | nc localhost 12345

        # Check for success/failure of data transfer
        if [ $? -eq 0 ]; then
            echo "$(date): Data successfully sent from $iface1 to $iface2." | tee -a "$log_file"
        else
            echo "$(date): Data transfer between $iface1 and $iface2 failed." | tee -a "$log_file"
        fi

        # Ensure all netcat processes are terminated after each test
        echo "Cleaning up any background processes for $iface2" | tee -a "$log_file"
        pkill -f "nc -l 12345"  # Kill listener if still running
    done
}

# Loop through each interface and test connectivity
for iface in "${interfaces[@]}"; do
    test_interface "$iface"
done

# Run data throughput test if multiple interfaces detected
test_data_throughput "${interfaces[@]}"

echo "Diagnostics completed. Check $log_file for detailed logs."
