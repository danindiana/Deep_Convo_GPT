#!/bin/bash

# Enhanced NIC Diagnostic Script for Intel X540-T2
# Version 2.0 - Modernized with better error handling and reporting

set -euo pipefail  # Exit on error, undefined variables, and pipe failures

# Configuration
readonly SCRIPT_NAME="$(basename "$0")"
readonly SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
readonly DATA_SIZE="1000M"
readonly TRANSFER_SPEED=100  # MB/s
readonly TIMEOUT=$((${DATA_SIZE%M} / TRANSFER_SPEED + 5))
readonly LOG_FILE="${LOG_FILE:-network_test.log}"
readonly TEST_PORT="${TEST_PORT:-12345}"

# Colors for output
readonly COLOR_RESET='\033[0m'
readonly COLOR_RED='\033[0;31m'
readonly COLOR_GREEN='\033[0;32m'
readonly COLOR_YELLOW='\033[1;33m'
readonly COLOR_BLUE='\033[0;34m'
readonly COLOR_CYAN='\033[0;36m'

# Function to print colored messages
log_info() {
    echo -e "${COLOR_BLUE}[INFO]${COLOR_RESET} $*" | tee -a "$LOG_FILE"
}

log_success() {
    echo -e "${COLOR_GREEN}[SUCCESS]${COLOR_RESET} $*" | tee -a "$LOG_FILE"
}

log_warning() {
    echo -e "${COLOR_YELLOW}[WARNING]${COLOR_RESET} $*" | tee -a "$LOG_FILE"
}

log_error() {
    echo -e "${COLOR_RED}[ERROR]${COLOR_RESET} $*" | tee -a "$LOG_FILE"
}

log_header() {
    echo -e "${COLOR_CYAN}==== $* ====${COLOR_RESET}" | tee -a "$LOG_FILE"
}

# Function to check dependencies
check_dependencies() {
    local deps=("ip" "ping" "nc" "pv" "dd" "bc")
    local missing=()

    for dep in "${deps[@]}"; do
        if ! command -v "$dep" &> /dev/null; then
            missing+=("$dep")
        fi
    done

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        log_info "Install with: sudo apt-get install ${missing[*]}"
        exit 1
    fi
}

# Function to display progress bar
show_progress() {
    local duration=${1:-2}
    local progress=0

    while [ $progress -le 100 ]; do
        local bar_length=$((progress / 2))
        printf "\r${COLOR_CYAN}Testing...${COLOR_RESET} ["
        printf "%-50s" "$(head -c $bar_length < /dev/zero | tr '\0' '=')"
        printf "] %d%%" "$progress"
        sleep $(echo "scale=2; $duration / 100" | bc)
        ((progress+=10))
    done
    echo ""
}

# Function to discover network interfaces
discover_interfaces() {
    log_header "Discovering Network Interfaces"

    local interfaces=()
    while IFS= read -r iface; do
        # Skip loopback
        if [ "$iface" = "lo" ]; then
            continue
        fi

        # Check if interface has an IP
        if ip -4 addr show "$iface" | grep -q "inet "; then
            interfaces+=("$iface")
            local ip_addr=$(ip -4 addr show "$iface" | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
            log_info "Found interface: $iface (IP: $ip_addr)"
        fi
    done < <(ip -o link show | awk -F': ' '{print $2}' | grep -v '@')

    if [ ${#interfaces[@]} -eq 0 ]; then
        log_error "No configured network interfaces found"
        exit 1
    fi

    echo "${interfaces[@]}"
}

# Function to test interface connectivity
test_interface() {
    local iface=$1
    log_header "Testing Interface: $iface"

    # Get IP address
    local ip_addr=$(ip -4 addr show "$iface" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)

    if [ -z "$ip_addr" ]; then
        log_error "No IP address found for $iface"
        return 1
    fi

    log_info "Interface $iface has IP: $ip_addr"

    # Test connectivity with ping
    log_info "Pinging $ip_addr..."
    if ping -c 4 -W 2 "$ip_addr" > /dev/null 2>&1; then
        log_success "Interface $iface is reachable"
        return 0
    else
        log_warning "Interface $iface is not reachable via ping"
        return 1
    fi
}

# Function to test data throughput
test_throughput() {
    local src_iface=$1
    local dst_iface=$2

    log_header "Testing Throughput: $src_iface -> $dst_iface"

    # Get IP addresses
    local src_ip=$(ip -4 addr show "$src_iface" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)
    local dst_ip=$(ip -4 addr show "$dst_iface" | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | head -1)

    if [ -z "$src_ip" ] || [ -z "$dst_ip" ]; then
        log_error "Could not determine IP addresses"
        return 1
    fi

    log_info "Source: $src_iface ($src_ip)"
    log_info "Destination: $dst_iface ($dst_ip)"

    # Start listener
    log_info "Starting listener on $dst_ip:$TEST_PORT (timeout: ${TIMEOUT}s)"
    timeout "$TIMEOUT" nc -l "$dst_ip" "$TEST_PORT" > /dev/null 2>&1 &
    local listener_pid=$!

    sleep 2  # Give listener time to start

    # Transfer data
    log_info "Transferring $DATA_SIZE from $src_ip to $dst_ip"

    if dd if=/dev/zero bs=1M count="${DATA_SIZE%M}" 2>/dev/null | \
       pv -s "$DATA_SIZE" -p -t -e -r | \
       nc -w 10 "$dst_ip" "$TEST_PORT"; then
        log_success "Data transfer completed successfully"
    else
        log_error "Data transfer failed"
        kill "$listener_pid" 2>/dev/null || true
        return 1
    fi

    # Wait for listener to finish
    wait "$listener_pid" 2>/dev/null || true

    return 0
}

# Function to check interface status
check_interface_status() {
    local iface=$1

    log_header "Interface Status: $iface"

    # Link status
    if ip link show "$iface" | grep -q "state UP"; then
        log_success "Link is UP"
    else
        log_warning "Link is DOWN"
    fi

    # Speed (if ethtool is available)
    if command -v ethtool &> /dev/null; then
        local speed=$(ethtool "$iface" 2>/dev/null | grep -oP '(?<=Speed: )\d+')
        if [ -n "$speed" ]; then
            log_info "Link speed: ${speed}Mb/s"
        fi
    fi

    # Statistics
    log_info "Interface statistics:"
    ip -s link show "$iface" | tail -2 | tee -a "$LOG_FILE"
}

# Main function
main() {
    log_header "Intel X540-T2 NIC Diagnostics"
    log_info "Starting diagnostics at $(date)"
    log_info "Log file: $LOG_FILE"

    # Check dependencies
    check_dependencies

    # Discover interfaces
    local interfaces=($(discover_interfaces))
    log_info "Found ${#interfaces[@]} interface(s): ${interfaces[*]}"

    # Test each interface
    for iface in "${interfaces[@]}"; do
        test_interface "$iface"
        check_interface_status "$iface"
    done

    # Test throughput between interfaces
    if [ ${#interfaces[@]} -ge 2 ]; then
        log_header "Throughput Testing"
        for ((i = 0; i < ${#interfaces[@]} - 1; i++)); do
            test_throughput "${interfaces[$i]}" "${interfaces[$i+1]}"
        done
    else
        log_warning "Need at least 2 interfaces for throughput testing"
    fi

    log_header "Diagnostics Complete"
    log_success "All tests finished. Check $LOG_FILE for detailed logs."
}

# Cleanup function
cleanup() {
    log_info "Cleaning up..."
    # Kill any remaining netcat processes
    pkill -f "nc -l $TEST_PORT" 2>/dev/null || true
}

# Set trap for cleanup
trap cleanup EXIT INT TERM

# Run main function
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi
