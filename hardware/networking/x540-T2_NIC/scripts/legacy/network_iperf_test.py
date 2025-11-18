import psutil
import subprocess
import time
import socket

# Define the interfaces to test
interfaces = ["enp3s0f0", "enp3s0f1"]  # Replace with your actual interface names

# Function to get IP address of an interface
def get_ip_address(iface):
    addrs = psutil.net_if_addrs()
    if iface in addrs:
        for addr in addrs[iface]:
            if addr.family == socket.AF_INET:
                return addr.address
    return None

# Function to run iperf test
def run_iperf_test(server_ip, client_ip):
    print(f"Starting iperf server on {server_ip}...")
    server_cmd = f"iperf -s -B {server_ip} -i 1"
    server_process = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    time.sleep(2)  # Wait for server to start
    
    print(f"Starting iperf client to {server_ip} from {client_ip}...")
    client_cmd = f"iperf -c {server_ip} -B {client_ip} -t 10"
    client_process = subprocess.Popen(client_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    stdout, stderr = client_process.communicate()
    print(stdout.decode())
    
    server_process.terminate()
    server_process.wait()

# Main function to orchestrate the test
def main():
    if len(interfaces) != 2:
        print("Exactly two interfaces are required for this test.")
        return
    
    ip1 = get_ip_address(interfaces[0])
    ip2 = get_ip_address(interfaces[1])
    
    if not ip1 or not ip2:
        print("Could not retrieve IP addresses for one or both interfaces.")
        return
    
    print(f"Testing data transfer between {interfaces[0]} ({ip1}) and {interfaces[1]} ({ip2})")
    
    # Run iperf test
    run_iperf_test(ip1, ip2)
    run_iperf_test(ip2, ip1)

if __name__ == "__main__":
    main()