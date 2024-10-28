import psutil
import subprocess
import time
import socket
import logging
import argparse
import threading

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_ip_address(iface):
    addrs = psutil.net_if_addrs()
    if iface in addrs:
        for addr in addrs[iface]:
            if addr.family == socket.AF_INET:
                return addr.address
    logging.warning(f"Could not retrieve IP address for {iface}")
    return None

def run_iperf_test(server_ip, client_ip):
    try:
        logging.info(f"Starting iperf server on {server_ip}...")
        server_cmd = f"iperf -s -B {server_ip} -i 1"
        server_process = subprocess.Popen(server_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        time.sleep(2)  # Wait for server to start
        
        logging.info(f"Starting iperf client to {server_ip} from {client_ip}...")
        client_cmd = f"iperf -c {server_ip} -B {client_ip} -t 10"
        client_process = subprocess.Popen(client_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        
        stdout, stderr = client_process.communicate()
        logging.info(stdout.decode())
        
        server_process.terminate()
        server_process.wait()
    except Exception as e:
        logging.error(f"Error running iperf test: {e}")

def discover_interfaces():
    interfaces = psutil.net_if_addrs().keys()
    return [iface for iface in interfaces if iface != 'lo']  # Exclude loopback interface

def display_menu(interfaces):
    print("Select interfaces to test (enter numbers separated by spaces):")
    for idx, iface in enumerate(interfaces, start=1):
        print(f"{idx}. {iface}")

def main():
    parser = argparse.ArgumentParser(description="Network iperf test script")
    parser.add_argument('--interfaces', nargs='*', help="Network interfaces to test")
    args = parser.parse_args()

    if args.interfaces:
        interfaces = args.interfaces
    else:
        interfaces = discover_interfaces()
        display_menu(interfaces)
        selected_indices = input("Enter the numbers of the interfaces to test: ").strip().split()
        selected_interfaces = [interfaces[int(idx) - 1] for idx in selected_indices]

    if len(selected_interfaces) < 2:
        logging.error("At least two interfaces are required for this test.")
        return
    
    ips = []
    valid_interfaces = []
    for iface in selected_interfaces:
        ip = get_ip_address(iface)
        if ip:
            ips.append(ip)
            valid_interfaces.append(iface)
        else:
            logging.warning(f"Skipping interface {iface} due to missing IP address.")
    
    if len(ips) < 2:
        logging.error("At least two valid interfaces with IP addresses are required for this test.")
        return
    
    logging.info(f"Testing data transfer between {valid_interfaces[0]} ({ips[0]}) and {valid_interfaces[1]} ({ips[1]})")
    
    # Use threading to run tests concurrently
    thread1 = threading.Thread(target=run_iperf_test, args=(ips[0], ips[1]))
    thread2 = threading.Thread(target=run_iperf_test, args=(ips[1], ips[0]))
    
    thread1.start()
    thread2.start()
    
    thread1.join()
    thread2.join()

if __name__ == "__main__":
    main()