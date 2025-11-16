import socket
import time
import logging
import subprocess
from tqdm import tqdm
import psutil

# Configuration
DATA_SIZE_MB = 500  # Size of data to transfer for testing (in MB)
BLOCK_SIZE = 1024 * 1024  # Transfer in 1 MB chunks
LOG_FILE = "x540t2_network_test.log"
LISTENER_TIMEOUT = 15  # Increased timeout for listener in seconds
MAX_RETRIES = 3  # Number of retries for establishing the listener

# Configure logging to both console and file
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
file_handler = logging.FileHandler(LOG_FILE)
file_handler.setLevel(logging.INFO)
file_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
file_handler.setFormatter(file_formatter)
logging.getLogger().addHandler(file_handler)

def discover_ports():
    """Automatically discover IPs of Intel X540-T2 NIC ports."""
    logging.info("Starting interface discovery for Intel X540-T2 NIC...")
    intel_ports = []
    for iface_name, iface_addrs in psutil.net_if_addrs().items():
        logging.info(f"Checking interface {iface_name} for Intel NIC...")
        if "eth" in iface_name or "enp" in iface_name:
            for addr in iface_addrs:
                if addr.family == socket.AF_INET:
                    intel_ports.append((iface_name, addr.address))
                    logging.info(f"Discovered Intel port: {iface_name} with IP {addr.address}")
                    break
    
    if len(intel_ports) >= 2:
        logging.info("Successfully discovered Intel X540-T2 NIC ports.")
        return intel_ports[:2]
    else:
        logging.error("Failed to discover two Intel NIC ports. Check NIC configuration.")
        raise RuntimeError("Failed to discover two Intel NIC ports. Check NIC configuration.")

def ping_test(ip):
    """Ping a given IP to check connectivity."""
    logging.info(f"Pinging {ip} to check connectivity...")
    response = subprocess.run(['ping', '-c', '4', ip], stdout=subprocess.PIPE)
    if response.returncode == 0:
        logging.info(f"Ping successful to {ip}")
        return True
    else:
        logging.error(f"Ping failed to {ip}")
        return False

def start_listener(ip, port):
    """Set up a listener socket to receive data with retry logic."""
    for attempt in range(MAX_RETRIES):
        logging.info(f"Setting up listener on {ip}:{port} (Attempt {attempt + 1}/{MAX_RETRIES})...")
        listener_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener_socket.bind((ip, port))
        listener_socket.listen(1)
        listener_socket.settimeout(LISTENER_TIMEOUT)

        try:
            logging.info(f"Listening on {ip}:{port} for incoming data...")
            conn, addr = listener_socket.accept()
            logging.info(f"Connection accepted from {addr}")
            return conn
        except socket.timeout:
            logging.warning(f"Listener timed out after {LISTENER_TIMEOUT} seconds on {ip}:{port}")
        finally:
            listener_socket.close()

    logging.error(f"Failed to establish listener connection after {MAX_RETRIES} attempts.")
    return None

def send_data(ip, port, data_size_mb):
    """Send data to the specified IP and port, measuring throughput."""
    logging.info(f"Attempting to connect to {ip}:{port} for data transfer of {data_size_mb}MB...")
    time.sleep(2)  # Delay to ensure listener is ready

    sender_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sender_socket.connect((ip, port))
        logging.info(f"Connected to {ip}:{port}, starting data transfer.")
        total_sent = 0
        start_time = time.time()
        with tqdm(total=data_size_mb, unit='MB', desc="Data Transfer") as pbar:
            while total_sent < data_size_mb * BLOCK_SIZE:
                chunk = b'\0' * BLOCK_SIZE
                sender_socket.sendall(chunk)
                total_sent += len(chunk)
                pbar.update(1)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        throughput = data_size_mb / elapsed_time
        logging.info(f"Data transfer complete. Throughput: {throughput:.2f} MB/s")
        return throughput
    except Exception as e:
        logging.error(f"Error during data transfer: {e}")
    finally:
        sender_socket.close()

def receive_data(conn, data_size_mb):
    """Receive data from a socket and measure performance."""
    if not conn:
        logging.error("No connection established for receiving data.")
        return
    logging.info(f"Receiving data of {data_size_mb}MB...")
    total_received = 0
    with tqdm(total=data_size_mb, unit='MB', desc="Receiving Data") as pbar:
        while total_received < data_size_mb * BLOCK_SIZE:
            data = conn.recv(BLOCK_SIZE)
            if not data:
                break
            total_received += len(data)
            pbar.update(1)
    conn.close()
    logging.info("Data reception complete.")

# Run the tests
try:
    # Discover the NIC ports automatically
    ports = discover_ports()
    port_a, port_b = ports[0][1], ports[1][1]
    logging.info(f"Ports discovered: {ports[0][0]} ({port_a}), {ports[1][0]} ({port_b})")

    # Check connectivity with ping
    if ping_test(port_b):
        logging.info("Starting data transfer test between ports...")
        
        # Set up listener on Port B with retry mechanism
        listener_conn = start_listener(port_b, 12345)
        
        # Start sending data if listener is ready
        if listener_conn:
            throughput = send_data(port_b, 12345, DATA_SIZE_MB)
            receive_data(listener_conn, DATA_SIZE_MB)
            logging.info(f"Test completed. Achieved throughput: {throughput:.2f} MB/s")
            print(f"Test completed. Achieved throughput: {throughput:.2f} MB/s")
        else:
            logging.error("Failed to establish listener connection.")
            print("Failed to establish listener connection.")
    else:
        logging.error("Ping test failed. Ensure both ports are configured correctly.")
        print("Ping test failed. Ensure both ports are configured correctly.")
except RuntimeError as e:
    logging.error(f"Error: {e}")
    print(f"Error: {e}")
