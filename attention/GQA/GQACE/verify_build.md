Below is a bash script to verify and ensure all necessary dependencies are installed on the target host system for compiling and running the generated C programs. The script checks for the presence of essential tools (like GCC and Make), installs them if missing, and confirms the system meets the requirements.

### Script: `verify_dependencies.sh`

```bash
#!/bin/bash

# Function to check if a command exists
check_command() {
    command -v "$1" >/dev/null 2>&1
}

# Function to install missing packages
install_package() {
    echo "Installing $1..."
    sudo apt update -y && sudo apt install -y "$1"
}

echo "Starting dependency verification for target host system..."

# Verify GCC (GNU Compiler Collection)
if check_command gcc; then
    echo "GCC is installed."
else
    echo "GCC is not installed. Installing GCC..."
    install_package gcc
fi

# Verify Make
if check_command make; then
    echo "Make is installed."
else
    echo "Make is not installed. Installing Make..."
    install_package make
fi

# Verify presence of build-essential (includes GCC and other tools)
if dpkg -s build-essential >/dev/null 2>&1; then
    echo "build-essential package is installed."
else
    echo "build-essential package is not installed. Installing it..."
    install_package build-essential
fi

# Verify libc-dev (C standard library development headers)
if dpkg -s libc6-dev >/dev/null 2>&1; then
    echo "libc-dev package is installed."
else
    echo "libc-dev package is not installed. Installing it..."
    install_package libc6-dev
fi

# Optional: Check for additional useful utilities
if check_command nano; then
    echo "Nano text editor is installed."
else
    echo "Nano text editor is not installed. Installing it..."
    install_package nano
fi

if check_command vim; then
    echo "Vim text editor is installed."
else
    echo "Vim text editor is not installed. Installing it..."
    install_package vim
fi

# Display system information
echo "System Information:"
echo "Kernel Version: $(uname -r)"
echo "Operating System: $(lsb_release -d | cut -f2)"
echo "Architecture: $(uname -m)"

echo "Dependency verification complete. Your system is ready."
```

---

### **How to Use the Script**

1. Save the script as `verify_dependencies.sh`.
2. Make it executable:
   ```bash
   chmod +x verify_dependencies.sh
   ```
3. Run the script:
   ```bash
   ./verify_dependencies.sh
   ```

---

### **What the Script Does**
1. **Checks for Required Tools:**
   - Verifies if `gcc` and `make` are installed using the `command -v` command.
   - Checks for `build-essential`, which includes compilers and essential libraries.
   - Ensures `libc6-dev`, which provides the C standard library headers, is present.

2. **Installs Missing Tools:**
   - Uses `sudo apt install` to install any missing tools.
   - Updates the package list first with `sudo apt update`.

3. **Provides System Information:**
   - Displays kernel version, operating system, and architecture to confirm system compatibility.

4. **Optional Utilities:**
   - Checks for text editors (`nano` and `vim`) for convenience.

---

### **Example Output**
```plaintext
Starting dependency verification for target host system...
GCC is installed.
Make is installed.
build-essential package is installed.
libc-dev package is installed.
Nano text editor is installed.
Vim text editor is installed.
System Information:
Kernel Version: 5.15.0-79-generic
Operating System: Ubuntu 22.04.3 LTS
Architecture: x86_64
Dependency verification complete. Your system is ready.
```

This script ensures the target system has all the tools required to compile and run C programs while providing an easy-to-follow setup process.
