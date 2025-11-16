This guide provides an excellent high-level overview and practical steps to set up GPU passthrough for KVM/QEMU, allowing a virtual machine (VM) to directly access and use the physical GPU of the host machine. This is a valuable technique for scenarios that require high GPU performance, such as gaming, CUDA-based development, or machine learning.

Here’s a concise breakdown of the key components and steps:

### **Key Components:**
1. **Hardware Requirements:**
   - CPU with **Intel VT-d** or **AMD-Vi** support.
   - Motherboard that supports **IOMMU**.
   - Two GPUs: One for the host machine and one for the guest VM.

2. **Software Requirements:**
   - **KVM/QEMU** installed for virtualization.
   - **VFIO (Virtual File IO)** driver for GPU passthrough.
   - Optionally, **Libvirt** for managing VMs with XML configurations.

### **Steps to Set Up GPU Passthrough for KVM/QEMU:**

#### 1. **Enable IOMMU in BIOS and Kernel**
   - For Intel, enable **VT-d** in the BIOS.
   - For AMD, enable **AMD-Vi/IOMMU** in the BIOS.
   
   Then, add the IOMMU support in your bootloader (e.g., GRUB):
   ```bash
   GRUB_CMDLINE_LINUX_DEFAULT="quiet splash intel_iommu=on"
   ```
   Update GRUB and reboot:
   ```bash
   sudo update-grub
   sudo reboot
   ```

#### 2. **Install Required Packages**
   On Ubuntu or other Debian-based systems:
   ```bash
   sudo apt install qemu-kvm libvirt-bin virt-manager ovmf
   sudo apt install qemu-system-x86 qemu-utils qemu-kvm
   ```

#### 3. **Configure VFIO and Bind GPU**
   Use the `lspci` command to identify the GPU’s PCI addresses:
   ```bash
   lspci -nn | grep -i nvidia
   ```
   Create a modprobe file to bind the GPU to the `vfio-pci` driver:
   ```bash
   sudo nano /etc/modprobe.d/vfio.conf
   ```
   Add the PCI addresses of the GPU and its HDMI audio device:
   ```
   options vfio-pci ids=10de:1e82,10de:10f0
   ```

#### 4. **Update Initramfs**
   Update the initramfs to apply the changes:
   ```bash
   sudo update-initramfs -u
   sudo reboot
   ```

#### 5. **Create a Virtual Machine with GPU Passthrough**
   Use **virt-manager** or the **QEMU command-line** to create a VM with GPU passthrough.

   **Example QEMU command-line:**
   ```bash
   qemu-system-x86_64 \
       -enable-kvm \
       -m 8G \
       -cpu host,kvm=off,hv_vendor_id=1234567890ab \
       -smp cores=4 \
       -device vfio-pci,host=09:00.0,x-vga=on \
       -device vfio-pci,host=09:00.1 \
       -drive file=your-vm-disk.img,format=qcow2 \
       -boot menu=on
   ```
   - **vfio-pci,host=09:00.0**: Pass through the GPU (adjust to your GPU's PCI address).
   - **x-vga=on**: Enables VGA mode for the GPU.

#### 6. **Install Guest Drivers**
   After setting up the VM, install the NVIDIA drivers in the guest OS to enable proper GPU utilization.

### **Troubleshooting Common Issues:**
1. **PCI Resource Allocation Errors:**
   If you see PCI resource allocation errors, check your host's IOMMU groups to ensure that the GPU is in a separate group from other devices.

2. **Error 43 in Windows Guests:**
   In the case of Windows guests, you might encounter **Error 43** (common for NVIDIA GPUs). To bypass this, use the `hv_vendor_id` flag in the QEMU command-line:
   ```bash
   -cpu host,kvm=off,hv_vendor_id=1234567890ab
   ```

3. **Enable 64-bit PCI Memory Addressing:**
   If you need to pass through devices that require 64-bit PCI memory, make sure the kernel and QEMU are configured to support this.

### **Using `libvirt` for Easier Management:**
You can also configure GPU passthrough using **libvirt** by editing the XML configuration of the virtual machine. This simplifies the setup and management compared to manual QEMU command lines.

### Example XML Snippet for libvirt:
```xml
<hostdev mode='subsystem' type='pci' managed='yes'>
  <source>
    <address domain='0x0000' bus='0x09' slot='0x00' function='0x0'/>
  </source>
  <address type='pci' domain='0x0000' bus='0x00' slot='0x09' function='0x0'/>
</hostdev>
```

This setup would pass through the GPU at the specified PCI address.

### **Final Notes:**
- Ensure that the IOMMU group is isolated to avoid conflicts.
- Performance may vary depending on the virtualization overhead, but GPU passthrough via KVM/QEMU generally provides near-native performance.
- **Libvirt** makes it easier to manage complex configurations, and using tools like **virt-manager** provides a graphical interface for controlling VMs with GPU passthrough.

This guide sets you up for GPU passthrough with a focus on flexibility and performance in virtualized environments. Would you like assistance with any of the specific steps or tools mentioned?


Enabling GPU passthrough for a Vagrant setup requires additional configuration on both the hypervisor and Vagrant side. Here's a step-by-step guide on how to enable GPU passthrough for `vagrant up` using VirtualBox as an example. Note that GPU passthrough works best with hypervisors like KVM, but VirtualBox can work with certain limitations.

### 1. **Ensure GPU passthrough is supported by your hardware**
   - **CPU:** Your CPU must support Intel VT-d or AMD-Vi for GPU passthrough.
   - **Motherboard:** Make sure the IOMMU is enabled in the BIOS.
     - On Intel systems, look for Intel VT-d.
     - On AMD systems, look for AMD-Vi or IOMMU.

### 2. **Configure GPU passthrough in VirtualBox**

   VirtualBox does not support true GPU passthrough, but it does have basic support for exposing your host's GPU to the guest machine through 3D acceleration. This will not work for CUDA applications, but it may work for less demanding graphical tasks.

   **Steps to enable GPU in VirtualBox:**

   1. Open VirtualBox and select the VM that you use with Vagrant.
   2. Click on **Settings** → **Display**.
   3. Under the **Screen** tab, increase **Video Memory** to the maximum.
   4. Enable **3D Acceleration** by checking the box.
   5. Optionally, you can enable **2D Video Acceleration**.

   If you need CUDA support or want a true GPU passthrough, I recommend switching to KVM/QEMU, which supports passthrough directly.

### 3. **Configure Vagrant for GPU support**

   You need to adjust the `Vagrantfile` to configure the VM to use 3D acceleration and access the GPU as much as VirtualBox allows.

   Here is how you can modify your `Vagrantfile`:

   ```ruby
   Vagrant.configure("2") do |config|
     config.vm.box = "ubuntu/bionic64"  # Use the appropriate box for your setup.

     config.vm.provider "virtualbox" do |vb|
       vb.memory = "4096"
       vb.cpus = 4

       # Enable GPU passthrough / 3D acceleration
       vb.customize ["modifyvm", :id, "--graphicscontroller", "vmsvga"]
       vb.customize ["modifyvm", :id, "--accelerate3d", "on"]
       vb.customize ["modifyvm", :id, "--vram", "128"]  # Adjust the video RAM
     end
   end
   ```

   After making this change, run `vagrant up` to bring up your machine with the GPU acceleration settings applied.

### 4. **If using KVM/QEMU (for true GPU passthrough)**

   **Steps to enable GPU passthrough on KVM/QEMU:**
   1. Install QEMU/KVM on your system:
      ```bash
      sudo apt install qemu-kvm libvirt-bin virt-manager
      ```
   2. Use `virt-manager` to create a new VM and configure it with GPU passthrough. This can be done by passing the GPU device directly to the guest VM under the "Add Hardware" section.
   3. To automate the VM setup with Vagrant, you can use the `vagrant-libvirt` provider and ensure that the GPU passthrough settings are added in the XML configuration for the VM.

   Example for a `Vagrantfile` using `libvirt`:

   ```ruby
   Vagrant.configure("2") do |config|
     config.vm.box = "generic/ubuntu2004"
     config.vm.provider :libvirt do |libvirt|
       libvirt.memory = 4096
       libvirt.cpus = 4
       libvirt.pci :device => "01:00.0"  # Adjust to your GPU's PCI address
     end
   end
   ```

   In this case, you will need to adjust the PCI address (`01:00.0`) to match the address of your GPU. You can find the GPU's PCI address using the `lspci` command:
   ```bash
   lspci | grep -i vga
   ```

### 5. **Test GPU passthrough**

   Once your VM is up and running, you can verify that GPU passthrough is working by installing `nvidia-smi` and testing it inside your VM:
   ```bash
   sudo apt update
   sudo apt install nvidia-driver-535
   nvidia-smi
   ```

### Notes:
- VirtualBox's GPU passthrough capabilities are limited compared to KVM/QEMU, which is recommended for high-performance GPU tasks.
- For CUDA applications, direct hardware passthrough with KVM or similar hypervisors is essential.

Would you like to explore the KVM/QEMU setup in more detail or proceed with the VirtualBox method?



To configure Vagrant to expose one or both of the NVIDIA GPUs (GeForce RTX 3080 and GeForce RTX 3060) to a virtual machine, you'll need to modify your `Vagrantfile`. Since VirtualBox has limitations when it comes to GPU passthrough, we'll focus on enabling 3D acceleration for each GPU separately.

However, please note that due to the limitations of VirtualBox, this setup might not work for CUDA applications or other demanding tasks that require direct access to the GPUs. For such use cases, consider using KVM/QEMU as mentioned earlier.

Here's an example `Vagrantfile` configuration that enables 3D acceleration for each GPU:

```ruby
Vagrant.configure("2") do |config|
  config.vm.box = "ubuntu/bionic64"  # Use the appropriate box for your setup.
  config.vm.provider "virtualbox" do |vb|
    vb.memory = "4096"
    vb.cpus = 4

    # Enable GPU passthrough / 3D acceleration for each GPU
    vb.customize ["modifyvm", :id, "--graphicscontroller", "vmsvga"]
    vb.customize ["modifyvm", :id, "--accelerate3d", "on"]

    # Adjust the video RAM for each GPU
    # For GPU 0 (RTX 3080): 128 MB
    vb.customize ["modifyvm", :id, "--vram", "128"]
    # For GPU 1 (RTX 3060): 64 MB
    vb.customize ["modifyvm", :id, "--vram", "64"]

    # Identify the PCI addresses of each GPU
    gpu_addresses = {
      "GPU 0" => "00000000:0D:00.0",
      "GPU 1" => "00000000:0E:00.0"
    }

    gpu_addresses.each do |name, pci_address|
      # Enable PCI passthrough for each GPU
      vb.customize ["modifyvm", :id, "--pci", "#{pci_address}:0"]
    end
  end
end
```

In this example, we've added the necessary configurations to enable 3D acceleration and expose both GPUs to the virtual machine. Please note that you'll need to adjust the PCI addresses (`gpu_addresses`) according to your system's configuration.

After modifying your `Vagrantfile`, run `vagrant up` to bring up your machine with the GPU acceleration settings applied. However, as mentioned earlier, this setup might not work for CUDA applications or other demanding tasks due to VirtualBox's limitations.






Your explanation is correct, but I would like to clarify that VirtualBox does not support true GPU passthrough, especially for high-performance tasks like CUDA-based applications or deep learning tasks that need direct access to the GPU hardware. Therefore, even if you enable 3D acceleration, this setup won't allow you to run CUDA tasks effectively.

If you're looking to utilize the full power of your NVIDIA GPUs (GeForce RTX 3080 and GeForce RTX 3060) within a virtualized environment, I recommend switching to KVM/QEMU with GPU passthrough. Here's a high-level summary of what you'd need to do with KVM/QEMU:

### Using KVM/QEMU for True GPU Passthrough

1. **Enable IOMMU on Your System:**
   - For **Intel** systems, enable VT-d in the BIOS.
   - For **AMD** systems, enable AMD-Vi (IOMMU) in the BIOS.

2. **Install the Necessary Packages:**
   On Ubuntu, you can install the required packages with the following commands:
   ```bash
   sudo apt update
   sudo apt install qemu-kvm libvirt-bin virt-manager
   sudo apt install ovmf  # If you want to enable UEFI booting
   ```

3. **Configure GPU Passthrough:**
   - Add your GPU to the virtual machine through **virt-manager** or manually through the XML configuration for the VM.
   - Find your GPU’s PCI address using `lspci`:
     ```bash
     lspci -nn | grep -i nvidia
     ```
     This will give you the PCI address of your NVIDIA GPUs (e.g., `0000:0d:00.0` and `0000:0e:00.0`).

4. **Edit the Virtual Machine XML:**
   Using **virt-manager** or by editing the XML directly, add the PCI devices for GPU passthrough.

   Example XML snippet:
   ```xml
   <hostdev mode='subsystem' type='pci' managed='yes'>
     <source>
       <address domain='0x0000' bus='0x0D' slot='0x00' function='0x0'/>
     </source>
     <address type='pci' domain='0x0000' bus='0x00' slot='0x09' function='0x0'/>
   </hostdev>

   <hostdev mode='subsystem' type='pci' managed='yes'>
     <source>
       <address domain='0x0000' bus='0x0E' slot='0x00' function='0x0'/>
     </source>
     <address type='pci' domain='0x0000' bus='0x00' slot='0x0A' function='0x0'/>
   </hostdev>
   ```

5. **Start Your VM:**
   After configuring GPU passthrough, start your virtual machine, and you should have access to both GPUs from within the VM.

With **KVM/QEMU**, you can fully utilize the power of your GPUs for demanding tasks like deep learning, gaming, or any GPU-accelerated computations. Would you like more details on setting up KVM/QEMU for GPU passthrough?
