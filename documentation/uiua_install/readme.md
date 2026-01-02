# Uiua Installation & GLIBC 2.39 Update Document

**Date:** January 2, 2026  
**System:** Ubuntu 22.04.5 LTS (Jammy)  
**Architecture:** x86_64

## Overview

This document details the complete process for installing Uiua binary and upgrading GLIBC from 2.35 to 2.39 on Ubuntu 22.04 LTS, which was necessary because the pre-compiled Uiua binary requires GLIBC 2.39+.

---

## Phase 1: Uiua Binary Download & Verification

### Step 1: Download Uiua Binary

**Source:** https://github.com/uiua-lang/uiua/releases

**Command:**
```bash
cd /tmp
wget https://github.com/uiua-lang/uiua/releases/download/latest/uiua-bin-x86_64-unknown-linux-gnu.zip
```

**Details:**
- File: `uiua-bin-x86_64-unknown-linux-gnu.zip`
- Size: 24.66 MB (24660918 bytes)
- Download speed: ~10.1 MB/s

### Step 2: Verify SHA256 Checksum

**Expected SHA256:** `5d94ccce7c34d6b2b771e8cc4a57e87883ad4a56557488fd34dfdf715e5f0a68`

**Command:**
```bash
sha256sum uiua-bin-x86_64-unknown-linux-gnu.zip
```

**Output:**
```
5d94ccce7c34d6b2b771e8cc4a57e87883ad4a56557488fd34dfdf715e5f0a68  uiua-bin-x86_64-unknown-linux-gnu.zip
```

✓ **Verification Successful**

### Step 3: Extract Binary

**Command:**
```bash
unzip -o uiua-bin-x86_64-unknown-linux-gnu.zip
```

**Output:**
```
Archive:  uiua-bin-x86_64-unknown-linux-gnu.zip
  inflating: uiua
```

### Step 4: Make Executable

**Command:**
```bash
chmod +x uiua
```

**Verification:**
```bash
ls -lh uiua
# Output: -rwxr-xr-x 1 jeb jeb 60M Dec 30 15:16 uiua
```

---

## Phase 2: GLIBC Compatibility Issue & Resolution

### Compatibility Problem

**Initial Issue:**
```
./uiua: /lib/x86_64-linux-gnu/libc.so.6: version `GLIBC_2.39' not found
```

**Root Cause:**
- System GLIBC: 2.35 (Ubuntu 22.04 LTS maximum)
- Uiua requires: GLIBC 2.39+
- Ubuntu 22.04 repositories don't provide GLIBC 2.39

**System Info:**
```bash
lsb_release -a
# Distributor ID: Ubuntu
# Description:    Ubuntu 22.04.5 LTS
# Codename:       jammy

ldd --version | head -1
# ldd (Ubuntu GLIBC 2.35-0ubuntu3.11) 2.35
```

---

## Phase 3: GLIBC 2.39 Compilation & Installation

### Step 1: Download GLIBC 2.39 Source

**Source:** https://ftpmirror.gnu.org/glibc/

**Command:**
```bash
cd /tmp
wget https://ftpmirror.gnu.org/glibc/glibc-2.39.tar.gz
```

**Details:**
- File: `glibc-2.39.tar.gz`
- Size: 36.7 MB (36704719 bytes)
- Download speed: ~7.68 MB/s

### Step 2: Extract & Create Build Directory

**Commands:**
```bash
tar -xzf glibc-2.39.tar.gz
cd glibc-2.39
mkdir build
cd build
```

### Step 3: Configure

**Command:**
```bash
../configure --prefix=/opt/glibc-2.39
```

**Key Compilation Flags Applied:**
- Installation prefix: `/opt/glibc-2.39`
- Architecture: x86_64
- CET support enabled
- PIE (Position Independent Executable) enabled
- CFProtection enabled

### Step 4: Build

**Command:**
```bash
make -j$(nproc)
```

**Build Time:** ~2-3 minutes (on multi-core system)

**Verification:**
```
make[1]: Leaving directory '/tmp/glibc-2.39'
[Build finished with exit code: 0]
```

### Step 5: Install

**Command:**
```bash
sudo make install
```

**Installation Details:**
- Installed to: `/opt/glibc-2.39`
- Key files:
  - `/opt/glibc-2.39/lib/libc.so.6`
  - `/opt/glibc-2.39/lib/ld-linux-x86-64.so.2` (dynamic linker)
  - `/opt/glibc-2.39/lib64/` (64-bit libraries)
  - Supporting headers and tools

---

## Phase 4: Uiua Binary Setup with GLIBC 2.39

### Issue: Direct Binary Execution Fails

**Problem:**
The Uiua binary is hardcoded to use the system dynamic linker at `/lib64/ld-linux-x86-64.so.2`, which loads the system GLIBC 2.35 instead of our new GLIBC 2.39.

### Solution: Wrapper Script

**File:** `/home/jeb/programs/uiua-wrapper.sh`

**Content:**
```bash
#!/bin/bash
exec /opt/glibc-2.39/lib/ld-linux-x86-64.so.2 --library-path /opt/glibc-2.39/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu /home/jeb/programs/uiua "$@"
```

**Explanation:**
- Uses GLIBC 2.39's dynamic linker explicitly
- Sets library path to prioritize GLIBC 2.39 libraries
- Falls back to system libraries for dependencies not in GLIBC 2.39
- Passes all arguments through to the actual Uiua binary

**Make Executable:**
```bash
chmod +x /home/jeb/programs/uiua-wrapper.sh
```

### Shell Alias Configuration

**File:** `~/.zshrc`

**Addition:**
```bash
alias uiua='/home/jeb/programs/uiua-wrapper.sh'
```

**Verification:**
```bash
source ~/.zshrc
which uiua
# Output: uiua: aliased to /home/jeb/programs/uiua-wrapper.sh

uiua --version
# Output: uiua 0.18.0-dev.4
```

---

## Final Status

### Installation Summary

| Component | Status | Location |
|-----------|--------|----------|
| Uiua Binary | ✓ Installed | `/home/jeb/programs/uiua` |
| Binary Verification | ✓ SHA256 Verified | - |
| GLIBC 2.39 | ✓ Built & Installed | `/opt/glibc-2.39` |
| Wrapper Script | ✓ Created | `/home/jeb/programs/uiua-wrapper.sh` |
| Shell Alias | ✓ Configured | `~/.zshrc` |

### Testing

**Command:**
```bash
uiua --version
```

**Output:**
```
uiua 0.18.0-dev.4
```

✓ **Installation Successful**

---

## Maintenance & Updates

### Future GLIBC Updates

If you need to update GLIBC further:

1. Download new GLIBC version
2. Build with `--prefix=/opt/glibc-X.YZ`
3. Update wrapper script library path
4. Test Uiua binary

### Troubleshooting

**If Uiua fails to run:**

```bash
# Test with explicit dynamic linker
/opt/glibc-2.39/lib/ld-linux-x86-64.so.2 --library-path /opt/glibc-2.39/lib:/lib/x86_64-linux-gnu:/usr/lib/x86_64-linux-gnu /home/jeb/programs/uiua --version
```

**If missing library errors occur:**

1. Identify missing library: `ldd /home/jeb/programs/uiua`
2. Install system package: `sudo apt install lib<name>`
3. Add path to wrapper script if needed

---

## Documentation References

- Uiua Project: https://www.uiua.org/
- Uiua Releases: https://github.com/uiua-lang/uiua/releases
- GLIBC Project: https://sourceware.org/glibc/
- GLIBC 2.39 Release: https://www.sourceware.org/glibc/wiki/Release/2.39

---

**Document Generated:** 2026-01-02  
**Completion Status:** ✓ Complete
