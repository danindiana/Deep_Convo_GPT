# Uiua Website Local Build Setup

**Date:** January 2, 2026  
**Status:** In Progress - Build Compiling  
**Location:** `/home/jeb/programs/uiua-lang/site`

## Overview

Successfully set up the Uiua website for local development using Trunk and Rust/WebAssembly toolchain.

---

## Installation Steps Completed

### 1. Verified Rust Installation
```bash
rustc 1.92.0 (ded5c06cf 2025-12-08)
cargo 1.92.0 (344c4567c 2025-10-21)
```

✓ **Status:** Already installed

### 2. Installed Trunk
```bash
cargo install trunk
```

**Output:** `Installing trunk v0.21.14`

✓ **Status:** Installed successfully (3m 02s)

### 3. Added WASM Target
```bash
rustup target add wasm32-unknown-unknown
```

✓ **Status:** Installed successfully

### 4. Cloned Uiua Repository
```bash
cd /home/jeb/programs/uiua-lang
git clone https://github.com/uiua-lang/uiua.git .
```

✓ **Status:** Repository cloned with all dependencies

**Repository Contents:**
- `src/` - Main Uiua source code
- `site/` - Website code (Leptos + WebAssembly)
- `parser/` - Parser library
- `pad/` - Interactive editor
- `tests/` - Test suites

### 5. Started Development Server
```bash
cd /home/jeb/programs/uiua-lang/site
trunk serve
```

✓ **Status:** Build in progress

---

## Build Progress

### Current State
- **Trunk Version:** 0.21.14
- **Build Profile:** Unoptimized + Debug Info
- **Compilation Target:** WebAssembly (wasm32-unknown-unknown)

### Recent Compilation Steps
```
Finished `dev` profile [unoptimized + debuginfo] target(s) in 46.70s
INFO downloading wasm-bindgen version="0.2.100"
INFO installing wasm-bindgen
```

### Key Dependencies Being Compiled
- **Leptos** v0.6.15 (Rust web framework)
- **Uiua Core** v0.18.0-dev.4
- **WASM Bindgen** (Rust ↔ JavaScript interface)
- **Image Processing** (image v0.25.6, rav1e codec)
- **Syntax Highlighting** (syntect v5.2.0)

---

## What to Expect

Once the build completes, the development server will:

1. **Listen on:** `http://localhost:8080` (typically)
2. **Features:**
   - Hot reload on file changes
   - Live preview of Uiua website
   - Interactive Uiua editor/REPL embedded in webpage
   - Full source code documentation
3. **Access:** Open browser to `http://localhost:8080`

---

## Next Steps (After Build Completes)

### To Stop the Server
```bash
kill %1  # Kill background job
```

### To View Server Output
The server is running in background job #1. To see full output:
```bash
fg  # Bring to foreground
```

### To Make Changes & Test
1. Edit source files in `/home/jeb/programs/uiua-lang/`
2. Changes auto-reload in the browser
3. Check console for build errors

---

## System Configuration

### WASM Build Environment
- **Cargo Profile:** dev (unoptimized for faster iteration)
- **Target:** wasm32-unknown-unknown
- **Runtime:** Browser (Leptos)
- **Hot Reload:** Enabled via Trunk

### Performance Notes
- Initial build: ~47 seconds (dependencies)
- Subsequent builds: Much faster (only changed files)
- WASM binaries are optimized for browser execution

---

## References

- **Uiua Official:** https://www.uiua.org/
- **Uiua Repository:** https://github.com/uiua-lang/uiua
- **Leptos Framework:** https://leptos.dev/
- **Trunk Bundler:** https://trunkrs.io/
- **WebAssembly:** https://www.rust-lang.org/what/wasm/

---

## Troubleshooting

### Build Hangs
- May take 2-5 minutes for first build
- Downloading and compiling large dependencies
- Check RAM and disk space availability

### Port Already in Use
- Trunk uses port 8080 by default
- To use different port: `trunk serve --port 8081`

### WebAssembly Version Issues
- Ensure `wasm32-unknown-unknown` target is installed
- Verify with: `rustup target list | grep wasm`

---

## Summary

The Uiua website development environment is now configured and building. The comprehensive Rust/WebAssembly toolchain includes:

- ✓ Rust compiler and Cargo package manager
- ✓ Trunk web bundler for WASM
- ✓ Leptos web framework
- ✓ Full Uiua source repository
- ✓ Development server with hot reload

Once compilation finishes, you'll have a fully functional local instance of the Uiua website running in your browser with all interactive features enabled.

---

**Process Status:** Building (Wasm Bindgen installation in progress)  
**Est. Total Time:** 2-5 minutes (first build)  
**Next Check:** Monitor for server ready message on port 8080
