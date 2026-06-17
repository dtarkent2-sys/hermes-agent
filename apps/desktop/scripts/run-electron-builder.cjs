#!/usr/bin/env node
"use strict"

// Run electron-builder with build.electronDist resolved at runtime instead of
// hardcoded in package.json.
//
// Why this exists (the bug it kills for good):
//   electron-builder 26.8.x can stage an Electron.app without its main
//   MacOS/Electron binary when it re-unpacks Electron from its own cache
//   (#38673), so the desktop build pins build.electronDist at the *already
//   unpacked* electron package's dist to make electron-builder reuse it. But
//   electronDist in package.json is a static relative path, and npm workspace
//   hoisting is NOT deterministic: depending on the npm version and what else
//   is installed, npm may nest the workspace-only electron devDep under
//   apps/desktop/node_modules/electron OR hoist it to the repo-root
//   node_modules/electron. A static path matches only one layout, so a clean
//   install intermittently fails with "The specified electronDist does not
//   exist" (the June desktop-build outage: #47917, #48019, #48021, #48084).
//
// The fix: resolve the electron package the same way Node's runtime does —
// require.resolve("electron/package.json") walks node_modules from this script
// upward and finds electron wherever npm actually put it. The path can never
// drift out of sync with the install layout again, on any OS or npm version.
//
//   - dist present  -> pass -c.electronDist=<abs>/dist so electron-builder
//                      reuses the unpacked runtime (keeps the #38673 fast path
//                      that dodges the 26.8.x missing-binary re-unpack bug).
//   - dist absent   -> omit electronDist entirely and let electron-builder
//                      fetch Electron itself via @electron/get (honoring
//                      build.electronVersion + any ELECTRON_MIRROR). This is
//                      the network-blocked / skipped-postinstall case; the
//                      mac-binary prebuilder patch is the backstop for it.

const fs = require("node:fs")
const path = require("node:path")
const { spawnSync } = require("node:child_process")

function electronDistDir() {
  let pkgJson
  try {
    pkgJson = require.resolve("electron/package.json")
  } catch {
    return null
  }
  return path.join(path.dirname(pkgJson), "dist")
}

function distBinary(dist) {
  if (process.platform === "darwin") {
    return path.join(dist, "Electron.app", "Contents", "MacOS", "Electron")
  }
  if (process.platform === "win32") {
    return path.join(dist, "electron.exe")
  }
  return path.join(dist, "electron")
}

function electronBuilderCli() {
  const pkgJson = require.resolve("electron-builder/package.json")
  const bin = require(pkgJson).bin
  const rel = typeof bin === "string" ? bin : bin["electron-builder"]
  return path.join(path.dirname(pkgJson), rel)
}

const passthrough = process.argv.slice(2)
const args = []

const dist = electronDistDir()
if (dist && fs.existsSync(distBinary(dist))) {
  // Absolute so electron-builder never re-resolves it against the app dir.
  args.push(`-c.electronDist=${dist}`)
} else {
  console.warn(
    "[run-electron-builder] electron dist not found on disk; letting " +
      "electron-builder fetch Electron via @electron/get (electronVersion + " +
      "ELECTRON_MIRROR apply)."
  )
}
args.push(...passthrough)

const result = spawnSync(process.execPath, [electronBuilderCli(), ...args], {
  stdio: "inherit",
})
if (result.error) {
  console.error(`[run-electron-builder] failed to launch electron-builder: ${result.error.message}`)
  process.exit(1)
}
process.exit(result.status == null ? 1 : result.status)
