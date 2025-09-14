#!/usr/bin/env bash
set -euo pipefail

targetdir=target

if [ ! -d "$targetdir" ]; then
  mkdir -p "$targetdir"
fi

javac -encoding UTF-8 -sourcepath src \
  -d "$targetdir" \
  -cp "lib/ECLA.jar:lib/DTNConsoleConnection.jar" \
  src/core/*.java \
  src/movement/*.java \
  src/report/*.java \
  src/routing/*.java \
  src/gui/*.java \
  src/input/*.java \
  src/applications/*.java \
  src/interfaces/*.java

if [ ! -d "$targetdir/gui/buttonGraphics" ]; then
  mkdir -p "$targetdir/gui"
  cp -R src/gui/buttonGraphics "$targetdir/gui/"
fi
