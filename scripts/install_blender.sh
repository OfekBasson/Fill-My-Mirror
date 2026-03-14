#!/usr/bin/env bash
set -e

BLENDER_VERSION="4.4.3"
BLENDER_FILE="blender-${BLENDER_VERSION}-linux-x64.tar.xz"
BLENDER_URL="https://download.blender.org/release/Blender4.4/${BLENDER_FILE}"

INSTALL_DIR="external/blender"

echo "Installing Blender ${BLENDER_VERSION}..."

mkdir -p ${INSTALL_DIR}

cd ${INSTALL_DIR}

if [ -d "blender-${BLENDER_VERSION}-linux-x64" ]; then
    echo "Blender already installed."
    exit 0
fi

echo "Downloading Blender..."
wget ${BLENDER_URL}

echo "Extracting Blender..."
tar -xf ${BLENDER_FILE}

rm ${BLENDER_FILE}

echo "Blender installed at:"
echo "${INSTALL_DIR}/blender-${BLENDER_VERSION}-linux-x64/"

echo ""
echo "Test with:"
echo "./external/blender/blender-${BLENDER_VERSION}-linux-x64/blender --version"