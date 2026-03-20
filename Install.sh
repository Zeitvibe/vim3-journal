#!/bin/bash
# ==============================================
# NPU JOURNAL INSTALLER
# ZeitVibe - VIM3 Project
# ==============================================

set -e

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  NPU JOURNAL - INSTALLATION        ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
echo ""

# Detect shell
CURRENT_SHELL=$(basename "$SHELL")
if [ "$CURRENT_SHELL" = "zsh" ]; then
    RC_FILE="$HOME/.zshrc"
elif [ "$CURRENT_SHELL" = "bash" ]; then
    RC_FILE="$HOME/.bashrc"
else
    RC_FILE="$HOME/.profile"
fi

# Create directories
mkdir -p ~/.local/bin
mkdir -p ~/npu-journal

# Create journal script
cat > ~/.local/bin/journal << 'EOF'
#!/bin/bash
EVENT="${1:-Journal entry}"
TEMP_RAW=$(cat /sys/class/thermal/thermal_zone0/temp 2>/dev/null || echo "N/A")
if [ "$TEMP_RAW" != "N/A" ]; then
    TEMP_C=$((TEMP_RAW / 1000))
else
    TEMP_C="N/A"
fi
UPTIME=$(uptime -p 2>/dev/null | sed 's/up //' || echo "N/A")
KERNEL=$(uname -r)
NPU_STATUS=$(dmesg 2>/dev/null | grep -i npu | tail -1 || echo "Not found")
{
  echo "========================================"
  echo "Entry: $(date '+%Y-%m-%d %H:%M:%S')"
  echo "========================================"
  echo "Event: $EVENT"
  echo "Temperature: ${TEMP_C}°C"
  echo "Uptime: $UPTIME"
  echo "Kernel: $KERNEL"
  echo "NPU: $NPU_STATUS"
  echo "----------------------------------------"
} >> ~/npu-journal/learnings.md
echo ""
echo "✅ Journal entry added:"
tail -3 ~/npu-journal/learnings.md
echo ""
EOF

chmod +x ~/.local/bin/journal

# Add to PATH if needed
if [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$RC_FILE"
fi

# Add alias
if ! grep -q "alias journal=" "$RC_FILE" 2>/dev/null; then
    echo "" >> "$RC_FILE"
    echo "# NPU Journal" >> "$RC_FILE"
    echo "alias journal='~/.local/bin/journal'" >> "$RC_FILE"
fi

# Test
~/.local/bin/journal "NPU Journal installed"

echo ""
echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
echo -e "${BLUE}║  INSTALLATION COMPLETE              ║${NC}"
echo -e "${BLUE}║                                     ║${NC}"
echo -e "${BLUE}║  Run: source $RC_FILE               ║${NC}"
echo -e "${BLUE}║  Then: journal \"event\"            ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
