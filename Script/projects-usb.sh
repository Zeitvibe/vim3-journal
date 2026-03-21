#!/bin/bash
# ZeitVibe USB Project Manager
# Manages VIM3-Projects USB drive

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m'
BOLD='\033[1m'

USB_MOUNT="/mnt/usb"
USB_LABEL="VIM3-Projects"
PROJECTS_PATH="$USB_MOUNT/projects"

show_menu() {
    clear
    echo -e "${BLUE}╔════════════════════════════════════╗${NC}"
    echo -e "${GREEN}${BOLD}   VIM3 PROJECTS USB MANAGER${NC}"
    echo -e "${BLUE}╚════════════════════════════════════╝${NC}"
    echo -e "${YELLOW}USB Label:${NC} $USB_LABEL"
    echo -e "${YELLOW}Mount point:${NC} $USB_MOUNT"
    echo ""
    
    if mount | grep -q "$USB_MOUNT"; then
        echo -e "${GREEN}✓ USB is MOUNTED${NC}"
        USED=$(df -h $USB_MOUNT | tail -1 | awk '{print $3}')
        TOTAL=$(df -h $USB_MOUNT | tail -1 | awk '{print $2}')
        echo -e "  Used: $USED / $TOTAL"
    else
        echo -e "${RED}✗ USB is NOT mounted${NC}"
    fi
    
    echo ""
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}1.${NC} Mount USB drive"
    echo -e "${YELLOW}2.${NC} Unmount USB drive (safe eject)"
    echo -e "${YELLOW}3.${NC} Show USB info & projects"
    echo -e "${YELLOW}4.${NC} List all projects on USB"
    echo -e "${YELLOW}5.${NC} Open shell in projects directory"
    echo -e "${YELLOW}6.${NC} Exit"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -n "Choose an option [1-6]: "
}

mount_usb() {
    if mount | grep -q "$USB_MOUNT"; then
        echo -e "${YELLOW}USB already mounted at $USB_MOUNT${NC}"
    else
        echo -e "${YELLOW}Mounting USB...${NC}"
        sudo mount /dev/sda1 $USB_MOUNT
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ USB mounted at $USB_MOUNT${NC}"
        else
            echo -e "${RED}✗ Failed to mount USB${NC}"
        fi
    fi
    read -p "Press Enter to continue..."
}

unmount_usb() {
    if mount | grep -q "$USB_MOUNT"; then
        echo -e "${YELLOW}Unmounting USB...${NC}"
        sudo umount $USB_MOUNT
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}✓ USB unmounted safely. You can now remove it.${NC}"
        else
            echo -e "${RED}✗ Failed to unmount. Is something using it?${NC}"
        fi
    else
        echo -e "${YELLOW}USB is not mounted${NC}"
    fi
    read -p "Press Enter to continue..."
}

show_info() {
    echo -e "${BLUE}--- USB Device Info ---${NC}"
    sudo blkid /dev/sda1
    echo ""
    echo -e "${BLUE}--- Mount Info ---${NC}"
    df -h | grep "$USB_MOUNT"
    echo ""
    echo -e "${BLUE}--- Projects on USB ---${NC}"
    if [ -d "$PROJECTS_PATH" ]; then
        ls -la "$PROJECTS_PATH"
    else
        echo -e "${YELLOW}No projects directory yet${NC}"
    fi
    read -p "Press Enter to continue..."
}

list_projects() {
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${GREEN}Projects on $USB_LABEL${NC}"
    echo -e "${BLUE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    
    if [ ! -d "$PROJECTS_PATH" ]; then
        echo -e "${YELLOW}No projects directory. Creating...${NC}"
        mkdir -p "$PROJECTS_PATH"
    fi
    
    cd "$PROJECTS_PATH"
    
    if [ -z "$(ls -A)" ]; then
        echo -e "${YELLOW}No projects yet. Add your projects here:${NC}"
        echo -e "  $PROJECTS_PATH"
    else
        for project in */; do
            if [ -d "$project" ]; then
                echo -e "  📁 ${GREEN}${project%/}${NC}"
            fi
        done
    fi
    echo ""
    read -p "Press Enter to continue..."
}

open_shell() {
    if [ ! -d "$PROJECTS_PATH" ]; then
        mkdir -p "$PROJECTS_PATH"
    fi
    echo -e "${GREEN}Opening shell in $PROJECTS_PATH${NC}"
    echo -e "${YELLOW}Type 'exit' to return${NC}"
    cd "$PROJECTS_PATH"
    exec $SHELL
}

while true; do
    show_menu
    read choice
    case $choice in
        1) mount_usb ;;
        2) unmount_usb ;;
        3) show_info ;;
        4) list_projects ;;
        5) open_shell ;;
        6) echo -e "${GREEN}Goodbye!${NC}"; exit 0 ;;
        *) echo -e "${RED}Invalid option${NC}"; read -p "Press Enter..." ;;
    esac
done
