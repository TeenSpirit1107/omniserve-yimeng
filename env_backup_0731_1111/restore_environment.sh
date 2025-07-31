#!/bin/bash
echo "Restoring environment..."
conda env create -f environment_backup.yml
echo "Note: flash-attn may need to be recompiled"
echo "Check flash_attn_backup.txt for details"
