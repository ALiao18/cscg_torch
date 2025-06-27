#!/bin/bash
# CSCG Torch Environment Activation Script

echo "Activating CSCG Torch environment..."
source /Users/Andrew/Documents/Code/Research/LLM_WorldModel/CSCG_Maze/cscg_torch/cscg_env/bin/activate

echo "Environment activated!"
echo "Python: $(which python)"
echo "Pip: $(which pip)"

# Set useful aliases
alias cscg-test="python -m pytest tests/"
alias cscg-generate-rooms="cd rooms && python generate_rooms.py && cd .."

echo ""
echo "Available commands:"
echo "  cscg-test          - Run all tests"
echo "  cscg-generate-rooms - Generate test rooms"
echo "  deactivate         - Exit environment"
echo ""