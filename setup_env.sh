# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo "Environment setup complete!"
echo "Python path: $PYTHONPATH"
