#!/bin/bash

# RAG Pipeline Setup Script
# This script sets up the RAG Pipeline CLI for easy system-wide access

echo "🚀 RAG Pipeline Setup"
echo "===================="

# Check if running on macOS or Linux
if [[ "$OSTYPE" == "darwin"* ]]; then
    echo "✓ Detected macOS"
    BIN_DIR="/usr/local/bin"
elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "✓ Detected Linux"
    BIN_DIR="/usr/local/bin"
else
    echo "❌ Unsupported OS: $OSTYPE"
    exit 1
fi

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Create the executable wrapper
echo "📝 Creating executable wrapper..."
cat > "$SCRIPT_DIR/rag" << 'EOF'
#!/bin/bash
# RAG Pipeline CLI Wrapper

# Get the directory where this script is installed
INSTALL_DIR="$(dirname "$(readlink -f "$0")")"

# Set PYTHONPATH to include the installation directory
export PYTHONPATH="$INSTALL_DIR:$PYTHONPATH"

# Load environment variables from .env if it exists
if [ -f "$INSTALL_DIR/.env" ]; then
    # Use set -a to export all variables, then source the file
    set -a
    source "$INSTALL_DIR/.env"
    set +a
fi

# Set default data directory if not specified
if [ -z "$RAG_DATA_DIR" ]; then
    export RAG_DATA_DIR="$HOME/.rag_pipeline"
fi

# Execute the main script with all arguments
exec python3 "$INSTALL_DIR/main.py" "$@"
EOF

# Make the wrapper executable
chmod +x "$SCRIPT_DIR/rag"

# Check if user has write permission to /usr/local/bin
if [ -w "$BIN_DIR" ]; then
    echo "📦 Installing to $BIN_DIR..."
    ln -sf "$SCRIPT_DIR/rag" "$BIN_DIR/rag"
    echo "✅ Installation complete!"
else
    echo "⚠️  Need sudo permission to install to $BIN_DIR"
    echo "Running: sudo ln -sf $SCRIPT_DIR/rag $BIN_DIR/rag"
    sudo ln -sf "$SCRIPT_DIR/rag" "$BIN_DIR/rag"
    echo "✅ Installation complete!"
fi

# Create default data directory
DEFAULT_DATA_DIR="$HOME/.rag_pipeline"
if [ ! -d "$DEFAULT_DATA_DIR" ]; then
    echo "📁 Creating default data directory: $DEFAULT_DATA_DIR"
    mkdir -p "$DEFAULT_DATA_DIR"
fi

# Test the installation
echo ""
echo "🧪 Testing installation..."
if command -v rag &> /dev/null; then
    echo "✅ 'rag' command is available!"
    echo ""
    echo "📋 Available commands:"
    rag --help | grep -E "^\s+(ingest-|search|query|list|delete|status)" | head -10
else
    echo "❌ 'rag' command not found. Please add $BIN_DIR to your PATH"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "Usage examples:"
echo "  rag ingest-dir ./docs --name my-docs"
echo "  rag search \"how to use the API\""
echo "  rag query \"explain the authentication system\""
echo "  rag list"
echo ""
echo "Data directory: $DEFAULT_DATA_DIR"
echo "To use a different data directory, set RAG_DATA_DIR environment variable"