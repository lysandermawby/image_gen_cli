#!/bin/sh
# Install imggen CLI

: << EOF
Sets up a CLI for local image generation
Automatically installs to a ~/.image-gen/ directory
Adds a symlink to the bin directory, and adds to shell path. 
Warning: Will override existing sim link with the CLI name, if it already exists
EOF

# exit on any error
set -e 

# installation directories
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
INSTALL_DIR="$HOME/.image-gen"
BIN_DIR="$HOME/.local/bin"
CLI_NAME="imggen"
# CLI_PATH="$BIN_DIR/$CLI_NAME"

echo "Installing $CLI_NAME CLI"
echo "Project directory linked to: $PROJECT_DIR"


# creating and copying to an installation directory
mkdir -p "$INSTALL_DIR"
echo "Copying files to $INSTALL_DIR"
cp -r "$PROJECT_DIR"/* "$INSTALL_DIR/"

# create wrapper script, passing all command line arguments back to main.py
cat > "$INSTALL_DIR/$CLI_NAME" << EOF
#!/bin/sh
# Generated wrapper for imggen
exec uv run --directory "$INSTALL_DIR" main.py "\$@"
EOF
chmod +x "$INSTALL_DIR/$CLI_NAME"

echo "Installed $CLI_NAME to $INSTALL_DIR/$CLI_NAME"

# creating bin directory if it doesn't exist
echo "Creating symlink in $BIN_DIR"
mkdir -p "$BIN_DIR"
ln -sf "$INSTALL_DIR/$CLI_NAME" "$BIN_DIR/$CLI_NAME"

# check if BIN_DIR is in path, warn if not found
if ! echo ":$PATH:" | grep -q ":$BIN_DIR:"; then
    echo ""
    echo "Add this to ~/.zshrc:"
    echo '    export PATH="$HOME/.local/bin:$PATH"'
    echo ""
    echo "Then run: source ~/.zshrc"
else
    echo "$BIN_DIR is already in PATH"
    echo "Installation complete! Run: $CLI_NAME \"your prompt\""
fi
