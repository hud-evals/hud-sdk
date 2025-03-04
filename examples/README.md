## Claude Computer Use evaluation on OSWorld

### 1. Setup

Step 1: Install from the source repository:

```bash
# Clone the repository
git clone https://github.com/Human-Data/hud-sdk.git
cd hud-sdk
```

Step 2: Create a virtual environment:
```bash
# Option 1: using venv
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Option 2: using uv (recommended)
uv venv
# Then activate according to your shell
```

Step 3: Install in development mode with all dependencies:
```bash
# Option 1: using pip
pip install -e ".[dev]"

# Option 2: using uv (recommended)
uv pip install -e ".[dev]"
```

### 2. Set up environment variables

```bash
HUD_API_KEY=<your-api-key>
ANTHROPIC_API_KEY=<your-api-key>
```

### 3. Run the OSWorld example

Explore the [claude_osworld.ipynb](https://github.com/Human-Data/hud-sdk/blob/main/examples/claude_osworld.ipynb) notebook from this folder in Jupyter Notebook.


