# Automation Tools

Quick automation scripts and configurations for Docker, testing, and code quality.

## 🚀 30-Second Setup

```bash
./auto dev              # Build Docker + launch containers (pick ONE tool)
python auto.py dev     # or use Python version (all platforms)
make dev               # or use Makefile
```

## 📋 Available Commands

### Docker
```bash
./auto build            # Build Docker image
./auto launch           # Start containers
./auto stop             # Stop containers
```

### Testing
```bash
./auto test tests/      # Run tests locally
./auto test-docker      # Run tests in Docker
```

### Code Quality
```bash
./auto fix              # Format with black (100 chars)
./auto lint             # Check with flake8
./auto typecheck        # Type check with mypy
./auto quality          # All checks: fix + lint + typecheck
```

### Combined
```bash
./auto dev              # Build + launch
./auto all              # Full pipeline: build + launch + quality + test
./auto help             # Show all commands
```

## 🛠️ Three Tools (Pick One)

All three do the same thing. Choose based on preference:

| Tool | Platform | Usage |
|------|----------|-------|
| **`auto`** | macOS/Linux | `./auto <command>` |
| **`auto.py`** ⭐ | macOS/Linux/Windows | `python auto.py <command>` |
| **`Makefile`** | macOS/Linux | `make <command>` |

From project root, symlinks are provided: `./auto`, `./auto.py`, `./Makefile`

## ⚙️ Configuration

- **`pyproject.toml`** - Black (line-length: 100), mypy, pytest, coverage
- **`.flake8`** - Flake8 rules (max-line: 100, ignores: E203, W503, W504, E501)

## 💻 Typical Workflow

```bash
# Morning: setup (once)
./auto dev

# During development:
./auto fix              # Format code
./auto test tests/      # Quick tests
./auto quality          # Full checks

# Before commit:
./auto test-docker      # Docker tests
git add -A && git commit -m "..."

# End of day:
./auto stop
```

## ✨ Features

✅ Docker management with health checks  
✅ Local & Docker testing  
✅ Code formatting, linting, type checking  
✅ Color output & error handling  
✅ Cross-platform (Python version)

## ❓ FAQ

**Which tool should I use?**  
Use `auto.py` for cross-platform (Windows), `auto` for speed, `make` if familiar.

**Can I run from root?**  
Yes! Symlinks: `./auto`, `./auto.py`, `./Makefile` work from project root.

**How to customize?**  
Edit scripts, `pyproject.toml`, or `.flake8`. All three scripts are well-commented.

**Commands not working?**  
Ensure Docker is installed. Make executable: `chmod +x auto auto.py`

---

**Status:** ✅ Ready to use. Run `./auto help` for full command reference.
