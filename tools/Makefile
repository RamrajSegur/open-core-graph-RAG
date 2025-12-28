# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m

# Default target
.DEFAULT_GOAL := help

# Mark all targets as phony
.PHONY: help dev build launch stop test test-docker fix lint typecheck quality all clean
.PHONY: check fmt ci

help:
	@echo "$(BLUE)Open Core Graph RAG - Makefile$(NC)"
	@echo ""
	@echo "$(GREEN)Usage:$(NC)"
	@echo "  make <target>"
	@echo ""
	@echo "$(GREEN)Targets:$(NC)"
	@echo "  $(YELLOW)dev$(NC)         Setup development environment (build + launch)"
	@echo "  $(YELLOW)build$(NC)       Build Docker container"
	@echo "  $(YELLOW)launch$(NC)      Launch Docker containers"
	@echo "  $(YELLOW)stop$(NC)        Stop Docker containers"
	@echo "  $(YELLOW)test$(NC)        Run tests locally (default: all tests)"
	@echo "  $(YELLOW)test-docker$(NC) Run tests inside Docker container"
	@echo "  $(YELLOW)fix$(NC)         Format code with black"
	@echo "  $(YELLOW)lint$(NC)        Check code with flake8"
	@echo "  $(YELLOW)typecheck$(NC)   Type checking with mypy"
	@echo "  $(YELLOW)quality$(NC)     Run all quality checks (format + lint + typecheck)"
	@echo "  $(YELLOW)all$(NC)         Full pipeline (build + launch + quality + test)"
	@echo "  $(YELLOW)check$(NC)       Quick check: quality + test"
	@echo "  $(YELLOW)ci$(NC)          CI checks: quality + test-docker"
	@echo "  $(YELLOW)clean$(NC)       Stop Docker containers"
	@echo ""
	@echo "$(GREEN)Examples:$(NC)"
	@echo "  make dev              # Setup environment"
	@echo "  make fix              # Format code"
	@echo "  make test             # Run tests"
	@echo "  make quality          # Check quality"
	@echo "  make all              # Full pipeline"

dev: build launch
	@echo "$(GREEN)[SUCCESS]$(NC) Development environment ready"
	@echo "Run 'make test' to run tests"
	@echo "Run 'make quality' to check code quality"

build:
	@./auto build

launch:
	@./auto launch

stop:
	@./auto stop

test:
	@./auto test tests/

test-docker:
	@./auto test-docker tests/

fix:
	@./auto fix

lint:
	@./auto lint

typecheck:
	@./auto typecheck

quality: fix lint typecheck
	@echo "$(GREEN)[SUCCESS]$(NC) All quality checks passed"

all: build launch quality test
	@echo "$(GREEN)[SUCCESS]$(NC) Full pipeline completed"

clean: stop
	@echo "$(GREEN)[SUCCESS]$(NC) Cleaned up Docker resources"

check: quality test
	@echo "$(GREEN)[SUCCESS]$(NC) Checks passed"

fmt: fix
	@echo "$(GREEN)[SUCCESS]$(NC) Code formatted"

ci: quality test-docker
	@echo "$(GREEN)[SUCCESS]$(NC) CI checks passed"
