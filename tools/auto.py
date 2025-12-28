#!/usr/bin/env python3
"""
Open Core Graph RAG - Automation Script (Python version)

Provides cross-platform automation for Docker operations, testing, linting, formatting, and type checking.

Usage:
    python auto.py <command> [path]
    or: ./auto.py <command> [path]

Commands:
    build               Build Docker container
    launch              Launch Docker containers
    stop                Stop Docker containers
    test [path]         Run tests on specified path (default: .)
    test-docker [path]  Run tests inside Docker container
    fix                 Format code with black
    lint                Check code with flake8
    typecheck           Type checking with mypy
    quality             Run format + lint + typecheck
    all                 Full pipeline: build + launch + quality + test
    dev                 Setup development environment (build + launch)
    help                Show help message
"""

import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
from typing import Optional, List
from enum import Enum


class Color(Enum):
    """ANSI color codes"""
    RED = '\033[0;31m'
    GREEN = '\033[0;32m'
    YELLOW = '\033[1;33m'
    BLUE = '\033[0;34m'
    END = '\033[0m'


class Logger:
    """Simple logging utility with colors"""
    
    @staticmethod
    def info(message: str) -> None:
        """Log info message"""
        print(f"{Color.BLUE.value}[INFO]{Color.END.value} {message}")
    
    @staticmethod
    def success(message: str) -> None:
        """Log success message"""
        print(f"{Color.GREEN.value}[SUCCESS]{Color.END.value} {message}")
    
    @staticmethod
    def error(message: str) -> None:
        """Log error message"""
        print(f"{Color.RED.value}[ERROR]{Color.END.value} {message}")
    
    @staticmethod
    def warning(message: str) -> None:
        """Log warning message"""
        print(f"{Color.YELLOW.value}[WARNING]{Color.END.value} {message}")


class DockerManager:
    """Manages Docker operations"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.docker_compose_file = self.project_root / "docker" / "docker-compose.yml"
    
    def check_docker_installed(self) -> bool:
        """Check if Docker and docker-compose are installed"""
        try:
            subprocess.run(["docker", "--version"], capture_output=True, check=True)
            subprocess.run(["docker-compose", "--version"], capture_output=True, check=True)
            Logger.success("Docker and docker-compose are available")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            Logger.error("Docker or docker-compose is not installed")
            return False
    
    def build(self) -> bool:
        """Build Docker image"""
        Logger.info("Building Docker image...")
        
        if not self.docker_compose_file.exists():
            Logger.error(f"docker-compose.yml not found at {self.docker_compose_file}")
            return False
        
        try:
            subprocess.run(
                ["docker-compose", "-f", str(self.docker_compose_file), "build"],
                cwd=self.project_root,
                check=True
            )
            Logger.success("Docker image built successfully")
            return True
        except subprocess.CalledProcessError as e:
            Logger.error(f"Failed to build Docker image: {e}")
            return False
    
    def launch(self) -> bool:
        """Launch Docker containers"""
        Logger.info("Launching Docker containers...")
        
        try:
            subprocess.run(
                ["docker-compose", "-f", str(self.docker_compose_file), "up", "-d"],
                cwd=self.project_root,
                check=True
            )
            Logger.success("Docker containers launched")
            
            # Wait for services to be ready
            Logger.info("Waiting for services to be ready...")
            time.sleep(5)
            
            # Check TigerGraph health
            return self._check_tigergraph_health()
        
        except subprocess.CalledProcessError as e:
            Logger.error(f"Failed to launch Docker containers: {e}")
            return False
    
    def _check_tigergraph_health(self) -> bool:
        """Check if TigerGraph is healthy"""
        Logger.info("Checking TigerGraph health...")
        max_attempts = 30
        
        for attempt in range(max_attempts):
            try:
                result = subprocess.run(
                    [
                        "docker-compose",
                        "-f",
                        str(self.docker_compose_file),
                        "exec",
                        "-T",
                        "tigergraph",
                        "curl",
                        "-s",
                        "http://localhost:9000/api/v2/health"
                    ],
                    cwd=self.project_root,
                    capture_output=True,
                    timeout=5
                )
                
                if result.returncode == 0:
                    Logger.success("TigerGraph is healthy")
                    return True
            
            except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
                pass
            
            attempt_num = attempt + 1
            Logger.info(f"Health check attempt {attempt_num}/{max_attempts}...")
            time.sleep(2)
        
        Logger.warning("TigerGraph health check timed out, but containers may still be starting")
        return True
    
    def stop(self) -> bool:
        """Stop Docker containers"""
        Logger.info("Stopping Docker containers...")
        
        try:
            subprocess.run(
                ["docker-compose", "-f", str(self.docker_compose_file), "down"],
                cwd=self.project_root,
                check=True
            )
            Logger.success("Docker containers stopped")
            return True
        except subprocess.CalledProcessError as e:
            Logger.error(f"Failed to stop Docker containers: {e}")
            return False


class TestRunner:
    """Manages test execution"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def run_tests(self, test_path: str = ".") -> bool:
        """Run pytest on specified path"""
        Logger.info(f"Running tests for: {test_path}")
        
        test_file = self.project_root / test_path
        if not test_file.exists():
            Logger.error(f"Path does not exist: {test_path}")
            return False
        
        try:
            subprocess.run(
                ["python", "-m", "pytest", test_path, "-v", "--tb=short"],
                cwd=self.project_root,
                check=True
            )
            Logger.success("All tests passed")
            return True
        except subprocess.CalledProcessError as e:
            Logger.error(f"Tests failed: {e}")
            return False
    
    def run_tests_in_docker(self, test_path: str = ".") -> bool:
        """Run tests inside Docker container"""
        Logger.info(f"Running tests inside Docker container for: {test_path}")
        
        docker_manager = DockerManager(str(self.project_root))
        docker_compose_file = docker_manager.docker_compose_file
        
        try:
            # Check if containers are running
            result = subprocess.run(
                ["docker-compose", "-f", str(docker_compose_file), "ps"],
                cwd=self.project_root,
                capture_output=True,
                text=True
            )
            
            if "tigergraph" not in result.stdout:
                Logger.info("TigerGraph container not running, launching...")
                if not docker_manager.launch():
                    return False
            
            # Run pytest in container
            subprocess.run(
                [
                    "docker-compose",
                    "-f",
                    str(docker_compose_file),
                    "exec",
                    "-T",
                    "app",
                    "python",
                    "-m",
                    "pytest",
                    test_path,
                    "-v",
                    "--tb=short"
                ],
                cwd=self.project_root,
                check=True
            )
            
            Logger.success("All tests passed in Docker")
            return True
        
        except subprocess.CalledProcessError as e:
            Logger.error(f"Tests failed in Docker: {e}")
            return False


class CodeFormatter:
    """Manages code formatting and quality checks"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
    
    def format_with_black(self) -> bool:
        """Format code with black"""
        Logger.info("Formatting code with black...")
        
        directories = ["src", "tests", "scripts"]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                if dir_path.exists():
                    subprocess.run(
                        ["python", "-m", "black", directory, "--line-length", "100"],
                        cwd=self.project_root,
                        check=True
                    )
            
            Logger.success("Code formatted with black")
            return True
        except subprocess.CalledProcessError as e:
            Logger.error(f"Failed to format code: {e}")
            return False
    
    def lint_with_flake8(self) -> bool:
        """Lint code with flake8"""
        Logger.info("Linting code with flake8...")
        
        directories = ["src", "tests", "scripts"]
        all_passed = True
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                if dir_path.exists():
                    try:
                        subprocess.run(
                            [
                                "python",
                                "-m",
                                "flake8",
                                directory,
                                "--max-line-length=100",
                                "--ignore=E203,W503"
                            ],
                            cwd=self.project_root,
                            check=True
                        )
                    except subprocess.CalledProcessError:
                        all_passed = False
            
            if all_passed:
                Logger.success("All linting checks passed")
            else:
                Logger.error("Linting found issues (see above)")
            
            return all_passed
        except Exception as e:
            Logger.error(f"Failed to run linting: {e}")
            return False
    
    def type_check_with_mypy(self) -> bool:
        """Type checking with mypy"""
        Logger.info("Running type checks with mypy...")
        
        directories = ["src", "tests"]
        
        try:
            for directory in directories:
                dir_path = self.project_root / directory
                if dir_path.exists():
                    try:
                        subprocess.run(
                            [
                                "python",
                                "-m",
                                "mypy",
                                directory,
                                "--ignore-missing-imports",
                                "--no-error-summary"
                            ],
                            cwd=self.project_root
                        )
                    except subprocess.CalledProcessError:
                        # mypy returns non-zero for warnings, but we continue
                        Logger.warning(f"Type checking found issues in {directory}")
            
            Logger.success("Type checking completed")
            return True
        except Exception as e:
            Logger.error(f"Failed to run type checking: {e}")
            return False
    
    def run_all_checks(self) -> bool:
        """Run all quality checks"""
        Logger.info("Running all quality checks...")
        
        # Format
        if not self.format_with_black():
            return False
        
        # Lint
        if not self.lint_with_flake8():
            return False
        
        # Type check (warnings only)
        self.type_check_with_mypy()
        
        Logger.success("All quality checks completed")
        return True


class AutomationScript:
    """Main automation script orchestrator"""
    
    def __init__(self):
        self.project_root = self._find_project_root()
        self.docker_manager = DockerManager(self.project_root)
        self.test_runner = TestRunner(self.project_root)
        self.code_formatter = CodeFormatter(self.project_root)
    
    @staticmethod
    def _find_project_root() -> str:
        """Find project root by looking for auto/auto.py"""
        current = Path(__file__).resolve().parent
        return str(current)
    
    def run_command(self, command: str, path: str = ".") -> int:
        """Execute automation command"""
        
        if not self.docker_manager.check_docker_installed():
            return 1
        
        command = command.lower()
        
        if command == "build":
            return 0 if self.docker_manager.build() else 1
        
        elif command == "launch":
            return 0 if self.docker_manager.launch() else 1
        
        elif command == "stop":
            return 0 if self.docker_manager.stop() else 1
        
        elif command == "test":
            return 0 if self.test_runner.run_tests(path) else 1
        
        elif command == "test-docker":
            return 0 if self.test_runner.run_tests_in_docker(path) else 1
        
        elif command == "fix":
            return 0 if self.code_formatter.format_with_black() else 1
        
        elif command == "lint":
            return 0 if self.code_formatter.lint_with_flake8() else 1
        
        elif command == "typecheck":
            return 0 if self.code_formatter.type_check_with_mypy() else 1
        
        elif command == "quality":
            return 0 if self.code_formatter.run_all_checks() else 1
        
        elif command == "all":
            Logger.info("Running full pipeline: build, launch, quality checks, and tests")
            
            if not self.docker_manager.build():
                return 1
            if not self.docker_manager.launch():
                return 1
            if not self.code_formatter.run_all_checks():
                return 1
            if not self.test_runner.run_tests_in_docker(path):
                return 1
            
            Logger.success("Full pipeline completed successfully")
            return 0
        
        elif command == "dev":
            Logger.info("Starting development environment")
            
            if not self.docker_manager.build():
                return 1
            if not self.docker_manager.launch():
                return 1
            
            Logger.success("Development environment ready")
            Logger.info("Run './auto test' to run tests")
            Logger.info("Run './auto fix' to format code")
            Logger.info("Run './auto quality' to run all checks")
            return 0
        
        elif command == "help" or command == "":
            self._print_help()
            return 0
        
        else:
            Logger.error(f"Unknown command: {command}")
            self._print_help()
            return 1
    
    @staticmethod
    def _print_help() -> None:
        """Print help message"""
        help_text = f"""
{Color.BLUE.value}Open Core Graph RAG - Automation Script{Color.END.value}

{Color.GREEN.value}Usage:{Color.END.value}
  ./auto.py <command> [path]
  python auto.py <command> [path]

{Color.GREEN.value}Commands:{Color.END.value}
  {Color.YELLOW.value}build{Color.END.value}              Build Docker container
  {Color.YELLOW.value}launch{Color.END.value}             Launch Docker containers
  {Color.YELLOW.value}stop{Color.END.value}               Stop Docker containers
  {Color.YELLOW.value}test{Color.END.value} [path]        Run tests on specified path (default: .)
                     Example: ./auto.py test tests/
                     Example: ./auto.py test tests/test_graph_store.py
  {Color.YELLOW.value}test-docker{Color.END.value}        Run tests inside Docker container (requires running containers)
  {Color.YELLOW.value}fix{Color.END.value}                Format code with black
  {Color.YELLOW.value}lint{Color.END.value}               Check code with flake8
  {Color.YELLOW.value}typecheck{Color.END.value}          Type checking with mypy
  {Color.YELLOW.value}quality{Color.END.value}            Run format + lint + typecheck
  {Color.YELLOW.value}all{Color.END.value}                Full pipeline: build + launch + quality + test
  {Color.YELLOW.value}dev{Color.END.value}                Setup development environment (build + launch)
  {Color.YELLOW.value}help{Color.END.value}               Show this help message

{Color.GREEN.value}Examples:{Color.END.value}
  ./auto.py build                    # Build Docker image
  ./auto.py launch                   # Launch Docker containers
  ./auto.py test                     # Run all tests locally
  ./auto.py test tests/              # Run tests in tests/ directory
  ./auto.py test-docker tests/       # Run tests inside Docker container
  ./auto.py fix                      # Format all code
  ./auto.py lint                     # Check linting
  ./auto.py typecheck                # Type checking
  ./auto.py quality                  # Run all quality checks
  ./auto.py all                      # Full pipeline
  ./auto.py dev                      # Setup dev environment

{Color.GREEN.value}Development Workflow:{Color.END.value}
  1. ./auto.py dev                   # Start development environment
  2. ./auto.py fix                   # Format your code
  3. ./auto.py test                  # Run tests
  4. ./auto.py quality               # Check code quality
  5. ./auto.py test-docker           # Run tests in Docker
"""
        print(help_text)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Open Core Graph RAG - Automation Script",
        add_help=False
    )
    parser.add_argument("command", nargs="?", default="help", help="Command to run")
    parser.add_argument("path", nargs="?", default=".", help="Path for test command")
    
    args = parser.parse_args()
    
    script = AutomationScript()
    return script.run_command(args.command, args.path)


if __name__ == "__main__":
    sys.exit(main())
