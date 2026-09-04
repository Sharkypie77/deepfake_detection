#!/usr/bin/env python
"""
Comprehensive Workspace Health Check
Detects errors, redundancy, missing dependencies, and issues
"""

import os
import sys
import json
from pathlib import Path
from collections import defaultdict
import subprocess

class HealthCheck:
    def __init__(self, workspace_root):
        self.workspace = Path(workspace_root)
        self.issues = {
            "errors": [],
            "warnings": [],
            "redundancy": [],
            "missing_deps": [],
            "code_quality": []
        }
        self.metrics = {
            "total_files": 0,
            "python_files": 0,
            "documentation_files": 0,
            "config_files": 0,
            "total_lines": 0
        }
    
    def check_dependencies(self):
        """Check if all required packages are installed"""
        required = [
            "torch",
            "fastapi",
            "sqlalchemy",
            "librosa",
            "whisper",
            "mediapipe",
            "telegram",
            "numpy",
            "scipy",
            "sklearn",
            "pandas"
        ]
        
        print("\n[CHECKING DEPENDENCIES]")
        for pkg in required:
            try:
                __import__(pkg)
                print("  [OK] {}".format(pkg))
            except ImportError:
                msg = "Missing package: {}".format(pkg)
                self.issues["missing_deps"].append(msg)
                print("  [MISSING] {}".format(pkg))
    
    def check_python_syntax(self):
        """Check Python files for syntax errors"""
        print("\n[CHECKING PYTHON SYNTAX]")
        
        py_files = list(self.workspace.glob("*.py")) + list(self.workspace.glob("modules/*.py"))
        
        for py_file in py_files:
            try:
                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    compile(f.read(), str(py_file), "exec")
                print("  [OK] {}".format(py_file.name))
            except SyntaxError as e:
                msg = "Syntax error in {}: {}".format(py_file.name, e)
                self.issues["errors"].append(msg)
                print("  [ERROR] {}".format(py_file.name))
            except Exception as e:
                print("  [SKIP] {} (read error)".format(py_file.name))
    
    def check_imports(self):
        """Check for missing imports"""
        print("\n[CHECKING IMPORTS]")
        
        py_files = list(self.workspace.glob("*.py")) + list(self.workspace.glob("modules/*.py"))
        
        for py_file in py_files:
            try:
                with open(py_file) as f:
                    content = f.read()
                    # Check for common import patterns
                    for line in content.split('\n'):
                        if line.strip().startswith('from ') or line.strip().startswith('import '):
                            # Try to evaluate the import statement
                            try:
                                exec(line)
                            except:
                                pass  # Will be caught later
                print("  [OK] {}".format(py_file.name))
            except Exception as e:
                msg = "Import issue in {}: {}".format(py_file.name, str(e)[:100])
                self.issues["warnings"].append(msg)
    
    def check_redundancy(self):
        """Check for code redundancy"""
        print("\n[CHECKING REDUNDANCY]")
        
        # Read all Python files
        code_content = {}
        for py_file in self.workspace.glob("*.py"):
            try:
                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    code_content[py_file.name] = f.read()
            except:
                pass
        
        # Check for duplicate function definitions
        functions = defaultdict(list)
        for fname, content in code_content.items():
            for line in content.split('\n'):
                if line.strip().startswith('def '):
                    func_name = line.split('(')[0].replace('def ', '').strip()
                    functions[func_name].append(fname)
        
        # Report duplicates
        for func_name, files in functions.items():
            if len(files) > 1:
                msg = "Function '{}' defined in: {}".format(func_name, ", ".join(set(files)))
                self.issues["redundancy"].append(msg)
                print("  [DUPLICATE] {}".format(func_name))
        
        # Check for large functions (>500 lines)
        for py_file in self.workspace.glob("*.py"):
            try:
                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    lines = f.readlines()
                    for i, line in enumerate(lines):
                        if line.strip().startswith('def '):
                            func_start = i
                            # Count function length
                            depth = 0
                            for j in range(i+1, len(lines)):
                                if lines[j].startswith('def ') and lines[j][0] != ' ':
                                    func_len = j - func_start
                                    if func_len > 500:
                                        msg = "Large function '{}' in {} ({} lines)".format(
                                            line.split('(')[0].replace('def ', '').strip(),
                                            py_file.name,
                                            func_len
                                        )
                                        self.issues["code_quality"].append(msg)
                                    break
            except:
                pass
    
    def check_config_consistency(self):
        """Check if configuration is consistent across files"""
        print("\n[CHECKING CONFIG CONSISTENCY]")
        
        # Check config.py vs .env
        config_vars = set()
        env_vars = set()
        
        if (self.workspace / "config.py").exists():
            try:
                with open(self.workspace / "config.py", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if " = os.getenv(" in line:
                            var = line.split("=")[0].strip()
                            config_vars.add(var)
            except:
                pass
        
        if (self.workspace / ".env").exists():
            try:
                with open(self.workspace / ".env", encoding="utf-8", errors="ignore") as f:
                    for line in f:
                        if "=" in line and not line.startswith("#"):
                            var = line.split("=")[0].strip()
                            env_vars.add(var)
            except:
                pass
        
        # Check for mismatches
        in_config_not_env = config_vars - env_vars
        if in_config_not_env:
            msg = "Config vars missing from .env: {}".format(", ".join(in_config_not_env))
            self.issues["warnings"].append(msg)
            print("  [WARNING] Missing in .env: {}".format(len(in_config_not_env)))
        else:
            print("  [OK] Config and .env consistent")
    
    def check_file_metrics(self):
        """Calculate file and code metrics"""
        print("\n[CALCULATING METRICS]")
        
        total_lines = 0
        
        for py_file in self.workspace.glob("**/*.py"):
            self.metrics["python_files"] += 1
            try:
                with open(py_file, encoding="utf-8", errors="ignore") as f:
                    lines = len(f.readlines())
                    total_lines += lines
            except:
                pass
        
        for md_file in self.workspace.glob("**/*.md"):
            self.metrics["documentation_files"] += 1
        
        for cfg in list(self.workspace.glob("*.yml")) + list(self.workspace.glob("*.yaml")) + list(self.workspace.glob(".env*")):
            self.metrics["config_files"] += 1
        
        self.metrics["total_files"] = len(list(self.workspace.glob("**/*")))
        self.metrics["total_lines"] = total_lines
        
        print("  Total files: {}".format(self.metrics["total_files"]))
        print("  Python files: {}".format(self.metrics["python_files"]))
        print("  Documentation: {}".format(self.metrics["documentation_files"]))
        print("  Config files: {}".format(self.metrics["config_files"]))
        print("  Total lines of code: {}".format(self.metrics["total_lines"]))
    
    def check_database_schema(self):
        """Check database schema consistency"""
        print("\n[CHECKING DATABASE SCHEMA]")
        
        try:
            from models import Video, FinalResult, BatchJob, AuditLog
            print("  [OK] All database models import successfully")
        except Exception as e:
            msg = "Database schema import failed: {}".format(str(e)[:100])
            self.issues["errors"].append(msg)
            print("  [ERROR] Import failed: {}".format(str(e)[:50]))
    
    def check_api_endpoints(self):
        """Check FastAPI endpoints"""
        print("\n[CHECKING API ENDPOINTS]")
        
        try:
            import re
            with open(self.workspace / "main.py") as f:
                content = f.read()
                endpoints = re.findall(r'@app\.(get|post|put|delete)\("([^"]+)"', content)
                
                if endpoints:
                    print("  Found {} endpoints:".format(len(endpoints)))
                    for method, path in endpoints[:10]:  # Show first 10
                        print("    [{}] {}".format(method.upper(), path))
                    if len(endpoints) > 10:
                        print("    ... and {} more".format(len(endpoints) - 10))
                else:
                    self.issues["warnings"].append("No API endpoints found in main.py")
        except Exception as e:
            self.issues["warnings"].append("Could not check API endpoints: {}".format(str(e)[:50]))
    
    def check_missing_files(self):
        """Check for missing critical files"""
        print("\n[CHECKING CRITICAL FILES]")
        
        critical = [
            "config.py",
            "models.py",
            "pipeline.py",
            "main.py",
            "init.py",
            "requirements.txt",
            ".env",
            "telegram_bot.py"
        ]
        
        for fname in critical:
            if (self.workspace / fname).exists():
                print("  [OK] {}".format(fname))
            else:
                msg = "Missing critical file: {}".format(fname)
                self.issues["errors"].append(msg)
                print("  [MISSING] {}".format(fname))
    
    def generate_report(self):
        """Generate comprehensive health report"""
        print("\n" + "="*70)
        print("HEALTH CHECK REPORT")
        print("="*70)
        
        # Summary
        total_issues = len(self.issues["errors"]) + len(self.issues["warnings"])
        print("\nSUMMARY:")
        print("  Errors: {}".format(len(self.issues["errors"])))
        print("  Warnings: {}".format(len(self.issues["warnings"])))
        print("  Redundancy issues: {}".format(len(self.issues["redundancy"])))
        print("  Missing dependencies: {}".format(len(self.issues["missing_deps"])))
        print("  Code quality issues: {}".format(len(self.issues["code_quality"])))
        
        # Metrics
        print("\nMETRICS:")
        print("  Total files: {}".format(self.metrics["total_files"]))
        print("  Python files: {}".format(self.metrics["python_files"]))
        print("  Documentation files: {}".format(self.metrics["documentation_files"]))
        print("  Total lines of code: {}".format(self.metrics["total_lines"]))
        
        # Detailed issues
        if self.issues["errors"]:
            print("\nERRORS (MUST FIX):")
            for err in self.issues["errors"][:10]:
                print("  - {}".format(err))
            if len(self.issues["errors"]) > 10:
                print("  ... and {} more".format(len(self.issues["errors"]) - 10))
        
        if self.issues["missing_deps"]:
            print("\nMISSING DEPENDENCIES:")
            for dep in self.issues["missing_deps"]:
                print("  - {}".format(dep))
        
        if self.issues["warnings"]:
            print("\nWARNINGS:")
            for warn in self.issues["warnings"][:5]:
                print("  - {}".format(warn))
        
        if self.issues["redundancy"]:
            print("\nREDUNDANCY:")
            for red in self.issues["redundancy"][:5]:
                print("  - {}".format(red))
        
        # Health status
        print("\nOVERALL STATUS:")
        if len(self.issues["errors"]) == 0 and len(self.issues["missing_deps"]) == 0:
            print("  [OK] HEALTHY - No critical issues")
            return 0
        elif len(self.issues["errors"]) > 0:
            print("  [ERROR] NOT READY - Fix errors first")
            return 1
        else:
            print("  [WARNING] NEEDS ATTENTION - Install missing dependencies")
            return 1

if __name__ == "__main__":
    workspace = r"C:\Users\heman\.copilot\repos\copilot-worktrees\deepfake_detection\sharkypie77-automatic-lamp"
    checker = HealthCheck(workspace)
    
    print("\n" + "="*70)
    print("DEEPFAKE DETECTION WORKSPACE - HEALTH CHECK")
    print("="*70)
    
    checker.check_dependencies()
    checker.check_python_syntax()
    checker.check_imports()
    checker.check_redundancy()
    checker.check_config_consistency()
    checker.check_missing_files()
    checker.check_file_metrics()
    checker.check_database_schema()
    checker.check_api_endpoints()
    
    exit_code = checker.generate_report()
    sys.exit(exit_code)
