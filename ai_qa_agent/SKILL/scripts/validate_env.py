"""Validate RAG QA System environment setup.

Checks: Python version, uv, dependencies, .env file, data/ directory.
Run from project root:
    python ai_qa_agent/SKILL/rag-qa-system/scripts/validate_env.py
"""
import os
import sys


def check_python() -> bool:
    major, minor = sys.version_info[:2]
    if (major, minor) >= (3, 9):
        print(f"  Python {major}.{minor} OK")
        return True
    print(f"  Python {major}.{minor} < 3.9 FAIL")
    return False


def check_env_file(project_root: str) -> bool:
    env_path = os.path.join(project_root, ".env")
    if os.path.exists(env_path):
        with open(env_path, encoding="utf-8") as f:
            content = f.read()
        if "OPENAI_API_KEY=sk-" in content and "sk-your" not in content:
            print("  .env with API key OK")
            return True
        else:
            print("  .env found but API key not set — replace sk-your-key-here")
            return False
    print("  .env not found — create at project root with OPENAI_API_KEY=sk-...")
    return False


def check_data_dir(agent_dir: str) -> bool:
    data_dir = os.path.join(agent_dir, "data")
    if os.path.isdir(data_dir):
        files = [f for f in os.listdir(data_dir) if not f.startswith(".")]
        if files:
            print(f"  data/ directory OK ({len(files)} files)")
            return True
        print("  data/ directory empty — add .md/.pdf/.docx files")
        return False
    print("  data/ directory not found")
    return False


def main() -> int:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # script_dir = SKILL/rag-qa-system/scripts
    # agent_dir = ai_qa_agent (4 levels up: scripts -> rag-qa-system -> SKILL -> ai_qa_agent)
    agent_dir = os.path.dirname(os.path.dirname(os.path.dirname(script_dir)))
    # project_root = parent of ai_qa_agent
    project_root = os.path.dirname(agent_dir)

    print("RAG QA System — Environment Validation")
    print("=" * 50)

    all_ok = True

    print("\n[1/3] Python version:")
    all_ok &= check_python()

    print("\n[2/3] Environment file:")
    all_ok &= check_env_file(project_root)

    print("\n[3/3] Knowledge base:")
    all_ok &= check_data_dir(agent_dir)

    print("\n" + "=" * 50)
    if all_ok:
        print("All checks passed.")
        return 0
    else:
        print("Some checks failed. Fix issues above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
