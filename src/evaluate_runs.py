import sys

from _bootstrap import PROJECT_ROOT, maybe_reexec_in_venv

if __name__ == "__main__":
    maybe_reexec_in_venv(__file__)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from evaluate_runs import main


if __name__ == "__main__":
    main()
