import argparse
import shutil
import subprocess
import sys

from config import CORPUS_PATH, INDEX_DIR, INDEX_INPUT_DIR
from utils import configure_java_environment, ensure_exists


def check_java():
    java_home = configure_java_environment()
    java_path = shutil.which("java")
    if java_path is None:
        raise EnvironmentError(
            "Java was not found in PATH. Install a JDK and verify with `java -version`."
        )

    result = subprocess.run(
        ["java", "-version"],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise EnvironmentError(
            "Java is installed but `java -version` failed.\n"
            f"Output:\n{result.stdout}"
        )

    java_description = result.stdout.strip()
    if java_home is not None:
        java_description += f"\nJAVA_HOME={java_home}"
    return java_description


def build_index(threads: int = 2, overwrite: bool = False):
    ensure_exists(CORPUS_PATH, "prepared FiQA corpus")
    java_version = check_java()

    INDEX_INPUT_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(CORPUS_PATH, INDEX_INPUT_DIR / CORPUS_PATH.name)

    if INDEX_DIR.exists() and any(INDEX_DIR.iterdir()):
        if not overwrite:
            print(f"Index already exists at {INDEX_DIR}. Use --overwrite to rebuild.")
            return
        shutil.rmtree(INDEX_DIR)

    cmd = [
        sys.executable,
        "-m",
        "pyserini.index.lucene",
        "--collection",
        "JsonCollection",
        "--input",
        str(INDEX_INPUT_DIR),
        "--index",
        str(INDEX_DIR),
        "--generator",
        "DefaultLuceneDocumentGenerator",
        "--threads",
        str(threads),
        "--storePositions",
        "--storeDocvectors",
        "--storeRaw",
    ]

    print("Java check:")
    print(java_version)
    print("Running index build command:")
    print(" ".join(cmd))
    subprocess.run(cmd, check=True)
    print(f"Lucene index saved to {INDEX_DIR}")


def main():
    parser = argparse.ArgumentParser(description="Build the Lucene index for FiQA.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the existing index.",
    )
    args = parser.parse_args()

    build_index(threads=args.threads, overwrite=args.overwrite)


if __name__ == "__main__":
    main()
