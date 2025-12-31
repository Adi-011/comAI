import subprocess

print(
    subprocess.run(
        ["ollama", "run", "gemma:2b"],
        input="Summarize a comic about a hero saving a city.",
        text=True,
        capture_output=True
    ).stdout
)
