"""Run interactively; never pass a token as a command argument."""
from huggingface_hub import login

if __name__ == "__main__":
    login(add_to_git_credential=False)
