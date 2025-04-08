import os
import git
from pip._internal.cli.main import main as pip_main
from .utils import get_str2speech_home


class Cloner:
    @staticmethod
    def clone_and_install(repo_url, dev_mode=True):
        original_dir = os.getcwd()
        installation_path = None
        success = False

        target_dir = get_str2speech_home()

        try:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)

            os.chdir(target_dir)

            repo_name = repo_url.split("/")[-1].replace(".git", "")
            print(f"Cloning repository from {repo_url} into {target_dir}...")

            if not os.path.exists(repo_name):
                git.Repo.clone_from(repo_url, repo_name)
            else:
                print("Already installed.")
                return

            if os.path.exists(repo_name):
                os.chdir(repo_name)
                print("Installing repository...")

                if dev_mode:
                    install_result = pip_main(["install", "-e", "."])
                else:
                    install_result = pip_main(["install", "."])

                if install_result == 0:
                    installation_path = os.path.abspath(".")
                    success = True
                    print("Successfully cloned and installed the repository!")
                else:
                    print(f"pip install failed with code {install_result}")
            else:
                print(
                    f"Repository directory {repo_name} was not created after git clone"
                )

        except git.GitCommandError as e:
            print(f"Git error: {e}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        finally:
            os.chdir(original_dir)

        return {
            "success": success,
            "installation_path": installation_path,
            "repo_name": repo_name if success else None,
        }
