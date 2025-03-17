import os
import git
from pip._internal.cli.main import main as pip_main
from .utils import is_colab
import subprocess

class Cloner:
    @staticmethod
    def clone_and_install(repo_url):
        original_dir = os.getcwd()
        installation_path = None
        success = False
        
        target_dir = os.path.join(os.path.expanduser("~"), '.str2speech')
            
        try:
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            os.chdir(target_dir)
            
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            print(f"Cloning repository from {repo_url} into {target_dir}...")
            
            git.Repo.clone_from(repo_url, repo_name)
            
            if os.path.exists(repo_name):
                os.chdir(repo_name)
                print("Installing repository...")
                
                install_result = pip_main(['install', '-e', '.'])
                
                if install_result == 0:
                    installation_path = os.path.abspath('.')
                    success = True
                    print("Successfully cloned and installed the repository!")

                    if is_colab():
                        try:
                            result = subprocess.run(
                                ['sudo', 'apt', 'install', 'espeak-ng', '-y'],
                                check=True,
                                capture_output=True,
                                text=True
                            )
                            print(result.stdout)
                        except subprocess.CalledProcessError as e:
                            pass                                                
                else:
                    print(f"pip install failed with code {install_result}")
            else:
                print(f"Repository directory {repo_name} was not created after git clone")
                
        except git.GitCommandError as e:
            print(f"Git error: {e}")
        except Exception as e:
            print(f"Unexpected error: {str(e)}")
        finally:
            os.chdir(original_dir)
            
        return {
            "success": success,
            "installation_path": installation_path,
            "repo_name": repo_name if success else None
        }