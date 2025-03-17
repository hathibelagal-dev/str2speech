import os
import tempfile
import shutil
import git
from pip._internal.cli.main import main as pip_main

class Cloner:
    @staticmethod
    def clone_and_install(repo_url, keep_temp_dir=False):
        temp_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        installation_path = None
        success = False
        
        try:
            os.chdir(temp_dir)
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            print(f"Cloning repository from {repo_url} into temporary directory...")            
            repo = git.Repo.clone_from(repo_url, repo_name)
            
            if os.path.exists(repo_name):
                os.chdir(repo_name)
                print("Installing repository...")            
                install_result = pip_main(['install', '-e', '.'])
                if install_result == 0:
                    installation_path = os.path.abspath('.')
                    success = True
                    print("Successfully cloned and installed the repository!")
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
            if not keep_temp_dir and os.path.exists(temp_dir):
                try:
                    shutil.rmtree(temp_dir)
                    print("Temporary directory cleaned up.")
                except Exception as e:
                    print(f"Error cleaning up temporary directory: {str(e)}")
            
        return {
            "success": success,
            "installation_path": installation_path,
            "temp_dir": temp_dir if keep_temp_dir else None
        }