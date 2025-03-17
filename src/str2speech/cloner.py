import subprocess
import os
import tempfile
import shutil

class Cloner:
    @staticmethod
    def clone_and_install(repo_url):
        temp_dir = tempfile.mkdtemp()
        original_dir = os.getcwd()
        
        try:
            os.chdir(temp_dir)
            repo_name = repo_url.split('/')[-1].replace('.git', '')
            print(f"Cloning repository from {repo_url} into temporary directory...")
            subprocess.run(['git', 'clone', repo_url], check=True)
            os.chdir(repo_name)
            print("Installing repository...")
            subprocess.run(['pip3', 'install', '.'], check=True)
            print("Successfully cloned and installed the repository!")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred: {e}")
        except Exception as e:
            print(f"Unexpected error: {e}")
        finally:
            os.chdir(original_dir)
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    print("Temporary directory cleaned up.")
            except Exception as e:
                print(f"Error cleaning up temporary directory: {e}")