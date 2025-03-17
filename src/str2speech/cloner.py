import subprocess
import os
import tempfile
import shutil

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
            
            subprocess.run(
                ['git', 'clone', repo_url], 
                check=True,
                capture_output=True,
                text=True
            )
            
            if os.path.exists(repo_name):
                os.chdir(repo_name)
                print("Installing repository...")
                subprocess.run(
                    ['pip3', 'install', '-e', '.'], 
                    check=True,
                    capture_output=True,
                    text=True
                )
                
                installation_path = os.path.abspath('.')
                success = True
                print("Successfully cloned and installed the repository!")
            else:
                print(f"Repository directory {repo_name} was not created after git clone")
                
        except subprocess.CalledProcessError as e:
            print(f"Command '{e.cmd}' failed with exit code {e.returncode}")
            print(f"STDOUT: {e.stdout}")
            print(f"STDERR: {e.stderr}")
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