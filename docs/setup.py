import subprocess

def install_dependencies():
    subprocess.run(['pip', 'install', '-r', 'requirements.txt'])
    
if __name__ == '__main__':
  install_dependencies()
  print("Setup completed successfully!")