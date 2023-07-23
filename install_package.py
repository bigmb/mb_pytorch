#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import subprocess
import sys

py_version = sys.version
print(py_version)
if py_version[:4] == '3.10':
    py_requires = 'python3.10'
else:
    py_requires = 'python3.8'
print(py_requires)

file = os.getcwd() 


#subprocess.run(["cd",file]), check=True, stdout=subprocess.PIPE).stdout
os.system('cd ' + file)

os.system('./make_version.sh')

print("version file updated")
print('*'*100)

subprocess.run(["git", "add", "."], check=True, stdout=subprocess.PIPE).stdout
subprocess.run(["git", "commit", "-am", "Bug fix commit"], check=True, stdout=subprocess.PIPE).stdout

print('git commit done')

subprocess.run(["git", "pull"], check=True, stdout=subprocess.PIPE).stdout
print('git pull done')
print('*'*100)

subprocess.run(["git", "push"], check=True, stdout=subprocess.PIPE).stdout
print('*'*100)
print('removing dist and build folders')

if os.path.exists(file+'/dist'):
    os.system('sudo rm -rf '+file+'/dist')
    os.system('sudo rm -rf '+file+'/build')
#subprocess.run(["ls"]),check=True, stdout=subprocess.PIPE).stdout
os.system("ls")

os.system(py_requires + ' -m setup bdist_wheel')

print('*'*100)
print('wheel built')
os.system(py_requires + ' -m pip install '+file + '/dist/' +os.listdir(file +'/dist')[-1])

print('package installed')
print('*'*100)
os.system(py_requires + ' -m twine upload dist/*')