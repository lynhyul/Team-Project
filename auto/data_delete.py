import os,shutil

if os.path.isdir("./data/training"):
    shutil.rmtree("./data/training")
if os.path.isdir("./logs"):
    shutil.rmtree("./logs")


os.mkdir("./data/training")
os.mkdir("./logs")

f = open("./data/training/data.csv", 'w')
f.close()