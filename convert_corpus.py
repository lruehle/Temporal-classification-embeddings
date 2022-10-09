import pandas as pd
import os


dn = os.path.abspath('convert_corpus.py')
txt_src = os.path.join(os.path.dirname(dn),'corpora\dta\Belletristik\\1600')
output_src = os.path.join(os.path.dirname(dn),'corpora\processed') #albertinus_landtstoertzer01_1615.txt')




"""with open("result.txt", "wb") as outfile:
    for file in os.listdir(txt_src):
        if file.endswith(".txt"):
            with open(file, "rb") as infile:
                outfile.write(infile.read())"""


#read_file = pd.read_csv (os.path.join(txt_src,'albertinus_landtstoertzer01_1615.txt'),sep=".\\n",header=None)
#read_file.to_csv (os.path.join(output_src,'merged_file.csv'), index=None)

with open(os.path.join(output_src,'merged_file.csv'), 'w') as outfile:
    #delete file content
    for file in os.listdir(txt_src):
        if file.endswith(".txt"): 
            file_path = f"{txt_src}\{file}"
            #output_path = f"{output_src}\{file}"
            #df = pd.read_fwf(file_path)
            df = pd.read_csv(file_path, sep=".\\n", header=None) 
            df = df.replace("/|,|[^\w\s]","",regex=True)
            df.to_csv(os.path.join(output_src,'merged_file.csv'),mode="a",index=False, header=False)
            #df.to_csv(output_path)
        

"""with open(os.path.join(os.path.dirname(dn),'corpora\dta\Belletristik\\1600\\albertinus_landtstoertzer01_1615.txt'),'r',encoding="utf8") as f:
    stripped = (line.strip() for line in f)
    lines = (line.split("/") for line in stripped if line)
    #print(lines)"""


#read_file.to_csv(os.path.join(os.path.dirname(dn),'corpora\processed\albertinus_landtstoertzer01_1615.csv'),index = None)
