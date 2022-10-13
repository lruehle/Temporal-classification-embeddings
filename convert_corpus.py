import pandas as pd
import os
#rerun for different time folders
# or adjust code to run over directories as well and take name from dir as time-variable

dn = os.path.abspath('convert_corpus.py')
txt_src = os.path.join(os.path.dirname(dn),'corpora\dta\Belletristik\\1600')
output_src = os.path.join(os.path.dirname(dn),'corpora\processed') #albertinus_landtstoertzer01_1615.txt')


#read_file = pd.read_csv (os.path.join(txt_src,'albertinus_landtstoertzer01_1615.txt'),sep=".\\n",header=None)
#read_file.to_csv (os.path.join(output_src,'merged_file.csv'), index=None)

with open(os.path.join(output_src,'1600_corpus_proc.csv'), 'w+') as outfile: #w+ should delete file content
    for file in os.listdir(txt_src):
        if file.endswith(".txt"): 
            file_path = f"{txt_src}\{file}"
            #output_path = f"{output_src}\{file}"
            #df = pd.read_fwf(file_path)
            df = pd.read_csv(file_path, sep=".\\n", header=None,names=["txt","year"]) 
            df = df.replace("/|,|[^\w\s]","",regex=True)
            #df = df[df['txt'].str.contains(' ')] #drop lines with only one word -> references to role/active speaker etc.
            df= df[df['txt'].str.count(' ') > 2]
            #convert all to lower() done in simple preprocessing at model level
            #df = df.replace({'ä':'ae','ö':'oe','ü':'ue','Ä':'Ae','Ö':'Oe','Ü':'Ue','ß':'ss'}) funktioniert nicht
            df = df.replace('\d+', '',regex=True)
            df = df.replace("ä","ae",regex=True)
            df = df.replace("ö","oe",regex=True)
            df = df.replace("ü","ue",regex=True)
            df = df.replace("Ä","Ae",regex=True)
            df = df.replace("Ö","Oe",regex=True)
            df = df.replace("Ü","Ue",regex=True)
            df = df.replace("ß","ss",regex=True)
            df['year']="1600"
            #df.insert(1, 'year', '1600')
            df.to_csv(os.path.join(output_src,'1600_corpus_proc.csv'),mode="a",index=False)
            #df.to_csv(output_path)

## randomize documents
#ds = pd.read_csv(os.path.join(output_src,'merged_file.csv')) 
#ds = ds.sample(frac=1) #random shuffle
#try out df['removed_stop_word']  = df['x'].apply(stop_word_removal)
#ds.to_csv(os.path.join(output_src,'merged_file2.csv'),index = False)

"""with open(os.path.join(os.path.dirname(dn),'corpora\dta\Belletristik\\1600\\albertinus_landtstoertzer01_1615.txt'),'r',encoding="utf8") as f:
    stripped = (line.strip() for line in f)
    lines = (line.split("/") for line in stripped if line)
    #print(lines)"""


#read_file.to_csv(os.path.join(os.path.dirname(dn),'corpora\processed\albertinus_landtstoertzer01_1615.csv'),index = None)
