#!ls /home/apsisdev/mobassir/data/molat/Molat-PDF-Files-18/Molat-PDF-Files
#!pip install pymupdf

import zipfile
import os
import fitz
from pathlib import Path
import shutil
from tqdm import tqdm
from os.path import basename
import re

#taken from : https://www.kaggle.com/xhlulu/recursion-2019-load-resize-and-save-images

def zip_and_remove(path):
    ziph = zipfile.ZipFile(f'{path}.zip', 'w', zipfile.ZIP_DEFLATED)
    for root, dirs, files in os.walk(path):
        for file in tqdm(files):
            file_path = os.path.join(root, file)
            #https://stackoverflow.com/questions/16091904/python-zip-how-to-eliminate-absolute-path-in-zip-archive-if-absolute-paths-for
            ziph.write(file_path, basename(file_path))
            #os.remove(file_path)
    ziph.close()
    shutil.rmtree(path)
    


def check_filename(string):
    regex = re.compile('[@!#$%^&*()"<=>?/\+|}`{~:]')
    if(regex.search(string) == None):
        return string
    else:
        print(f"String {string} is not accepted.")
        #fix = string.replace('[@!#$%^&*()"<=>?/\+|}`{~:]','_')
        fix = re.sub(r'[@!#$%^&*()"<=>?/\+|}`{~:]','_', string)
        print(fix)
        return fix
    
def pdf2imgs(pdffile):
    try:
        doc = fitz.open(pdffile)
    except:
        print("An exception occurred")
        return
    part = Path(pdffile).stem
    part = part.replace(" ", "_")
    part = check_filename(part)
    
    print("working on this pdf -> ",part)
    
    os.mkdir(f'/home/apsisdev/mobassir/data/molat{folder}/{part}')
    
    for page_number in range(len(doc)):
        page = doc.load_page(page_number)  # number of page load_page
        zoom = 2    # zoom factor https://github.com/pymupdf/PyMuPDF/issues/181
        mat = fitz.Matrix(zoom, zoom)
        pix = page.get_pixmap(matrix = mat)
        output = f"/home/apsisdev/mobassir/data/molat{folder}/{part}/{part}_page_{page_number}.png"
        pix.save(output)
    zip_and_remove(f'/home/apsisdev/mobassir/data/molat{folder}/{part}')
    
    

if __name__ == '__main__' :
    for ind in range(1,len(os.listdir(f'/home/apsisdev/mobassir/data/molat'))):
        folder = ind
        if(os.path.exists(f'/home/apsisdev/mobassir/data/molat{folder}')):
            print("deleting old data...")
            shutil.rmtree(f'/home/apsisdev/mobassir/data/molat{folder}')
            os.mkdir(f'/home/apsisdev/mobassir/data/molat{folder}')
        else:
            os.mkdir(f'/home/apsisdev/mobassir/data/molat{folder}')

        file_path = f'/home/apsisdev/mobassir/data/molat/Molat-PDF-Files-{folder}/Molat-PDF-Files/'
        directory = []
        for dirname, _, filenames in os.walk(file_path):
            for filename in filenames:
                directory.append(os.path.join(dirname, filename))
        print(folder,"->",len(directory))

        for i in range(len(directory)):
            try:
                pdf2imgs(directory[i])
            except:
                print("couldn't process ->",directory[i])
                continue
    
# #file_path = '/home/apsisdev/mobassir/data/molat/Molat-PDF-Files-17/Molat-PDF-Files/'

# folder = 11
# file_path = f'/home/apsisdev/mobassir/data/molat/molat{folder}'

# directory = []
# for dirname, _, filenames in os.walk(file_path):
#     for filename in filenames:
#         directory.append(os.path.join(dirname, filename))
# directory
# if __name__ == '__main__' :
    

#     for i in range(len(directory)):
#         part = Path(directory[i]).stem
#         part = part.replace(" ", "_")
#         fixed = check_filename(part)
#         os.rename(f'/home/apsisdev/mobassir/data/molat/molat{folder}/{part}.zip', f'/home/apsisdev/mobassir/data/molat/molat{folder}/{fixed}.zip')




