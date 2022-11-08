import csv,glob,os

# look for csv file in the input dir
csv_file = glob.glob('/gs/input/*.csv')
if not len(csv_file) >=1:
    print("You don't have CSV file")
    exit()

# checking input dir 
fls = glob.glob("/gs/input/*")
if len(fls) < 2:
    print("input folder is empty")
    exit()
    print('dsjkjsfhdsf')
    
print("sample of data in the input folder  : ")
print('------------------------------------')
m = len(fls)
if m > 10 :
    m = 10
print(fls[:m])
print('-----------processing')
csv_file = csv_file[0]
csv_file_data = open(csv_file)
csvreader = csv.reader(csv_file_data)

train = val = test = []
for i,row in enumerate(csvreader):
    image = row[1]
    group = row[0]
    Class = row[2]

    if 'gs://' in image :
        image = image.replace('gs://')
    image = f"/gs/input/{image}"
    if group == "TRAIN" : 
        train.append((image,Class))
    elif group == "TEST":
        test.append((image,Class))
    elif group == "VAL":
        val.append((image,Class))
    
os.system("mkdir /gs/input/train/")
os.system("mkdir /gs/input/test/")
os.system("mkdir /gs/input/validation/")
for im in train:
    if not os.path.isdir(f'/gs/input/train/{im[1]}'):
        os.system(f"mkdir /gs/input/train/{im[1]}")
    os.system(f"cp {im[0]} /gs/input/train/{im[1]}/{im[0].split('/')[-1]}")
for im in test:
    if not os.path.isdir(f'/gs/input/test/{im[1]}'):
        os.system(f"mkdir /gs/input/test/{im[1]}")
    os.system(f"cp {im[0]} /gs/input/test/{im[1]}/{im[0].split('/')[-1]}")
for im in val:
    if not os.path.isdir(f'/gs/input/validation/{im[1]}'):
        os.system(f"mkdir /gs/input/validation/{im[1]}")
    os.system(f"cp {im[0]} /gs/input/validation/{im[1]}/{im[0].split('/')[-1]}")

os.system("python3.7 tensorflow_tpu_models/tools/data_converter/image_classification/simple_example.py --data_path=/gs/input/ --generate=False --save_dir=data_final/")
os.system("cp -r data_final/image_classification_builder/Simple/0.1.0/ /gs/output")
