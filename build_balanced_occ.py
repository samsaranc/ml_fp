import os
import random
import shutil
import sys

PERCENT_VAL = 0.2
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(THIS_DIR, '..\\data_remake_orig_Nov2018')
BASE_DIR = os.path.join(THIS_DIR, '..\\data_split.test')
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
TRA_DIR = os.path.join(BASE_DIR, 'tra')
if not os.path.exists(TRA_DIR):
    os.makedirs(TRA_DIR)
VAL_DIR = os.path.join(BASE_DIR, 'val')
if not os.path.exists(VAL_DIR):
    os.makedirs(VAL_DIR)
# SUBDIR_NAMES = {'notproana':[]}

# f = open(os.path.join(THIS_DIR, "notproana_tags.txt"), "r")
# for line in f:
#     tag = line.strip().split('.py ')[1]
#     if tag not in SUBDIR_NAMES['notproana']:
#         SUBDIR_NAMES['notproana'].append(tag)

#move proana images
print("Copying proana images")
pa_images = os.listdir(os.path.join(SRC_DIR, 'proana'))
SPLIT = int(round(PERCENT_VAL * len(pa_images)))
END = len(pa_images)
print("SPLIT = " + str(SPLIT))
print("END = " + str(END))
random.shuffle(pa_images)
#copy val
print("   to val")
dest_dir = os.path.join(VAL_DIR, "PA")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for i in pa_images[:SPLIT]:
    src = os.path.join(SRC_DIR, 'proana', i)
    dest = os.path.join(dest_dir, i)
    shutil.copy(src, dest)
#copy tra
print("   to tra")
dest_dir = os.path.join(TRA_DIR, "PA")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for i in pa_images[SPLIT:]:
    src = os.path.join(SRC_DIR, 'proana', i)
    dest = os.path.join(dest_dir, i)
    shutil.copy(src, dest)

#move notproana images
print("Copying from notproana")
src_dir = os.path.join(SRC_DIR, 'notproana')
images = os.listdir(src_dir)
random.shuffle(images)
thissplit = SPLIT
thisend = END
if len(images) > len(pa_images):
    thissplit = int(round(PERCENT_VAL * len(pa_images))) #fixed from len(images)
    thisend = len(pa_images) #fixed from len(images)
print("length val = " + str(len(images[:thissplit])))
print("length tra = " + str(len(images[thissplit:thisend])))
# move val
print("   to val")
dest_dir = os.path.join(VAL_DIR, "NPA")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for i in images[:thissplit]:
    src = os.path.join(src_dir,i)
    dest = os.path.join(dest_dir,i)
    shutil.copy(src,dest)
# move tra
print("   to tra")
dest_dir = os.path.join(TRA_DIR, "NPA")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
j = 0;
for i in images[thissplit:thisend]:
    src = os.path.join(src_dir,i)
    dest = os.path.join(dest_dir,i)
    shutil.copy(src,dest)
    print(j)
    j = j+1
