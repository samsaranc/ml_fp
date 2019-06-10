import os
import random
import shutil
import sys

PERCENT_VAL = 0.2
THIS_DIR = os.path.dirname(os.path.realpath(__file__))
SRC_DIR = os.path.join(THIS_DIR, 'data')
BASE_DIR = os.path.join(THIS_DIR, 'data_split.test')
if not os.path.exists(BASE_DIR):
    os.makedirs(BASE_DIR)
TRA_DIR = os.path.join(BASE_DIR, 'tra')
if not os.path.exists(TRA_DIR):
    os.makedirs(TRA_DIR)
VAL_DIR = os.path.join(BASE_DIR, 'val')
if not os.path.exists(VAL_DIR):
    os.makedirs(VAL_DIR)
# SUBDIR_NAMES = {'scientist':[]}

# f = open(os.path.join(THIS_DIR, "scientist_tags.txt"), "r")
# for line in f:
#     tag = line.strip().split('.py ')[1]
#     if tag not in SUBDIR_NAMES['scientist']:
#         SUBDIR_NAMES['scientist'].append(tag)
fashion_images = os.listdir(os.path.join(SRC_DIR, 'fashion'))
scientist_images = os.listdir(os.path.join(SRC_DIR, 'scientist'))
artist_images = os.listdir(os.path.join(SRC_DIR, 'artist'))
len_sc =len(scientist_images)
len_fa =len(fashion_images)
len_ar =len(artist_images)
min_len_images = min(len_sc,len_fa,len_ar)
print("min", min_len_images)

'''
************************************************************************
************************ FASHION ***************************************
************************************************************************
'''
#move proana images
print("Copying fashion images")

SPLIT = int(round(PERCENT_VAL * min_len_images))
END = min_len_images
thissplit = SPLIT
thisend = END
random.shuffle(fashion_images)

print("SPLIT = " + str(SPLIT))
print("END = " + str(END))
print("length val = " + str(len(fashion_images[:thissplit])))
print("length tra = " + str(len(fashion_images[thissplit:thisend])))

#copy val
print("   to val")
dest_dir = os.path.join(VAL_DIR, "F")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for i in fashion_images[:SPLIT]:
    src = os.path.join(SRC_DIR, 'fashion', i)
    dest = os.path.join(dest_dir, i)
    shutil.copy(src, dest)
#copy tra
print("   to tra")
dest_dir = os.path.join(TRA_DIR, "F")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for i in fashion_images[SPLIT:END]:
    src = os.path.join(SRC_DIR, 'fashion', i)
    dest = os.path.join(dest_dir, i)
    shutil.copy(src, dest)

'''
************************************************************************
************************ SCIENTIST ***************************************
************************************************************************
'''

#move scientist images
print("Copying from scientist")
src_dir = os.path.join(SRC_DIR, 'scientist')
random.shuffle(scientist_images)
thissplit = SPLIT
thisend = END

# thissplit = int(round(PERCENT_VAL * min_len_images))
# thisend = min_len_images

print("length val = " + str(len(scientist_images[:thissplit])))
print("length tra = " + str(len(scientist_images[thissplit:thisend])))
# move val
print("   to val")
dest_dir = os.path.join(VAL_DIR, "S")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for i in scientist_images[:thissplit]:
    src = os.path.join(src_dir,i)
    dest = os.path.join(dest_dir,i)
    shutil.copy(src,dest)
# move tra
print("   to tra")
dest_dir = os.path.join(TRA_DIR, "S")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
j = 0;
for i in scientist_images[thissplit:thisend]:
    src = os.path.join(src_dir,i)
    dest = os.path.join(dest_dir,i)
    shutil.copy(src,dest)
    # print(j)
    j = j+1
print(j)

'''
************************************************************************
************************ ARTIST ***************************************
************************************************************************
'''

#move artist images
print("Copying from artist")
src_dir = os.path.join(SRC_DIR, 'artist')
random.shuffle(artist_images)
thissplit = SPLIT
thisend = END

# thissplit = int(round(PERCENT_VAL * min_len_images))
# thisend = min_len_images
print("length val = " + str(len(artist_images[:thissplit])))
print("length tra = " + str(len(artist_images[thissplit:thisend])))
# move val
print("   to val")
dest_dir = os.path.join(VAL_DIR, "A")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
for i in artist_images[:thissplit]:
    src = os.path.join(src_dir,i)
    dest = os.path.join(dest_dir,i)
    shutil.copy(src,dest)
# move tra
print("   to tra")
dest_dir = os.path.join(TRA_DIR, "A")
if not os.path.exists(dest_dir):
    os.makedirs(dest_dir)
j = 0;
for i in artist_images[thissplit:thisend]:
    src = os.path.join(src_dir,i)
    dest = os.path.join(dest_dir,i)
    shutil.copy(src,dest)
    # print(j)
    j = j+1
print(j)
