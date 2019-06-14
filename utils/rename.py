import glob, os

def rename(dir_t, pattern, titlePattern):
	# print("yeet")
	# sprint (glob.iglob(os.path.join(dir_t, pattern)))
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		title, ext = os.path.splitext(os.path.basename(pathAndFilename))
		end_str = (str(title) + str(ext)).replace(" ", "_")
		end_str = end_str.replace("download", "google")
		end_str = end_str.replace("images", "google")

		# print(end_str)
		new_name = str(titlePattern) + end_str
		print(new_name)

		os.rename(pathAndFilename, os.path.join(dir_t, new_name))
		# break

def rename_replace(dir_t, pattern, orig_pattern,rep_pattern):
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		# old_name, ext = os.path.splitext(os.path.basename(pathAndFilename))
		# print(end_str)
		new_name = pathAndFilename.replace(orig_pattern, rep_pattern)
		print(new_name)

		os.rename(pathAndFilename, new_name)
		# break

def rename_replace2(dir_t, pattern, separator):
	index =0
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		old_name, ext = os.path.splitext(os.path.basename(pathAndFilename))
		# print(end_str)
		lst = old_name.split(separator)

		pre = lst[0]
		print (pre)
		# post = lst[1]  
		
		new_name = dir_t + pre + "_w" + str(index)+"_" + str(ext)
		print(new_name)

		os.rename(pathAndFilename, new_name)
		index = index +1
		# break

import glob, os

def rename(dir_t, pattern, titlePattern):
	# print("yeet")
	# sprint (glob.iglob(os.path.join(dir_t, pattern)))
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		title, ext = os.path.splitext(os.path.basename(pathAndFilename))
		end_str = (str(title) + str(ext)).replace(" ", "_")
		end_str = end_str.replace("download", "google")
		end_str = end_str.replace("images", "google")

		# print(end_str)
		new_name = str(titlePattern) + end_str
		print(new_name)

		os.rename(pathAndFilename, os.path.join(dir_t, new_name))
		# break

def rename_replace(dir_t, pattern, orig_pattern,rep_pattern):
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		# old_name, ext = os.path.splitext(os.path.basename(pathAndFilename))
		# print(end_str)
		new_name = pathAndFilename.replace(orig_pattern, rep_pattern)
		print(new_name)

		os.rename(pathAndFilename, new_name)
		# break

def rename_replace2(dir_t, pattern, separator):
	index =0
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		old_name, ext = os.path.splitext(os.path.basename(pathAndFilename))
		# print(end_str)
		lst = old_name.split(separator)

		pre = lst[0]
		print (pre)
		# post = lst[1]  
		
		new_name = dir_t + pre + "_w" + str(index)+"_" + str(ext)
		print(new_name)

		os.rename(pathAndFilename, new_name)
		index = index +1
		# break


def rename_indian(dir_t, pattern, titlePattern):
	# print("yeet")
	# sprint (glob.iglob(os.path.join(dir_t, pattern)))
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		title, ext = os.path.splitext(os.path.basename(pathAndFilename))
		# print(title)
		# print (end_str)
		
		new_title = titlePattern + str(title)
		end_str = (new_title + str(ext)).replace(" ", "_")
		end_str = end_str.replace("Screen_Shot_2019-04-26_at_3.0", "")
		end_str = end_str.replace("-_2019-04-26T050049.","")
		end_str = end_str.replace("PM", "")
		end_str = end_str.replace("images", "")
		end_str = end_str.replace("download", "")

		print(end_str)
		# new_name = str(titlePattern) + end_str
		# print(new_name)

		os.rename(pathAndFilename, os.path.join(dir_t, end_str))
		# break

if __name__== "__main__":
	# rename("artist_woman_hispanic","*","A_artist_woman_hispanic_" )
	rename_indian("indian_painter_famous", "*.*", "A_painter_man_indian_")
	# rename_replace2("artist_asian_woman","*","_-_")
	# rename_replace(".","*","A_artist_asian_woman_A_artist_asian_woman_", "A_artist_asian_woman_")
	# rename_replace("painter_black_woman", "*","A_painter_woman_black_", "A_painter_woman_black_" )
	# rename(r'c:\temp\xx', r'*.doc', r'new(%s)')