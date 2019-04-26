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

def rename_celis(dir_t, pattern, titlePattern):
	# print("yeet")
	# sprint (glob.iglob(os.path.join(dir_t, pattern)))
	for pathAndFilename in glob.iglob(os.path.join(dir_t, pattern)):
		# print(pathAndFilename)
		title, ext = os.path.splitext(os.path.basename(pathAndFilename))
		# print(title)
		end_str = (str(title) + str(ext)).replace(" ", "_")
		# print (end_str)
		end_str = end_str.replace("_Search", "")
		end_str = end_str.replace("female", "woman")
		end_str = end_str.replace("(", "_celis-(")
		end_str = end_str.replace("_-", "")

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


if __name__== "__main__":
	rename_celis("PainterFemale","*.*","A_" )
	# scientist female - Google Search(
	# rename_replace2("artist_asian_woman","*","_-_")
	# rename_replace("painter_black_woman", "*","A_painter_womman_black_", "A_painter_woman_black_" )
	# rename(r'c:\temp\xx', r'*.doc', r'new(%s)')