from os import listdir
from PIL import Image
import shutil, os

def isIMG(filename): 
	imgs = [".png",".PNG", ".JPG",".jpeg", ".jpg", ".jpeg", ".tiff"]

	for i in imgs:
		if filename.endswith(i):
			return True

	return False

def move_suckers(bad): 
	
	for b in bad: 
		# os.rename("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
		abs_p = os.path.abspath(b)
		dest_path = "/Users/samsaracounts/ml_fp/data/scientist/bad/" + str(b)
		# os.rename(abs_p, dest_path)
		shutil.move(abs_p, dest_path)

	return

def find_bad(dirname) :
	bad = []

	for filename in listdir(dirname):
	if isIMG(filename):
		if filename.endswith(".tiff") or filename.endswith(".TIFF"):
			print("TIFF", filename)

		try:
			img = Image.open(dirname+filename) # open the image file
			img.verify() # verify that it is, in fact an image
		except (IOError, SyntaxError) as e:
			print(filename) # print out the names of corrupt files
			bad.append(filename)

	return bad


if __name__ == "__main__":
	bad = find_bad('./')
	move_suckers(bad)

'''	
bad = ["S_imagenet_300.jpg", 
			"S_bing_46.jpg", 
			"S_imagenet_261.jpg", 
			"S_imagenet_329.jpg", 
			"S_bing_86.jpg", 
			"S_imagenet_302.jpg", 
			"S_imagenet_138.jpg", 
			"S_imagenet_267.jpg", 
			"S_imagenet_259.jpg", 
			"S_imagenet_310.jpg", 
			"S_imagenet_161.jpg", 
			"S_imagenet_363.jpg", 
			"S_imagenet_377.jpg", 
			"S_imagenet_217.jpg", 
			"S_imagenet_376.jpg", 
			"S_imagenet_362.jpg", 
			"S_imagenet_174.jpg", 
			"S_imagenet_160.jpg", 
			"S_imagenet_176.jpg", 
			"S_imagenet_162.jpg", 
			"S_imagenet_348.jpg", 
			"S_imagenet_228.jpg", 
			"S_imagenet_188.jpg", 
			"S_imagenet_177.jpg", 
			"S_imagenet_365.jpg", 
			"S_imagenet_205.jpg", 
			"S_bing_22.jpg", 
			"S_imagenet_238.jpg", 
			"S_imagenet_358.jpg", 
			"S_imagenet_172.jpg", 
			"S_imagenet_158.jpg", 
			"S_imagenet_164.jpg", 
			"S_imagenet_170.jpg", 
			"S_imagenet_366.jpg", 
			"S_imagenet_372.jpg", 
			"S_imagenet_373.jpg", 
			"S_imagenet_159.jpg", 
			"S_imagenet_154.jpg", 
			"S_imagenet_342.jpg", 
			"S_imagenet_222.jpg", 
			"S_imagenet_182.jpg", 
			"S_imagenet_157.jpg", 
			"S_imagenet_382.jpg", 
			"S_imagenet_142.jpg", 
			"S_imagenet_152.jpg", 
			"S_imagenet_185.jpg", 
			"S_imagenet_224.jpg", 
			"S_bing_17.jpg", 
			"S_imagenet_386.jpg", 
			"S_imagenet_345.jpg", 
			"S_imagenet_351.jpg", 
			"S_imagenet_179.jpg", 
			"S_imagenet_151.jpg", 
			"S_imagenet_186.jpg", 
			"S_imagenet_353.jpg", 
			"S_imagenet_384.jpg", 
			"S_imagenet_227.jpg", 
			"S_bing_14.jpg", 
			"S_imagenet_187.jpg", 
			"S_imagenet_150.jpg", 
			"S_imagenet_309.jpg", 
			"S_imagenet_269.jpg", 
			"S_imagenet_296.jpg", 
			"S_imagenet_297.jpg", 
			"S_imagenet_268.jpg", 
			"S_imagenet_485.jpg", 
			"S_imagenet_336.jpg", 
			"S_imagenet_322.jpg", 
			"S_imagenet_257.jpg", 
			"S_imagenet_323.jpg", 
			"S_imagenet_327.jpg", 
			"S_imagenet_253.jpg", 
			"S_imagenet_318.jpg", 
			"S_imagenet_481.jpg", 
			"S_imagenet_250.jpg", 
			"S_imagenet_245.jpg"]
'''