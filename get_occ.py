import urllib3
import wget, os

# urllib.urlretrieve("http://www.digimouth.com/news/media/2011/09/google-logo.jpg", "local-filename.jpg")

def download_im_URLS():
	http = urllib3.PoolManager()

	big = ["http://www-personal.umich.edu/~mjskay/aws-images/google/chemist/","http://www-personal.umich.edu/~mjskay/aws-images/google/chemist/"]
	pair = ["biologist",
	# pair = ["biologist", "chemist", "computer programmer","doctor", "engineer", "lab\%20tech", "pharmacist", "software developer", "web developer"] 
	#lab\%20tech
	art = ["architect","carpenter", "designer"]

	for p in pair: 

		for i in range(400):
			local_name = p + "_" + str(i) + ".jpg"
			url =  "http://www-personal.umich.edu/~mjskay/aws-images/google/" + p + "/" + i + ".jpg"

			r = http.request('GET',
			try: 
				urllib3.urlretrieve(url, local_name);
			except Exception, e:
				filename = None
				print ("GET failed")

			if not os.path.exists(filename):
				print("wrong URl dood")
				# return make_response(jsonify({'message': 'Wrong URL'}), 404)

if __name__ == "__main__":
	download_im_URLS()

