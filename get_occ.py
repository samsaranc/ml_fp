import urllib3
import wget, os, csv

# urllib.urlretrieve("http://www.digimouth.com/news/media/2011/09/google-logo.jpg", "local-filename.jpg")

def download_im_URLS(dest):
	ID = 0
	http = urllib3.PoolManager()

	# big = ["http://www-personal.umich.edu/~mjskay/aws-images/google/chemist/","http://www-personal.umich.edu/~mjskay/aws-images/google/chemist/"]
	# pair = ["biologist"]
	pair = ["biologist", "chemist", "computer programmer","doctor", "engineer", "lab tech", "pharmacist", "software developer", "web developer"] 
	#lab\%20tech
	art = ["architect","carpenter", "designer"]
	out_csv = "ID, filename, class, occupation, is_female, is_poc, human_annotated \n"
	# out_csv = [["ID", "filename", "class", "occupation", "is_female", "is_poc", "human_annotated"]]

	with open("occupations.csv","a") as file1:
		file1.write(out_csv) 
	file1.close() 

	for p in pair: 

		for i in range(400):
			local_name =  dest + "S_" +   p + "_" + str(i) + ".jpg"
			img_url =  "http://www-personal.umich.edu/~mjskay/aws-images/google/" + p + "/" + str(i) + ".jpg"
			# img_url = "http://www-personal.umich.edu/~mjskay/aws-images/google/logistician/0.jpg"
			print (img_url)

			filename = local_name
			
			try: 
				# r = http.request('GET', url)
				# urllib3.urlretrieve(url, local_name);
				filename = wget.download(url=img_url, out=local_name)
				# out = [str(ID), local_name, "1", p, "-1", "-1", "0"] 
				# out_csv.append(out)
				out_r = str(ID) + ", " + local_name + ", 1, " + p + ", -1, -1, 0 \n" 
				
				ID = ID +1 

			except Exception, e:
				# filename = None
				print ("GET failed")

			if not os.path.exists(filename):
				print("wrong URl dood")
				# return make_response(jsonify({'message': 'Wrong URL'}), 404)

			with open("occupations.csv","a") as file1:
				file1.write(out_r)
				#Write to CSV 
			file1.close() 
			# with open('occupations.csv', 'wb') as csvfile:
			# 	w = csv.writer(csvfile)
			# 	w.writerow(out_csv)

	for p in art: 

		for i in range(400):
			local_name =  dest + "A_" +   p + "_" + str(i) + ".jpg"
			img_url =  "http://www-personal.umich.edu/~mjskay/aws-images/google/" + p + "/" + str(i) + ".jpg"
			# img_url = "http://www-personal.umich.edu/~mjskay/aws-images/google/logistician/0.jpg"
			print (img_url)

			filename = local_name
			
			try: 
				# r = http.request('GET', url)
				# urllib3.urlretrieve(url, local_name);
				filename = wget.download(url=img_url, out=local_name)
				out_r = str(ID) + ", " + local_name + ", 0, " + p + ", -1, -1, 0 \n" 
				ID = ID +1 

			except Exception, e:
				# filename = None
				print ("GET failed")

			if not os.path.exists(filename):
				print("wrong URl dood")
				# return make_response(jsonify({'message': 'Wrong URL'}), 404)

			with open("occupations.csv","a") as file1:
				file1.write(out_r)
				#Write to CSV 
			file1.close() 

if __name__ == '__main__':
	dest = "occupations/" 
	download_im_URLS(dest)


