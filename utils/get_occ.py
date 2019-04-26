import urllib3
import wget, os, csv

def download_sci_im_URLS(dest,id_start, synset_name="NO_SYN", wnid="NULL"):
	ID = id_start
	http = urllib3.PoolManager()
	urls = [""]

	out_csv = "ID, filename, class, occupation, is_female, is_poc, human_annotated, wnid, synset_name \n"
	#ut_csv = [["ID", "filename", "class", "occupation", "is_female", "is_poc", "human_annotated"]]

	#with open("scientist.csv","a") as file1:
	#	file1.write(out_csv) 
	# file1.close() 

	for p in urls: 
		local_name =  dest + "S_" +  "imagenet_" + str(ID) + ".jpg"
		# img_url =  "http://www-personal.umich.edu/~mjskay/aws-images/google/" + p + "/" + str(i) + ".jpg"
		# img_url = "http://www-personal.umich.edu/~mjskay/aws-images/google/logistician/0.jpg"
		img_url = p
		print (img_url)
		print(local_name)

		filename = local_name
		
		try: 
			# r = http.request('GET', url)
			# urllib3.urlretrieve(url, local_name);
			filename = wget.download(url=img_url, out=local_name)
			# out = [str(ID), local_name, "1", p, "-1", "-1", "0"] 
			# out_csv.append(out)
			out_r = str(ID) + ", " + local_name + ", 1, " + p + ", -1, -1, 0, " + str(wnid) + "\n" 
			
			ID = ID +1 

			with open("scientist.csv","a") as file1:
				file1.write(out_r)
				#Write to CSV 
			file1.close() 

		except Exception, e:
			# filename = None
			print ("GET failed")

		if not os.path.exists(filename):
			print("wrong URL dood")
			# return make_response(jsonify({'message': 'Wrong URL'}), 404)

	
		# with open('occupations.csv', 'wb') as csvfile:
		# 	w = csv.writer(csvfile)
		# 	w.writerow(out_csv)


def download_occ_im_URLS(dest):
	ID = 0
	http = urllib3.PoolManager()

	# big = ["http://www-personal.umich.edu/~mjskay/aws-images/google/chemist/","http://www-personal.umich.edu/~mjskay/aws-images/google/chemist/"]
	# pair = ["biologist"]
	pair = ["biologist", "chemist", "computer programmer","doctor", "engineer", "lab tech", "pharmacist", "software developer", "web developer"] 
	#lab\%20tech
	art = ["architect","carpenter", "designer"]
	art2 = ["baker", "drafter","chef", "photographer", "editor"]
	art3 = ["building painter"]
	out_csv = "ID, filename, class, occupation, is_female, is_poc, human_annotated \n"
	# out_csv = [["ID", "filename", "class", "occupation", "is_female", "is_poc", "human_annotated"]]

	# with open("occupations.csv","a") as file1:
	# 	file1.write(out_csv) 
	# file1.close() 

	# for p in pair: 

	# 	for i in range(400):
	# 		local_name =  dest + "S_" +   p + "_" + str(i) + ".jpg"
	# 		img_url =  "http://www-personal.umich.edu/~mjskay/aws-images/google/" + p + "/" + str(i) + ".jpg"
	# 		# img_url = "http://www-personal.umich.edu/~mjskay/aws-images/google/logistician/0.jpg"
	# 		print (img_url)

	# 		filename = local_name
			
	# 		try: 
	# 			# r = http.request('GET', url)
	# 			# urllib3.urlretrieve(url, local_name);
	# 			filename = wget.download(url=img_url, out=local_name)
	# 			# out = [str(ID), local_name, "1", p, "-1", "-1", "0"] 
	# 			# out_csv.append(out)
	# 			out_r = str(ID) + ", " + local_name + ", 1, " + p + ", -1, -1, 0 \n" 
				
	# 			ID = ID +1 

	# 		except Exception, e:
	# 			# filename = None
	# 			print ("GET failed")

	# 		if not os.path.exists(filename):
	# 			print("wrong URl dood")
	# 			# return make_response(jsonify({'message': 'Wrong URL'}), 404)

	# 		with open("occupations.csv","a") as file1:
	# 			file1.write(out_r)
	# 			#Write to CSV 
	# 		file1.close() 
	# 		# with open('occupations.csv', 'wb') as csvfile:
	# 		# 	w = csv.writer(csvfile)
	# 		# 	w.writerow(out_csv)

	for p in art3: 

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
	dest_s = "scientist/" 
	# dest = "occupations/" 
	# download_occ_im_URLS(dest)
	syn_name = "scientist_gen"
	download_sci_im_URLS(dest_s,486, syn_name, "n10560637")
	