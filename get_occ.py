import urllib3
import wget, os, csv

def download_sci_im_URLS(dest):
	ID = 78
	http = urllib3.PoolManager()

	# urls = ["https://i.pinimg.com/originals/27/22/2a/27222a5b6eab9480035e28e060369b88.jpg", "https://www.biography.com/.image/t_share/MTM2ODA1ODU5OTkxNzU4NDMz/george_washington_carver_promojpg.jpg", 
	# 		"https://i.pinimg.com/736x/3a/76/b0/3a76b0e0a6cab074377d9c95e7dc4583.jpg", "http://media.gettyimages.com/photos/face-of-a-black-man-in-lab-coat-using-microscope-picture-id478919606?s=612x612",
	# 		"https://upload.wikimedia.org/wikipedia/commons/8/82/Neil_deGrasse_Tyson_in_June_2017_%28cropped%29.jpg", "https://i.pinimg.com/236x/cd/26/2a/cd262a183458fa1e364e0d2595cdc184.jpg",
	# 		"https://s3.envato.com/files/221528915/__videouploadss--84220__p2.jpg", "http://i.huffpost.com/gen/1176789/thumbs/o-STEM-PROGRAMS-FOR-BLACK-MEN-facebook.jpg",
	# 		"https://photos.the-scientist.com/content/figures/0890-3670-051107-S11-1-1.jpg", "http://media2.s-nbcnews.com/j/msnbc/Components/Photos/z_Projects_in_progress/050418_Einstein/050321_neil_tyson_bcol_9a.grid-6x2.jpg",
	# 		"https://www.uncnri.org/wp-content/uploads/2017/06/male-female-white-black-scientists.jpg", "https://fthmb.tqn.com/AXj08Niich95gqD0P7lwTOnOs9U=/1200x1200/filters:fill(auto,1)/African_American-Black_Innovations_..._where_would_we_be_without_them-_140211-M-TJ398-001-588f27ef5f9b5874eeec2dbb.jpg",
	# 		"https://i.pinimg.com/736x/d7/99/0b/d7990b04fda6439e02d89ab16377dd33--african-american-inventors-american-history.jpg", "https://www.biography.com/.image/t_share/MTE5NTU2MzE2NjA0MzY4Mzk1/primary-photo.jpg",
	# 		"https://www.ebony.com/wp-content/uploads/2016/07/scientist_caro_original_48773.jpg", "https://www.blackenterprise.com/wp-content/blogs.dir/1/files/2013/06/STEMmalescience.jpg",
	# 		"http://66.media.tumblr.com/4fd0604eba00ee3474fa62710949a10f/tumblr_n6co4ovNae1qbtvcso1_1280.jpg", "https://cdn-images-1.medium.com/max/1600/1*MZ096hqWUv6PYHa0ANXlpA.jpeg",
	# 		"https://i.pinimg.com/736x/3e/db/10/3edb10a08e01e48cf46e377a88ab16c7.jpg", "https://i.pinimg.com/736x/50/ac/42/50ac429851a318b10a27a2ac7b0a4943--agricultural-extension-modern-agriculture.jpg",
	# 		"http://www.livescience.com/images/i/000/029/781/original/male-female-scientists-120808.jpg?1344457169", "https://www.bing.com/th?id=OIP.nKVvpJVxVLnQCb0NvpwkFQHaFc&w=264&h=194&c=7&o=5&dpr=2&pid=1.7",
	# 		"https://www.bing.com/images/search?view=detailV2&id=4B6673D3370EE4D3DBDEBC33E972BBF359C101A1&thid=OIP.mD9DwSX14djszvgLoc5DowHaFc&exph=957&expw=1300&q=male+asian+scientist&selectedindex=10&vt=0&eim=1,2,6", "https://www.bing.com/th?id=OIP.2e5StppmuLc151pXtN080wHaFN&w=275&h=194&c=7&o=5&dpr=2&pid=1.7",
	# 		"https://www.bing.com/th?id=OIP.MsWQN9IHbWpJX_rYQe18JAAAAA&w=129&h=194&c=7&o=5&dpr=2&pid=1.7", "https://www.bing.com/th?id=OIP.nEOf85IcGL43KSmHN-3cTAHaFc&w=264&h=194&c=7&o=5&dpr=2&pid=1.7",
	# 		"https://www.bing.com/th?id=OIP.FXiVanPB3KeBQLIzVEWguwAAAA&w=129&h=194&c=7&o=5&dpr=2&pid=1.7", "https://www.bing.com/th?id=OIP.W7U_BOBUcQ3pmtqFZsHfgwHaE6&w=213&h=160&c=7&o=5&dpr=2&pid=1.7",
	# 		"https://www.bing.com/th?id=OIP.yYi_0j2xY4AlNdFvS9l45gHaEL&w=213&h=160&c=7&o=5&dpr=2&pid=1.7", "https://www.bing.com/th?id=OIP._hIoj3Sv1W6LuPIs3j85sQAAAA&w=126&h=197&c=7&o=5&dpr=2&pid=1.7",
	# 		"https://www.bing.com/th?id=OIP.DQC9xm6IaZt2thMG6yhxBQHaFY&w=219&h=159&c=7&o=5&dpr=2&pid=1.7", "https://www.bing.com/th?id=OIP.vhq37diFYUAa3H__iO2U7gHaFS&w=288&h=204&c=7&o=5&dpr=2&pid=1.7",
	# 		"https://www.bing.com/th?id=OIP.jacct4cYiQXvID23xXcfIAHaKD&w=134&h=182&c=7&o=5&dpr=2&pid=1.7", "http://www.massey.ac.nz/massey/fms/Colleges/college-of-health/mifst/Asian-male-studying.jpg?51074CE70717B51C5326A4A32E96BBBB",
	# 		"https://i.cbc.ca/1.3672148.1468108801!/fileImage/httpImage/image.jpg_gen/derivatives/16x9_620/edward-ho.jpg", "https://il3.picdn.net/shutterstock/videos/308290/thumb/1.jpg?i10c=img.resize(height:160)",
	# 		"http://faobmb.com/wp-content/uploads/2018/01/varodom_YSA_2017_for-webpage-231x300.jpg", "https://image.shutterstock.com/image-photo/relaxed-asian-senior-man-using-260nw-1145698007.jpg",
	# 		"https://l7.alamy.com/zooms/56fca8c3205f4c70bacc397cc286230b/male-hispanic-lab-technician-scientist-in-research-lab-environment-b8kgh8.jpg", "http://www.stjhs.org/images/August-2015/iAsianMaleGaming_000054185694.jpg",
	# 		"https://www.ewa.org/sites/main/files/imagecache/medium/main-images/bigstock-hispanic-boy-in-science-class-73240339.jpg", "https://thumb9.shutterstock.com/display_pic_with_logo/499045/499045,1329286392,2/stock-photo-hispanic-scientist-holding-two-different-chemicals-in-beakers-95196859.jpg",
	# 		"https://n7.alamy.com/zooms/895c029d00ee49de839939e7ef1f0ae7/a-portrait-of-a-scientist-ajfakm.jpg", "https://www.bing.com/th?id=OIP._HFJOiC41BpTPFrPQOI6CgHaE7&pid=Api&rs=1&p=0",
	# 		"https://www.aps.org/careers/physicists/profiles/images/arodriguez-profile.jpg", "https://www.bing.com/th?id=OIP.FzWoEAVnpdUW0RISWhWqnQHaE8&pid=Api&rs=1&p=0",
	# 		"https://media.senscritique.com/media/000008232625/1200/Cherie_j_ai_retreci_les_gosses.jpg", "http://cdn.grid.fotosearch.com/BLD/BLD084/bld242586.jpg",
	# 		"https://thumbs.dreamstime.com/t/male-lab-researcher-technician-scientist-doctor-holding-test-tube-looking-concerned-isolated-white-48683624.jpg", "https://image.shutterstock.com/image-photo/adult-hispanic-scientist-doctor-man-260nw-1188758161.jpg",
	# 		"https://www.tutorsinjohannesburg.co.za/wp-content/uploads/2018/04/Science-Tutor-Smiling-in-Classroom.jpg", "http://coen.boisestate.edu/mbe/files/2014/06/Latino-College-Student.jpg",
	# 		"https://www.swarthmore.edu/sites/default/files/styles/main_page_image_floating/public/assets/images/news-events/Alex%201.jpg?itok=lWO_xWrv", "http://www.slate.com/content/dam/slate/articles/health_and_science/new_scientist/2012/07/120706_NEWSCIENTIST_SanalEdamarukuEX.jpg.CROP.promo-xlarge2.jpg",
	# 		"http://www.malegroomingacademy.com/wp-content/uploads/2018/03/2112-anilbhardwaj.jpg", "http://cdn.grid.fotosearch.com/BLD/BLD014/dm060707_018.jpg",
	# 		"http://media.gettyimages.com/photos/close-up-of-indian-male-scientist-wearing-respirator-picture-id72541184?s=612x612", "http://media.gettyimages.com/photos/indian-male-scientist-wearing-respirator-and-looking-at-beaker-picture-id72541180?s=170667a",
	# 		"https://www.thefamouspeople.com/profiles/images/homi-bhabha-4.jpg", "https://science.dodlive.mil/files/2014/07/140528-N-CM812-002_low-res.jpg",
	# 		"https://c8.alamy.com/comp/M51DWN/asianindian-male-scientist-or-doctor-or-science-student-experimenting-M51DWN.jpg", "https://c8.alamy.com/comp/M51D4D/asianindian-male-scientist-or-doctor-or-science-student-experimenting-M51D4D.jpg",
	# 		"http://media.gettyimages.com/photos/portrait-of-a-male-scientist-holding-a-conical-flask-picture-id56529745?s=170667a", "https://thumbs.dreamstime.com/z/male-scientist-doing-research-laboratory-asian-wearing-lab-coat-instruments-modern-45529913.jpg",
	# 		"https://thumbs.dreamstime.com/z/male-lab-researcher-technician-scientist-doctor-wearing-white-coat-isolated-white-48684248.jpg", "https://www.bing.com/th?id=OIP.Ge6Q0EQ0vBfU2jl_7mtJwgHaJ4&w=174&h=232&c=7&o=5&dpr=2&pid=1.7",
	# 		"https://i.pinimg.com/originals/df/4f/56/df4f56d30237e5a34f69d565a3d49f4d.jpg", "https://il9.picdn.net/shutterstock/videos/6065504/thumb/1.jpg",
	# 		"https://thumbs.dreamstime.com/z/male-scientist-working-lab-10640701.jpg", "https://thumbs.dreamstime.com/z/male-scientist-student-doing-research-young-using-chemical-fluid-isolated-white-45530621.jpg",
	# 		"https://c8.alamy.com/comp/KY1F8G/1950s-historical-picture-a-male-scientist-at-a-workbench-in-a-laboratory-KY1F8G.jpg", "https://thumbs.dreamstime.com/z/male-scientist-chemist-working-microscope-pharmacy-pharmaceutical-laboratory-47528427.jpg",
	# 		"http://media.gettyimages.com/photos/portrait-of-a-male-scientist-side-view-black-background-picture-id72617168", "https://www.bing.com/th?id=OIP.czGHrsSZZVwATWZo1umLGgHaFc&pid=Api&w=181&h=181&c=7&dpr=2",
	# 		"https://www.bing.com/th?id=OIP.lZsjc66t5eneqLFzUNESrgHaLH&w=174&h=262&c=7&o=5&dpr=2&pid=1.7", "https://p.motionelements.com/stock-video/business/me11456103-male-scientist-white-shirt-holds-abstract-ball-his-hands-belarus-4k-a0251.jpg",
	# 		"https://www.bing.com/th?id=OIP.RuDoOUZ3PHdyWhjSaysTcgHaE8&pid=Api&rs=1&p=0", "https://www.bing.com/th?id=OIP.Kk00LrU6_zM6efPiQ1NlYAHaFr&w=230&h=177&c=7&o=5&dpr=2&pid=1.7"]

	urls =[ "https://blogs.plos.org/speakingofmedicine/files/2014/03/Collin-300x279.jpg","http://www.useoul.edu/upload/news/LEESM.jpg",
			 "https://www.sciencemag.org/sites/default/files/styles/article_main_large/public/images/sc-karlbooksh-H.jpg?itok=qP0gvlaW","https://phys.org/newman/gfx/news/hires/2011/siteseeksfee.jpg",
			 "https://phys.org/newman/gfx/news/hires/2009/wheelchair_b_325.jpg","http://pvangels.com/news/images/201301/motor0915.jpg",
			 "http://dc-cdn.s3-ap-southeast-1.amazonaws.com/dc-Cover-cdiieck5n6hsm0ca6580jfcg63-20160606064813.Medi.jpeg","http://www.engineering.com/portals/0/BlogFiles/duerstock-bookletLO.jpg",
			 "https://cdn.technologyreview.com/i/images/2_2.jpg?sw=600&amp;cx=0&amp;cy=0&amp;cw=2000&amp;ch=1333","https://www.bing.com/th?id=OIP.3EXD1DDGyAQwjM1ZT8vHEgHaE9&w=266&h=175&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.BvcU4YZqwLBHUdXA0sqUvgHaHa&w=138&h=172&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.IzA-Y89OWlpGqYtHYrjwqwHaH2&w=130&h=172&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.LgJ-MleKZEarJ9Hk93PfFwHaKG&w=159&h=160&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.C9If84vLsmNEa66PUHftPgHaE8&w=229&h=160&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.C9If84vLsmNEa66PUHftPgHaE8&w=229&h=160&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.W_zS2d1BhlLTFnUIAi9FHgHaGN&w=227&h=183&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.7C6noN6l_Op_nnz5p_BOCAHaFa&w=259&h=183&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.yjCkFeqq9g75kRPI_8B4mwHaD4&w=300&h=157&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.MllYMKkO3b7-GoXKAYBRUgAAAA&w=146&h=160&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.sGyIlDaN7F_8Bn9zlYKf0QHaDt&w=300&h=150&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.fxRlfW6RJSVc15KdDHFuEQHaFa&w=209&h=160&c=7&o=5&dpr=2&pid=1.7","https://media.gettyimages.com/photos/disabled-medical-science-university-student-picture-id87922326",
			 "https://www.bing.com/th?id=OIP.UCdpytiFAzKnqWkX6ZflTwHaFj&w=176&h=160&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.hO4idlGa3u07tHuzFY8N2gHaEm&w=300&h=186&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.vaRocPS6AkwwIM5raiw_YQHaE8&w=300&h=194&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.2NCfai94K7wYrcjZya_xmAHaE8&w=299&h=200&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.6Ta-EtNIKdhi-ODkOJBqagHaE8&w=293&h=189&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.AKCf6tyaLITUmBa3jCGq1QHaEK&w=213&h=160&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.Ih64r5XIpMkq8fI2LQ2m8wAAAA&w=141&h=178&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.Oht7e5b5eL8DMHfe7HlDuQHaEK&w=300&h=168&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.KIhLG-7-FtGhXnByqgemswHaE7&w=271&h=160&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.3xHdJUFWfyZCULm0EvKt_AHaIf&w=159&h=178&c=7&o=5&dpr=2&pid=1.7",
			 "http://teacher.scholastic.com/activities/bhistory/inventors/images/drew.jpg","https://www.bing.com/th?id=OIP.ziYbgX7Droril1QdCOHCJwHaFS&w=263&h=184&c=7&o=5&dpr=2&pid=1.7",
			 "https://www.bing.com/th?id=OIP.w2EhHjn3RlEBrs1-EURl5gHaGZ&w=201&h=174&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.nUbHgKew19VXJwB36dGjjAHaE6&w=282&h=188&c=7&o=5&dpr=2&pid=1.7",
			 "http://emeagwali.com/gifs/African_June_1996.jpg","https://www.bing.com/th?id=OIP.kkvyTaXkb_-GhPHAu0wFmAHaF4&w=256&h=203&c=7&o=5&dpr=2&pid=1.7",
			 "http://emeagwali.com/photos/stock/african-american-scientist/emeagwali-dark-suit-science-museum-of-minnesota-saint-paul-june-1996.jpg","https://www.ancient-origins.net/sites/default/files/styles/large/public/korean-scientist_0.jpg?itok=OrmCoOrv",
			 "https://www.bing.com/th?id=OIP.wzbDgUCRdwnfGc2mhyKiiQHaJp&w=139&h=174&c=7&o=5&dpr=2&pid=1.7","https://www.bing.com/th?id=OIP.VQzZ5pynslIAmWOxJ-ZbeAHaKx&pid=Api&rs=1&p=0",
			 "http://blackmissouri.com/digest/wp-content/uploads/2008/02/gwc.jpg","https://www.famousafricanamericans.org/images/scientists/guion-s-bluford.jpg", "https://cdn.quotesgram.com/small/66/98/745529556-3415274_f260.jpg"
			]


	out_csv = "ID, filename, class, occupation, is_female, is_poc, human_annotated \n"
	# out_csv = [["ID", "filename", "class", "occupation", "is_female", "is_poc", "human_annotated"]]

	# with open("scientist.csv","a") as file1:
	# 	file1.write(out_csv) 
	# file1.close() 

	for p in urls: 
		local_name =  dest + "S_" +  "bing_" + str(ID) + ".jpg"
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
			out_r = str(ID) + ", " + local_name + ", 1, " + p + ", 0, -1, 0 \n" 
			
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
	# dest_s = "scientist/" 
	dest = "occupations/" 
	download_occ_im_URLS(dest)
	# download_sci_im_URLS(dest_s)


