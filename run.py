import Train_NP as P
#import Train_3way as P
#src = 'data/'
import Test as T



	# freeze_support()
def main():
	# src = '..\\..\\data_split.s3\\'
	#src = '../../data_split.s3/'
	src = '../data_split.ml_fp.3way/'


	dst = 'result.ml_fp.3way_NP/'
  #  dst = 'result.s3_2/'
	#gpuid = [0,1]
	gpuid = [3]
	L = P.learn(src, dst, gpuid)
	L.run()

	#.Test()

if __name__ == '__main__':
	main()
