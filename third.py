from common.DeepLab import *
from second import *

from sklearn.cluster import KMeans

#https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
#https://www.w3schools.com/python/numpy_array_shape.asp
#labels_ndarray of shape (n_samples,)
#cluster_centers_ndarray of shape (n_clusters, n_features)
def runKMeans(deepFeats,afterPCA):
	kM = KMeans(n_clusters=2, random_state = 0)
	kM.fit(afterPCA)
	print("Labels :",kM.labels_.shape)
	print("Cluster Centers :",kM.cluster_centers_.shape)
	return kM.labels_

def run_vis(deepFeats,labels):
	v = labels.reshape(deepFeats.shape[0],deepFeats.shape[1])
	#print(smt)
	plt.imshow(v)
	plt.show()

def main():
	L = sys.argv[1:]
	if not L:
		print("Run : python first.py example1.jpg example2.jpg ...")
		return
	MODEL = getModel("mobilenetv2_coco_voctrainaug","concat_projection/Conv2D:0")
	for i in L:
		if os.path.isfile(i):
			deepFeats = getFeature(i,MODEL)[1]
			afterPCA = perform_pca(deepFeats,8)
			run_vis(deepFeats,runKMeans(deepFeats,afterPCA))

if __name__ == "__main__":
	main()