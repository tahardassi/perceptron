import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

centre1=np.array([3,3])
centre2=np.array([-3,-3])
sigma1=np.array([[4,0],[0,4]])
sigma2=np.array([[4,0],[0,4]])
taille1=200
taille2=200
cluster1=np.random.multivariate_normal(centre1,sigma1,taille1)
cluster2=np.random.multivariate_normal(centre2,sigma2,taille2)
etq1 = 1.0;
etq0 = 0.0;

plt.scatter([point[0] for point in cluster1], [point[1] for point in cluster1], color="pink")
plt.scatter([point[0] for point in cluster2], [point[1] for point in cluster2], color="blue")
plt.scatter(centre1[0], centre1[1], color="red")
plt.scatter(centre2[0], centre2[1], color="red")

plt.axis([-7, 7, -7, 7])

#Initialisation des poids w
W = np.random.multivariate_normal([10., 1.], sigma1,1)
W = W.tolist()[0]

print(W)
#W1= [1.50,	3.2] 
W1= [0.0,	0.0] 
B = np.random.rand(1)[0] + W[0]
print("b = ", B)

def droite(a, b, color,max = 10):
	X = [] 
	Y = []
	for x in range(-max, max):
		X.append(x)
		Y.append(a * x + b)
	plt.plot(X, Y, color = color)




#fonction de décision
def predicat(x, w, b):
	#activation = 1. * w[0]
	activation = b;
	for i in range(len(w)):
		activation += x[i] * w[i]
	if activation >= 0.0: return 1
	else: return 0

preds1 = [0]*taille1
preds2 = [0]*taille2

def precision(x1, x2, w, b):
	nb_ok = 0.0
	for i in range(taille1):
		pred1 = predicat(x1[i], w, b)
		preds1[i] = pred1
		if pred1 == etq1: nb_ok += 1
	for j in range(taille2):
		pred2 = predicat(x2[j], w, b)
		preds2[j] = pred2
		if pred2 == etq0: nb_ok += 1
	return nb_ok/float(taille1+taille2)

#pre = precision(cluster1, cluster2, W1)





def train(x1, x2, w, b, epis = 0.01, nb_iter = 50):
	current_pre = 0.0
	meilleure_pre = 0.0

	meilleure_W = [0]*len(w)
	meilleure_B = 0.0
	itera = 0;
	for iteration in range(nb_iter):
		current_pre = precision(x1, x2, w, b)
		if current_pre > meilleure_pre:
			meilleure_pre = current_pre
			meilleure_W[0] = w[0]
			meilleure_W[1] = w[1]
			meilleure_B = b
			itera = iteration

		print("itération %d precision : "%iteration, current_pre)
		print("itération %d W: "%iteration, w)


		#print("cluster1 :", preds1)
		#print("===========")
		#print("cluster2 :", preds2)
		

		if current_pre == 1.0: 
			break;

		for i in range(taille1):
			pred1 = predicat(x1[i], w, b)
			error = etq1 - pred1
			for j in range(len(w)):
				w[j] = w[j]+(epis*error*x1[i][j])
				b += error * epis
		j = 0
		for i2 in range(taille2):
			pred2 = predicat(x2[i2], w, b)
			error = etq0 - pred2
			for j in range(len(w)):
				w[j] = w[j]+(epis *error*x2[i2][j])
				b += error * epis
		
		if iteration % 10 == 0:
			#droite(-w[0]/w[1], -b/w[1], "red")
			plt.pause(0.05)
	if current_pre == 1.0:
		droite(-w[0]/w[1], -b/w[1],"green")
		plt.pause(0.05)
	else:
		droite(-meilleure_W[0]/meilleure_W[1], -meilleure_B/meilleure_W[1], "silver")
		droite(-w[0]/w[1], -b/w[1], "yellow")
		print("meilleure_pre = ", meilleure_pre)
		print("meilleure_W meilleure_B itera", meilleure_W,meilleure_B, itera)
		plt.pause(0.05)	


	
train(cluster1, cluster2, W, B, 0.02, 100)


plt.show()