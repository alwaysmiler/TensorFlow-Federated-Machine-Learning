# TensorFlow-Federated-Machine-Learning
Install 64 bit Python V3.8
Install tensorflow and other required package
Change the data directory where all data stored in line 13 in both TFClass.py and TFClass2.py datafolder = r'C:\Users\tingx\Downloads\FirstMeasurement\AllData'
Run Server.py-->it will output the weights update for both client 1 and client2 for each iteration
Run Client1.py, Client2.py-> it will output the accuracy for each client.
Currently the stop criteria has not be been set up. It will continue to run. Final accuracy will be approaching 1 eventually and weights converge (weights do not change at local client and the weights of two client are close to each other).
