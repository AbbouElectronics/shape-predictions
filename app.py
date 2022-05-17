from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
import requests
import configparser
import numpy as np
app = Flask(__name__)

dic = ['massif',  'rive','mince','barre', 'billette','dechet','bol','scrap_in_motion']
poids=[2255,1772,2846,2532,1960,1380,1962]
model = load_model('13_05_2022_8_classes_Transfer_learning_70_epochs')
model_massif = load_model('17052022_model_Weights_massif_TF_CNN_6Layers_loss0.0082_300_epochs_filtre_163264')
model_rive = load_model('07042022_model_Weights_rive_TF_CNN_6Layers_loss0.0030_200_epochs_filtre_163264128')
model_mince = load_model('04042022_model_Weights_mince_TF_CNN_10Layers_loss0.0110_200_epochs_filtre_163264128')
model_barre = load_model('04042022_model_Weights_barre_TF_CNN_10Layers_loss0.0169_200_epochs_filtre_163264128')
model_bol = load_model('04042022_model_Weights_bol_TF_CNN_10Layers_loss0.0289_200_epochs_filtre_163264128')
model_billette = load_model('08042022_model_Weights_billette_TF_CNN_6Layers_loss0.0039_200_epochs_filtre_163264')
model_dechet = load_model('28_03_2022_poids_model_100_epochs_dechet')


model.make_predict_function()

def predict_label(img_path):
	i = image.load_img(img_path, target_size=(224,224))
	i = image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	p = model.predict(i)
	y_classes=np.argmax(p)
	acc=np.round(np.max(softmax(p))*100,3)
	if dic[y_classes]=='massif':
		poid = model_massif.predict(i)
		poid=np.round(poid[0][0]*poids[0],1)
	elif dic[y_classes]=='rive':
		poid = model_rive.predict(i)
		poid=np.round(poid[0][0]*poids[1],1)
	elif dic[y_classes]=='mince':
		poid = model_mince.predict(i)
		poid=np.round(poid[0][0]*poids[2],1)
	elif dic[y_classes]=='barre':
		poid = model_barre.predict(i)
		poid=np.round(poid[0][0]*poids[3],1)
	elif dic[y_classes]=='billette':
		poid = model_billette.predict(i)
		poid=np.round(poid[0][0]*poids[4],1)
	elif dic[y_classes]=='dechet':
		poid = model_dechet.predict(i)
		poid=np.round(poid[0][0]*poids[5],1)
	elif dic[y_classes]=='bol':
		poid = model_bol.predict(i)
		poid=np.round(poid[0][0]*poids[6],1)
	else:
		poid='scrap_in_motion'

	return str(dic[y_classes]),poid,acc


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")


@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p,poid,acc = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path,accuracy=acc,poid=poid)

def softmax(x):
	f_x=np.exp(x)/np.sum(np.exp(x))
	return f_x
if __name__ =='__main__':
	#app.debug = True
	app.run(port=3000,debug = True)