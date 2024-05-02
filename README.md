1.	Pre-Processing 
from pandas import read_csv from numpy import set_printoptions from sklearn import preprocessing 
 
dataframe = read_csv("ftp://10.10.211.61/IOT KAB/pima-indians-diabetes.csv") array = dataframe.values 
datascaling = preprocessing.MinMaxScaler(feature_range=(0, 1)) datascaled = datascaling.fit_transform(array) set_printoptions(precision=2) 
print("\nScaled data:\n", datascaled[0:3]) print(dataframe.head(3)) 
 
 
 
 
 
 
 
2.	Normalization 
from pandas import read_csv from numpy import set_printoptions from sklearn import preprocessing 
from sklearn.preprocessing import Normalizer 
 
dataframe = read_csv("ftp://10.10.211.61/IOT KAB/pima-indians-diabetes.csv") array = dataframe.values datanormal = Normalizer(norm="l2") 
datascaling = preprocessing.MinMaxScaler(feature_range=(0, 1)) datanormed = datanormal.fit_transform(array) set_printoptions(precision=2) 
print("\nScaled data:\n", datanormed[0:3]) print(dataframe.head(3)) 
 
3.	Binarization 
from pandas import read_csv from numpy import set_printoptions from sklearn import preprocessing 
from sklearn.preprocessing import Binarizer 
 
dataframe = read_csv("ftp://10.10.211.61/IOT KAB/pima-indians-diabetes.csv") array = dataframe.values databin = Binarizer(threshold=0.6) databined = databin.fit_transform(array) set_printoptions(precision=2) 
print("\nScaled data:\n", databined[0:3]) print(dataframe.head(3)) 
 
4.	Standardization 
from pandas import read_csv from numpy import set_printoptions from sklearn import preprocessing 
from sklearn.preprocessing import StandardScaler 
 dataframe = read_csv("ftp://10.10.211.61/IOT KAB/pima-indians-diabetes.csv") array = dataframe.values 
datast = StandardScaler().fit(array) datascaled = datast.fit_transform(array) set_printoptions(precision=2) 
print("\nScaled data:\n", datascaled[0:3]) print(dataframe.head(3)) 
 
----------------------------------------------------------------------------------------------------------------------
import pandas as pd import plotly.express as px import dash from dash import dcc from dash import html 
from dash.dependencies import Output,Input df = pd.read_csv("C:/Users/IOT/Desktop/vgsales.csv") print(df.Genre.unique()) print(df.Genre.nunique()) 
fig_pie = px.pie(df,names = 'Genre',values = 'NA_Sales') 
#fig_pie.show() 
fig_bar = px.bar(df,x = 'Genre',y = 'NA_Sales') 
#fig_bar.show() 
fig_hist = px.histogram(df,x = 'Genre',y = 'NA_Sales') 
#fig_hist.show() 
 
################################################ #PART2 
################################################ 
 app=dash.Dash(__name__) app.layout=html.Div([     html.H1("Dashboard"), 
    dcc.Dropdown(id='Genre',options = [{'label':x,'value':x}     for x in sorted(df.Genre.unique())],     value='Action'), 
    dcc.Graph(id='newgraph', figure=()) 
 
]) 
@app.callback( 
    Output(component_id='newgraph', component_property='figure'), 
    Input(component_id='Genre', component_property='value') 
)  def intgraph(val_genre): 
    dff=df[df.Genre==val_genre]     fig=px.bar(dff,x='Year',y='JP_Sales')     return fig 
 if __name__ == "__main__": 
    app.run_server() 
 
-----------------------------------------------------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
np.random.seed(42)
x = 2*np.random.rand(100,1)
y = 4+3*x+np.random.randn(100,1)

x_train,x_test,y_train,y_test = train_test_split(x,y, test_size = 0.2, random_state = 42)

model = LinearRegression().fit(x_train,y_train)

plt.scatter(x_train, y_train, color = 'blue', label = 'train')
plt.scatter(x_test, y_test, color = 'red', label = 'test')
plt.scatter(x_test, model.predict(x_test), color = 'black', label = 'predict')
plt.legend()
plt.show()

----------------------------------------------------------------------------------------------------------------------------

import cv2
img = cv2.imread('C:/Users/TOLIPTSET/OneDrive - CubeV3/Desktop/To Delete Python Data/deadmau5.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow('Original Image', img)
cv2.imshow('Another Image', img_rgb)
cv2.imshow('blacky', img_gray)
cascade = cv2.CascadeClassifier("C:/Users/TOLIPTSET/OneDrive - CubeV3/Desktop/To Delete Python Data/haarcascade_frontalface_default.xml")
faces = cascade.detectMultiScale(image=img, scaleFactor=1.1, minNeighbors=2)
for x,y,w,h in faces:
    img_green = cv2.rectangle(img, (x,y),(x+h,y+w),(0,255,0),2)
    img_edge = cv2.Canny(img,100,50)
    img_blur = cv2.medianBlur(img, 11)
    crop = img[100:200, 300:400]
    cv2.imshow('border', img_green)
    cv2.imshow('edge', img_edge)
    cv2.imshow('blur', img_blur)
    cv2.imshow('crop', crop)
cv2.waitKey(0)

---------------------------------------------------------------------------------------------------------------------------
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("C:/Users/TOLIPTSET/OneDrive - CubeV3/Desktop/To Delete Python Data/creditcard.csv")

normal = data[data['Class'] == 0]
fraud = data[data['Class'] == 1]

plt.bar(['normal','fraud'], [len(normal),len(fraud)])
plt.xlabel('types')
plt.ylabel('amount')

f, (ax1, ax2) = plt.subplots(2,1,sharex=True)
ax1.hist(normal['Amount'], bins =50, label = 'normal')
ax2.hist(fraud['Amount'], bins = 50, label = 'fraud')
plt.xlim(0,2000)
plt.yscale("log")
plt.legend()

f, (ax1, ax2) = plt.subplots(2,1, sharex = True)
ax1.scatter(normal['Time'], normal['Amount'],label = 'normal')
ax2.scatter(fraud['Time'], fraud['Amount'], label = 'fraud')
plt.legend()

ax1.set_title('lol')
plt.show()
--------------------------------------------------------------------------------------------------------------------------

import kivy
from kivy.app import App
from kivy.uix.image import Image
from kivy.uix.label import Label
from kivy.uix.textinput import TextInput
from kivy.uix.gridlayout import GridLayout
from kivy.uix.button import Button


class childApp(GridLayout):
 def __init__(self, **kwargs):
  super(childApp,self).__init__()

  self.cols = 2

  self.add_widget(Label(text = "name"))
  self.s_name = TextInput()
  self.add_widget(self.s_name)

  self.add_widget(Label(text="Roll lol"))
  self.s_roll = TextInput()
  self.add_widget(self.s_roll)

  self.add_widget(Label(text = "Gender"))
  self.s_gender = TextInput()
  self.add_widget(self.s_gender)

  self.press = Button(text = "Click")
  self.press.bind(on_press = self.click)
  self.add_widget(self.press)


 def click(self):
  print("Name "+self.s_name.text)
  print("Roll No "+self.s_roll.text)
  print("Gender "+self.s_gender.text)

class parentApp(App):
 def build(self):
  return childApp()

if __name__ =='__main__':
 app = parentApp()
 app.run()

 ---------------------------------------------------------------------------------------------------------------------------
 import sys
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QDialog, QApplication
from PyQt5.uic import loadUi

class Login(QDialog):
    def __init__(self):
        super(Login, self).__init__()
        loadUi("login.ui", self)
        self.signin.clicked.connect(self.loginfunction)
        self.password.setEchoMode(QtWidgets.QLineEdit.Password)

    def loginfunction(self):
        username = self.username.text()
        password = self.password.text()
        print(username, password)

app = QApplication(sys.argv)
mainwindow = Login()
mainwindow.show()
app.exec()

----------------------------------------------------------------------------------------------------------------------------
