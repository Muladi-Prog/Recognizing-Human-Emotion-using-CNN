

import PySimpleGUI as sg
import re
isLog=0
username = ''
password = ''

#PROGRESS BAR
def progress_bar3():
    sg.theme('LightBlue2')
    layout = [[sg.Text('Welcome to FaceEmo',font=50,justification='center',size=(20,1))],
            [sg.ProgressBar(1000, orientation='h', size=(20, 20), key='progbar')]
            ]

    window = sg.Window('Working...', layout)
    for i in range(1000):
        event, values = window.read(timeout=1)
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            break
        window['progbar'].update_bar(i + 1)
    window.close()

    
def progress_bar():
    sg.theme('LightBlue2')
    layout = [[sg.Text('Creating your account...')],
            [sg.ProgressBar(1000, orientation='h', size=(20, 20), key='progbar')],
            [sg.Cancel()]]

    window = sg.Window('Working...', layout)
    for i in range(1000):
        event, values = window.read(timeout=1)
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            break
        window['progbar'].update_bar(i + 1)
    window.close()
def progress_bar2():
    sg.theme('LightBlue2')
    layout = [[sg.Text('Loading...')],
            [sg.ProgressBar(1000, orientation='h', size=(20, 20), key='progbar')],
            [sg.Cancel()]]

    window = sg.Window('Working...', layout)
    for i in range(1000):
        event, values = window.read(timeout=1)
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            break
        window['progbar'].update_bar(i + 1)
    window.close()

def create_account():
    global username, password
    sg.theme('LightBlue2')
    layout = [[sg.Text("           Sign Up", size =(50, 1), font=80, justification='center')],
             [sg.Text("E-mail", size =(15, 1),font=16), sg.InputText(key='-email-', font=16)],
             [sg.Text("Re-enter E-mail", size =(15, 1), font=16), sg.InputText(key='-remail-', font=16)],
             [sg.Text("Create Username", size =(15, 1), font=16), sg.InputText(key='-username-', font=16)],
             [sg.Text("Create Password", size =(15, 1), font=16), sg.InputText(key='-password-', font=16, password_char='*')],
             [sg.Button("Have an account already?")],
             [sg.Button("Submit"), sg.Button("Cancel")]]

    window = sg.Window("Sign Up", layout)
    ismatch=0
    regex = '^(\w|\.|\_|\-)+[@](\w|\_|\-|\.)+[.]\w{2,3}$'
    while True:
        event,values = window.read()
        
        if event == 'Cancel' or event == sg.WIN_CLOSED:
            break
        else:
            if event =="Have an account already?":
                window.close()
                login()
            if event == "Submit":
                password = values['-password-']
                username = values['-username-']
                email = values["-email-"]
                if(len(username)==0):
                    sg.popup_error("Username can't be empty", font=16)
                    continue
                if(len(password) == 0):
                    sg.popup_error("Password can't be empty", font=16)
                    continue
                if(len(email)==0):
                    sg.popup_error("Email can't be empty", font=16)
                    continue
                if re.search(regex, email):
                    print("Valid")
                else:
                    sg.popup_error("Email format is not right", font=16)
                    continue
                if(len(username) < 5):
                    sg.popup_error("Username is too short[Min 5 Char]", font=16)
                    continue
                if(len(password) <8):
                    sg.popup_error("Password should be have 8 characters", font=16)
                    continue
                
                ismatch = 0
                if values['-email-'] != values['-remail-']:
                    sg.popup_error("Email and Re-enter Email is not match", font=16)
                    continue
                elif values['-email-'] == values['-remail-']:
                    file1 = open("userdata.txt","a")
                    file1.writelines(username)
                    file1.write("\n")
                    file1.writelines(password)
                    file1.write("\n")
                    file1.close()
                    progress_bar()
                    ismatch=1
                    
                    break
    
    
    window.close()
    if(ismatch==1):
        login()
    


def logout(maxim,predicted,frames,emosi):
    print(maxim)
    print(predicted[maxim])
    print(float(frames/3))
    sg.theme("LightBlue2")
    layout = [
        [sg.Text("Result",size=(15,1),font=50)],
        [sg.Text("Emosi keseluruhan adalah: ",size=(15,1),font = 40),sg.Text(predicted[maxim],size=(6,1),font = 60),sg.Text("dalam kurun waktu",size=(15,1),font = 40),sg.Text(int(float(frames/5)),size=(3,1),font = 40) ,sg.Text("Detik",size=(5,1),font = 40) ],
        # emosi =np.array((angry,disgust,fear,happy,sad,surprise,neutral))
        [sg.Text("More Details:",size=(15,1),font = 80)],
        [sg.Text("Angry: ",size=(15,1),font = 40),sg.Text(float(emosi[0]/5),size=(15,1),font = 40)],
        [sg.Text("Disgust: ",size=(15,1),font = 40),sg.Text(float(emosi[1]/5),size=(15,1),font = 40)],
        [sg.Text("Fear: ",size=(15,1),font = 40),sg.Text(float(emosi[2]/5),size=(15,1),font = 40)],
        [sg.Text("Happy: ",size=(15,1),font = 40),sg.Text(float(emosi[3]/5),size=(15,1),font = 40)],
        [sg.Text("Sad: ",size=(15,1),font = 40),sg.Text(float(emosi[4]/5),size=(15,1),font = 40)],
        [sg.Text("Surprise: ",size=(15,1),font = 40),sg.Text(float(emosi[5]/5),size=(15,1),font = 40)],
        [sg.Text("Neutral: ",size=(15,1),font = 40),sg.Text(float(emosi[6]/5),size=(15,1),font = 40)]
    ,[sg.Button('Exit')]]
    window = sg.Window("Result",layout)
    
    progress_bar2()
    while True:
        
        event,values = window.read()
        if(event =="Exit" or event == sg.WIN_CLOSED):
            break
    window.close()

def start():
    sg.theme("LightBlue2")
    layout= [
        [sg.Text("FaceEmo",size=(15,1),font=50)],
        [sg.Button("Start Detect")],
        [sg.Button("Exit")]
    ]
    window = sg.Window("FaceEmo", layout)
    while True:

        event,values = window.read()
        if event == "Exit" or event == sg.WIN_CLOSED:
            break
        elif event == "Start Detect":
            detect()
            

def login():
    global username,password
    sg.theme("LightBlue2")
    layout = [[sg.Text("Log In", size =(15, 1), font=40)],
            [sg.Text("Username", size =(15, 1), font=16),sg.InputText(key='-usrnm-', font=16)],
            [sg.Text("Password", size =(15, 1), font=16),sg.InputText(key='-pwd-', password_char='*', font=16)],
            [sg.Button('Ok'),sg.Button('Cancel')]]

    window = sg.Window("Log In", layout)

    while True:
        event,values = window.read()
        if event == "Cancel" or event == sg.WIN_CLOSED:
            break
        else:
            if event == "Ok":
                file1 = open("userdata.txt","r")
                lines  = file1.readlines()
                flag1=0
                flag2=0
                count=1
                isUser = 0
                for line in lines:
                    # print(line.strip())
                    # print(count)
                    # print("===")
                    if values['-usrnm-'] == line.strip() and count % 2 != 0:
                        flag1=1
                        isUser = 1
                        count+=1
                        if(values['-pwd-'] == line.strip()):
                            flag2=1
                            break
                    count+=1
                print(flag1)
                print(flag2)
                if(flag1 == 1 and flag2 == 1):
                    sg.popup("Welcome!")
                    global isLog 
                    isLog =1
                    
                    break
                else:
                    sg.popup("Invalid login. Try again")

    window.close()
progress_bar3()
create_account()


def detect():

    from tensorflow.keras import models
    import cv2
    import numpy as np
    from mtcnn.mtcnn import MTCNN
    #Importing the model
    trained_model = models.load_model('trained_vggface.h5', compile=False)
    trained_model.summary()
    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
    # dictionary which assigns each label an emotion (alphabetical order)
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Sad", 5: "Surprise", 6: 'Neutral'}
    detector = MTCNN()
    # start the webcam feed
    cap = cv2.VideoCapture(0)
    black = np.zeros((96,96))
    #finding most emotion detected
    emosi =np.array((0,0,0,0,0,0,0))
    angry = 0
    disgust =0
    fear =0
    happy = 0 
    sad = 0
    surprise = 0
    neutral = 0
    frames = 0
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        frames = frames+1
        
        # detect faces in the image
        results = detector.detect_faces(frame)
        # extract the bounding box from the first face
        if len(results) == 1: #len(results)==1 if there is a face detected. len ==0 if no face is detected
            try:
                x1, y1, width, height = results[0]['box']
                x2, y2 = x1 + width, y1 + height
                # extract the face
                face = frame[y1:y2, x1:x2]
                #Draw a rectangle around the face
                cv2.rectangle(frame, (x1, y1), (x1+width, y1+height), (255, 0, 0), 2)
                # resize pixels to the model size
                cropped_img = cv2.resize(face, (96,96)) 
                cropped_img_expanded = np.expand_dims(cropped_img, axis=0)
                cropped_img_float = cropped_img_expanded.astype(float)
                prediction = trained_model.predict(cropped_img_float)
                print(prediction)
                maxindex = int(np.argmax(prediction))
                predicted_emotion = emotion_dict[maxindex]
                if(predicted_emotion =="Angry"):
                    angry+=1
                if(predicted_emotion =="Disgust"):
                    disgust+=1
                if(predicted_emotion =="Fear"):
                    fear+=1
                if(predicted_emotion =="Happy"):
                    happy+=1
                if(predicted_emotion =="Sad"):
                    sad+=1
                if(predicted_emotion =="Surprise"):
                    surprise+=1
                if(predicted_emotion =="Neutral"):
                    neutral+=1
                # frame=frame+1
                
                emosi =np.array((angry,disgust,fear,happy,sad,surprise,neutral))
                global maxim
                maxim = np.argmax(emosi)
                
                maxemosi = emotion_dict[maxim]
                cv2.putText(frame, emotion_dict[maxindex], (x1+20, y1-60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
                
                cv2.putText(frame, "Press'q' to stop", (x1-240, y1-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            except:
                pass
        cv2.imshow('Video',frame)
        try:
            cv2.putText(cropped_img, maxemosi, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("frame", cropped_img)
        except:
            cv2.imshow("frame", black)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    
    # while(True):
    #     cv2.putText(white, maxemosi, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
    #     cv2.imshow("Result",white)
    cap.release()
    cv2.destroyAllWindows()
    logout(maxim,emotion_dict,frames,emosi)

if isLog==1:
    start()