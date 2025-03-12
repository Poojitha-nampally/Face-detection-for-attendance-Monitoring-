import tkinter as tk
import csv
import cv2
import os
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time

# Initialize the GUI window
window = tk.Tk()
window.title("Student Attendance System")
window.geometry('800x500')
window.configure(background='green')
window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)

def clear_name():
    std_name.delete(0, 'end')
    label4.configure(text="")

def clear_id():
    std_id.delete(0, 'end')
    label4.configure(text="")

def takeImage():
    name = std_name.get()
    Id = std_id.get()
    if name.isalpha() and Id.isdigit():
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.1, 4)
            for (x, y, w, h) in faces:
                sampleNum += 1
                cv2.imwrite(f"TrainingImages/{name}.{Id}.{sampleNum}.jpg", gray[y:y + h, x:x + w])
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                cv2.imshow("Capturing Image", img)
            if cv2.waitKey(100) & 0xFF == ord('q') or sampleNum > 50:
                break
        cam.release()
        cv2.destroyAllWindows()
        with open('studentDetails.csv', 'a+', newline='') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow([Id, name])
        label4.configure(text=f"Images saved for ID: {Id} Name: {name}")
    else:
        label4.configure(text="Enter valid name and numeric ID")

def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path) if not f.startswith('.')]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        try:
            pilImage = Image.open(imagePath).convert('L')
            imageNp = np.array(pilImage, 'uint8')
            Id = int(os.path.split(imagePath)[-1].split(".")[1])
            faces.append(imageNp)
            Ids.append(Id)
        except Exception as e:
            print(f"Skipped file: {imagePath}, Error: {str(e)}")
    return faces, Ids

def trainImage():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    faces, Ids = getImagesAndLabels("TrainingImages")
    recognizer.train(faces, np.array(Ids))
    recognizer.save("Trainner.yml")
    label4.configure(text="Images trained successfully!")

def trackImage():
    if not os.path.exists("Trainner.yml"):
        label4.configure(text="Trainner.yml not found! Train images first.")
        return

    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("Trainner.yml")
    faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    df = pd.read_csv("studentDetails.csv", names=['ID', 'NAME'])
    attendance = pd.DataFrame(columns=['ID', 'NAME', 'DATE', 'TIME'])

    cam = cv2.VideoCapture(0)
    start_time = time.time()

    while True:
        ret, img = cam.read()
        if not ret:
            label4.configure(text="Camera not accessible!")
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.1, 4)
        for (x, y, w, h) in faces:
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 60:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%d-%m-%Y')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                name = df.loc[df['ID'] == Id, 'NAME'].values[0] if Id in df['ID'].values else "Unknown"

                # Check if attendance for the ID and date is already logged
                if attendance[(attendance['ID'] == Id) & (attendance['DATE'] == date)].empty:
                    attendance.loc[len(attendance)] = [Id, name, date, timeStamp]
                    cv2.putText(img, f"{Id}-{name}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                else:
                    cv2.putText(img, f"{name} (Already Logged)", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                cv2.putText(img, "Unknown", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Tracking Image", img)
        if cv2.waitKey(1) & 0xFF == ord('q') or time.time() - start_time > 30:
            break

    cam.release()
    cv2.destroyAllWindows()

    if not attendance.empty:
        try:
            if not os.path.exists("AttendanceFile.csv"):
                attendance.to_csv("AttendanceFile.csv", index=False, header=True)
            else:
                # Append only new records
                existing_data = pd.read_csv("AttendanceFile.csv")
                attendance = pd.concat([existing_data, attendance]).drop_duplicates(subset=['ID', 'DATE'], keep='first')
                attendance.to_csv("AttendanceFile.csv", index=False, header=True)
            label4.configure(text="Attendance updated successfully!")
        except PermissionError:
            label4.configure(text="Error: AttendanceFile.csv is locked or read-only. Close it and try again.")
        except Exception as e:
            label4.configure(text=f"Unexpected error: {e}")
    else:
        label4.configure(text="No attendance recorded.")

# GUI Components
label1 = tk.Label(window, text="Name:", width=10, height=1, font=('Helvetica', 16), bg='green', fg='black')
label1.place(x=50, y=50)
std_name = tk.Entry(window, width=20, font=('Helvetica', 16))
std_name.place(x=200, y=50)

label2 = tk.Label(window, text="ID:", width=10, height=1, font=('Helvetica', 16), bg='green', fg='black')
label2.place(x=50, y=120)
std_id = tk.Entry(window, width=20, font=('Helvetica', 16))
std_id.place(x=200, y=120)

clearBtn1 = tk.Button(window, text="Clear Name", command=clear_name, font=('Helvetica', 12), bg="red", fg="white")
clearBtn1.place(x=500, y=50)

clearBtn2 = tk.Button(window, text="Clear ID", command=clear_id, font=('Helvetica', 12), bg="red", fg="white")
clearBtn2.place(x=500, y=120)

label3 = tk.Label(window, text="Notification:", width=15, font=('Helvetica', 16, 'bold'), bg='green', fg='red')
label3.place(x=50, y=200)
label4 = tk.Label(window, text="", width=55, height=4, font=('Helvetica', 12), bg='yellow', fg='black')
label4.place(x=50, y=250)

takeImg = tk.Button(window, text="Capture Image", command=takeImage, font=('Helvetica', 12), bg="yellow", fg="black")
takeImg.place(x=50, y=400)

trainImg = tk.Button(window, text="Train Images", command=trainImage, font=('Helvetica', 12), bg="yellow", fg="black")
trainImg.place(x=250, y=400)

trackImg = tk.Button(window, text="Track Images", command=trackImage, font=('Helvetica', 12), bg="yellow", fg="black")
trackImg.place(x=450, y=400)

window.mainloop()
