from fastapi import FastAPI, UploadFile,File,HTTPException,Form
from fastapi.middleware.cors import CORSMiddleware
import requests
from fastapi.responses import JSONResponse
import json
import boto3 
import os
import io
import dotenv
import mediapipe as mp
import cv2
import math
from pydantic import BaseModel
from openai import OpenAI
import numpy as np
dotenv.load_dotenv()

app = FastAPI()
# cors

app.add_middleware(
    CORSMiddleware,
    allow_origins=['http://localhost:3000'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# uploading into s3
s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION"),
    )
BUCKET_NAME = os.getenv("AWS_S3_BUCKET_NAME")
async def uploadToCloud(image:UploadFile=File()):
    try:
        contents = await image.read()
        s3.upload_fileobj(
            Fileobj=io.BytesIO(contents),
            Bucket=BUCKET_NAME,
            Key=image.filename,
            ExtraArgs={"ContentType":image.content_type}    
        )
        url = f"https://{BUCKET_NAME}.s3.amazonaws.com/{image.filename}"
        return {
            "bytes": len(contents),
            "filename": image.filename,
            "url": url
        }
    except Exception as err:
        raise HTTPException(status_code=500, detail=str(err))
# openai integration
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# metrics
def euclidean(p1, p2):
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def calculateMetrics(height,width,landmarks,poseHeight,poseWeight,gender):
    leftShoulder = (landmarks[11].x * width, landmarks[11].y * height)
    leftWaist = ((landmarks[11].x * width +landmarks[23].x * width)/2,(landmarks[11].y * height +landmarks[23].y * height)/2)
    rightWaist = ((landmarks[12].x * width +landmarks[24].x * width)/2,(landmarks[12].y * height +landmarks[24].y * height)/2)
    rightShoulder = (landmarks[12].x * width, landmarks[12].y * height)
    leftHip = (landmarks[23].x * width, landmarks[23].y * height)
    rightHip = (landmarks[24].x * width, landmarks[24].y * height)
    hipWidth = euclidean(leftHip,rightHip)
    waistWidth = euclidean(leftWaist,rightWaist)
    shoulderWidth = euclidean(leftShoulder,rightShoulder)
    neckCircumference =shoulderWidth * 2 * 1.6
    waistCircumference =hipWidth * 1.4
    hipCircumference = waistCircumference * 1.1; 
    shoulderToWaistRatio = shoulderWidth/waistWidth
    waistToHipRatio = waistWidth/hipWidth
    shoulderAsymmetricLine = (leftShoulder[0] - rightShoulder[0], leftShoulder[1] - rightShoulder[1])
    if gender =="Male":
        val = waistCircumference - neckCircumference
        if val <= 0:
            bodyFatPercent = 0; 
        else :
            bodyFatPercent = 86.010 * math.log10(val) - 70.041 * math.log10(poseHeight) + 36.76
    else :
        val = waistCircumference + hipCircumference - neckCircumference
        if val <= 0:
            bodyFatPercent = 0
        else :
            bodyFatPercent = 163.205 * math.log10(val) - 97.684 * math.log10(poseHeight) - 78.387
    leanBodyMass = poseWeight * (1 - bodyFatPercent / 100)
    muscleMass = leanBodyMass * 0.55
  
  
    dx = rightShoulder[0] - leftShoulder[0]
    dy = rightShoulder[1] - leftShoulder[1]
    angleRad = math.atan2(dy, dx)
    shoulderAngle = math.degrees(angleRad) 
  
        
    return {
        "shoulderToWaistRatio":shoulderToWaistRatio.toFixed(2),
        "waistToHipRatio":waistToHipRatio.toFixed(2),
        "shoulderAsymmetricLine":shoulderAsymmetricLine,
        "shoulderAngle":shoulderAngle,
        "bodyFatPercent":bodyFatPercent.toFixed(2),
        "muscleMass":muscleMass.toFixed(2),
        "leanBodyMass":leanBodyMass.toFixed(2),
    }

def getMetrics(url:str,poseHeight,gender,poseWeight):
    mp_pose = mp.solutions.pose
    Pose = mp_pose.Pose(static_image_mode=True)
    res = requests.get(url)
    imageArray = np.asarray(bytearray(res.content),dtype=np.uint8)
    image = cv2.imdecode(imageArray, cv2.IMREAD_COLOR)
    RGBImage= cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    results = Pose.process(RGBImage)
    h,w,_=image.shape
    return calculateMetrics(height=h,width=w,landmarks=results.pose_landmarks.landmark,poseHeight=poseHeight,gender=gender,poseWeight=poseWeight)

# api
class UserInfo(BaseModel):
    weight: int
    height: int
    targetWeight: int
    primaryFitnessGoal: str
    fitnessLevel: str
    gender:str
@app.post("/api")
async def func(image:UploadFile = File(),userInfo:str=Form()):
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400,detail="Wrong format")
    content = await uploadToCloud(image)
    data = json.loads(userInfo)
    userInfo = UserInfo(**data)
    metrics = getMetrics(url=content["url"],poseHeight=userInfo.height,gender=userInfo.gender,poseWeight=userInfo.weight)
    try:
        prompt = f"""
You are a professional fitness coach.

User data and body metrics:
- Height: {userInfo.height} cm
- Weight: {userInfo.weight} kg
- Waist to Hip Ratio: {metrics["waistToHipRatio"]}
- Shoulder to Waist Ratio: {metrics["shoulderToWaistRatio"]}
- Shoulder Asymmetric Line: {metrics["shoulderAsymmetricLine"]}
- Shoulder Angle: {metrics["shoulderAngle"]}
- Target Weight: {userInfo.targetWeight} kg
- Fitness Level: {userInfo.fitnessLevel}
- Primary Fitness Goal: {userInfo.primaryFitnessGoal}
-bodyFat Percent: {metrics["bodyFatPercent"]}
-muscle Mass: {metrics["muscleMass"]}
-lean body Mass: {metrics["leanBodyMass"]}
Please create a detailed 4-week fitness plan for the user.

The plan should be structured as JSON with the following format:

{{
  "briefAnalysis": {{
    "currentMetrics": {{
      "height": "number",
      "weight": "number",
      "waistToHipRatio": number,
      "shoulderToWaistRatio": number,
      "bodyFatPercent": number,
      "muscleMass": number,
      "leanBodyMass": number,
    }},
    "targetWeight": "number",
    "fitnessLevel": "string",
    "primaryFitnessGoal": "string"
  }},
  "plan":{{week1Title:"Muscle Gain & Endurance",week2Title:...,week3Title:...,week4Title:...,   days: [
      {{ "day": "string","calories":"number","status":"incompleted", "exercises": [{{ "title": "string","repeats": number|null, "time": number|null }}] }} -// time must be in seconds 
    {{...}},
    {{...}},
    {{...}},
 
    
   ]}}

  ,
  "advices": {{
    "nutrition": "string",
    "hydration": "string",
    "recovery": "string",
    "progress": "string"
  }}
}}

Do not include any explanations, only return the JSON object.Must be n items in plan where n -days of current month and item - day"""  
 
        completion= client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are fitness coach"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return {"AIreport": completion.choices[0].message.content,"imageUrl":content["url"]}
    except Exception as error:
        raise HTTPException(status_code=500,detail=str(error))
