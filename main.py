from fastapi import FastAPI, UploadFile,File,HTTPException,Form,Query
from fastapi.middleware.cors import CORSMiddleware
import requests
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
from typing import Optional
dotenv.load_dotenv()

app = FastAPI()
# cors

app.add_middleware(
    CORSMiddleware,
    allow_origins=[f'{os.getenv("CORS_ORIGIN")}'],
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

# uploading to s3-bucket
async def uploadToCloud(image:UploadFile=File()):
    try:
        contents = await image.read()
        s3.upload_fileobj(
            Fileobj=io.BytesIO(contents),
            Bucket=BUCKET_NAME,
            Key=f"trainix/body-images/{image.filename}",
            ExtraArgs={"ContentType":image.content_type}    
        )
        url = f"https://{BUCKET_NAME}.s3.eu-north-1.amazonaws.com/trainix/body-images/{image.filename}"
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
        "shoulderToWaistRatio":round(shoulderToWaistRatio,2),
        "waistToHipRatio":round(waistToHipRatio,2),
        "shoulderAsymmetricLine":shoulderAsymmetricLine,
        "shoulderAngle":shoulderAngle,
        "bodyFatPercent":round(bodyFatPercent,2),
        "muscleMass":round(muscleMass,2),
        "leanBodyMass":round(leanBodyMass,2),
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

#class to describe request-body object
class UserInfo(BaseModel):
    weight: float
    height: float
    targetWeight: float
    primaryFitnessGoal: str
    fitnessLevel: str
    gender:str
    waistToHipRatio: Optional[float]=None
    shoulderToWaistRatio: Optional[float]=None
    bodyFatPercent:Optional[float]=None
    muscleMass:Optional[float]=None
    leanBodyMass:Optional[float]=None
 
# api for analyzing photo and creating fitness plan
@app.post("/api/photo-analyze")
async def func(image:UploadFile = File(),userInfo=Form()):
    # if not jpeg or png -return error 400
    if image.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400,detail="Wrong format")
    # uploading to s3-bucket 
    content = await uploadToCloud(image)
    #from json to dict and unpacking to object
    data = json.loads(userInfo)
    userInfo = UserInfo(**data)
    # getting counted metrics from the image witth mediapipe
    metrics = getMetrics(url=content["url"],poseHeight=userInfo.height,gender=userInfo.gender,poseWeight=userInfo.weight)
    try:
        # metrics to return
        metrics = {
      "height": userInfo.height,
      "weight": userInfo.weight,
      "waistToHipRatio": metrics["waistToHipRatio"],
      "shoulderToWaistRatio": metrics["shoulderToWaistRatio"],
      "bodyFatPercent": metrics["bodyFatPercent"],
      "muscleMass": metrics["muscleMass"],
      "leanBodyMass": metrics["leanBodyMass"],
    }

        return {"AIreport": metrics,"imageUrl":content["url"]}
    except Exception as error:
        raise HTTPException(status_code=500,detail=str(error))
# api for pinging server 
@app.get("/api/ping")
async def func():
    return {"ping":"hello world!"}
    
    
    
    
#api for creating fitness plan 
@app.post("/api/fitnessPlan") 
async def func(userInfo:UserInfo,dayNumber: int = Query(..., description="Number of the day to generate")):
    try:
        # prompt to chatgpt
        prompt =  f"""
You are a professional certified fitness coach.

Your task is to generate JSON fitness plans depending on the day number.

STRICT RULES:
- If dayNumber == 1  generate the FULL JSON object with ALL fields listed below.
- If dayNumber > 1  generate ONLY the `day` field (do NOT include briefAnalysis, advices, weekTitles, etc.).

User data:
- Height: {userInfo.height} cm
- Weight: {userInfo.weight} kg
- Waist to Hip Ratio: {userInfo.waistToHipRatio}
- Shoulder to Waist Ratio: {userInfo.shoulderToWaistRatio}
- Target Weight: {userInfo.targetWeight} kg
- Fitness Level: {userInfo.fitnessLevel}
- Primary Fitness Goal: {userInfo.primaryFitnessGoal}
- BodyFat Percent: {userInfo.bodyFatPercent}
- Muscle Mass: {userInfo.muscleMass}
- Lean Body Mass: {userInfo.leanBodyMass}

Day number: {dayNumber}

---

### JSON FORMAT

If dayNumber == 1  use this format:

{{
  "briefAnalysis": {{
    "targetWeight": number,
    "fitnessLevel": string,
    "primaryFitnessGoal": string
  }},
  "advices": {{
    "nutrition": "string (at least 5 sentences)",
    "hydration": "string (at least 5 sentences)",
    "recovery": "string (at least 5 sentences)",
    "progress": "string (at least 5 sentences)"
  }},
  "week1Title": "string",
  "week2Title": "string",
  "week3Title": "string",
  "week4Title": "string",
  "day": {{
      "dayNumber": {dayNumber},
    "day": "Upper Body Focus" | "Lower Body Focus" | "Rest Day / Active Recovery" | "Full Body & Core",
    "calories": number,
    "status": "Pending",
    "exercises": [
      {{
        "imageUrl": "string",
        "status": "incompleted",
        "calories": number,
        "title": "string",
        "repeats": number | null,
        "time": number | null,  # time must be in seconds
        "instruction": "string",
        "advices": "string"
      }}
    ]
  }}
}}

If dayNumber > 1  use this format ONLY:
# other info will be added in the future
{{
  "day": {{
    "day": "Upper Body Focus" | "Lower Body Focus" | "Rest Day / Active Recovery" | "Full Body & Core", --- without exercicies choose specific type of exercises that will be added later
    "dayNumber": {dayNumber},
    "status": "Pending",
    "date": Date,
  }}
}}

---

Additional strict rules:
- For dayNumber > 1, choose the type of day properly from these options (as a detail 28 day plan according to metrics):
  "Upper Body Focus", "Lower Body Focus", "Rest Day / Active Recovery", "Full Body & Core".
- Avoid repeating the same type three days in a row.
- Return ONLY valid JSON. No explanations, no comments, no extra text.
"""  
        # configuration for gpt requests   
        completion= client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are fitness coach"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return {"AIreport": completion.choices[0].message.content}
    except Exception as error:
        raise HTTPException(status_code=500,detail=str(error))


class FitnessDayBody(BaseModel):
    dayNumber:int
    day:str
    date:str

#api for filling day container
@app.post("/api/fitnessPlan/day") 
async def func(userInfo:UserInfo,day:FitnessDayBody):
    try:
        # prompt to chatgpt
        prompt =  f"""
You are a professional certified fitness coach.

Your task is to generate JSON fitness plans depending on the day number.

User data:
- Height: {userInfo.height} cm
- Weight: {userInfo.weight} kg
- Waist to Hip Ratio: {userInfo.waistToHipRatio}
- Shoulder to Waist Ratio: {userInfo.shoulderToWaistRatio}
- Target Weight: {userInfo.targetWeight} kg
- Fitness Level: {userInfo.fitnessLevel}
- Primary Fitness Goal: {userInfo.primaryFitnessGoal}
- BodyFat Percent: {userInfo.bodyFatPercent}
- Muscle Mass: {userInfo.muscleMass}
- Lean Body Mass: {userInfo.leanBodyMass}

Day number: {day.dayNumber}

---

### JSON FORMAT

"day": {{
    "day": {day.day},
    "dayNumber": {day.dayNumber},
    "calories": number,
    "status": "Pending",
    "date":{day.date}
    "exercises": [
      {{
        "imageUrl": "string",
        "status": "incompleted",
        "calories": number,
        "title": "string",
        "repeats": number | null,
        "time": number | null,  # time must be in seconds
        "instruction": "string",
        "advices": "string"
      }}
    ]
  }}


  - For exercises: if `repeats` is not null  `time` must be null, and vice versa.
- `calories` inside `day` MUST equal the sum of all exercises' calories.
- day nad dayNymber field must be static dont create a new one leave these values for them
---


"""  
        # configuration for gpt requests   
        completion= client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are fitness coach"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return {"AIreport": completion.choices[0].message.content}
    except Exception as error:
        raise HTTPException(status_code=500,detail=str(error))

# api for creating nutrition plan 
@app.post("/api/nutrition")    
async def func(userInfo:UserInfo,dayNumber: int = Query(..., description="Number of the day to generate")):
   
    try:
         # prompt to chatgpt
        prompt = f"""
        You are a professional certified nutritionist.

        Generate a nutrition plan for **one day only**.

        User data:
        - Height: {userInfo.height} cm
        - Weight: {userInfo.weight} kg
        - Waist to Hip Ratio: {userInfo.waistToHipRatio}
        - Shoulder to Waist Ratio: {userInfo.shoulderToWaistRatio}
        - Target Weight: {userInfo.targetWeight} kg
        - Fitness Level: {userInfo.fitnessLevel}
        - Primary Fitness Goal: {userInfo.primaryFitnessGoal}
        - BodyFat Percent: {userInfo.bodyFatPercent}
        - Muscle Mass: {userInfo.muscleMass}
        - Lean Body Mass: {userInfo.leanBodyMass}

        Day number: {dayNumber}

        Format JSON as:
        {{
        "dayNumber": {dayNumber},
        "date": "YYYY-MM-DD",
        "dailyGoals": {{
            "calories": {{ "current": 0, "target": number }},
            "protein": {{ "current": 0, "target": number }},
            "carbs": {{ "current": 0, "target": number }},
            "fats": {{ "current": 0, "target": number }}
        }},
        
        ,
        "meals": [
            {{
            "imageUrl":"string",
            "foodIntake":"Snack"|"Lunch"|"Breakfast"|"Dinner"
            "mealTitle": "string,
            "time": "HH:MM",
            "description": "string",
            "ingredients": ["string", "string", ...],
            "preparation": "string",
            "mealCalories": number,
            "mealProtein": number,
            "mealCarbs": number,
            "status":"pending",
            "mealFats": number
            }}
        ]
        }}
        waterIntake : {{
            current: number,
            target: number
        }}
        
        Return ONLY valid JSON, parseable.IN the meal title must be only name of the dish without words breakfast - etc.Sum of all calories of the meals must be the value of dailyGoals.calories and so with fats protein carbs
        """
 
        # configuration for gpt requests 
        completion= client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "professional  certified nutritionist"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7
        )
        return {"AIreport": completion.choices[0].message.content}
    except Exception as error:
        raise HTTPException(status_code=500,detail=str(error))
