
import os

# from utils.FloorplanToBlenderLib import *
# from . import utils as ut
from .utils.FloorplanToBlenderLib import *

import numpy as np
import cv2
from PIL import Image

import json
from datetime import datetime


class RBGFloorPlanOpenCV():
    
    # rbgjson = {
    #     "imagePath": "NameOfImage.png", 	#string, filename of the image
    #     "imageHeight": 0,			#integer, height in pixels
    #     "imageWidth": 0,
    #     "predictions": []
    # }
    
    
    def writeJSON(jsondict, outputname):
        jsonobj = json.dumps(jsondict, indent=4)

        with open(outputname, "w") as jout:
            jout.write(jsonobj)
    
    def getOuterShell(img_path, timestamp=None, outputsFolder=None):
        
        
        
        # Read floorplan image
        img = cv2.imread(img_path)

        # jsonDict = {
        #     "imagePath": os.path.basename(img_path), 	
	    #     "imageHeight": img.shape[0],
	    #     "imageWidth": img.shape[1],
        #     "predictions": [],
        #     # "resultImg": None
        # }
        
        # predictions = [
        #     {
        #         "confidence": 0.95,	
        #         "class": "entirefloor",
        #         "name": "",	
        #         "points": []
        #     }
        # ]

        # Create blank image
        height, width, channels = img.shape
        blank_image = np.zeros((height,width,3), np.uint8)

        # Grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        ogImgInfo = [
            img_path, #imgpath
            img.shape[0],    #y
            img.shape[1],    #x
        ]

        # detect outer Contours (simple floor or roof solution), paint them red on blank_image
        #detectOuterContours has been modified to return a json object
        # contour, img = detect.detectOuterContours(detect_img=gray, imgInfo=ogImgInfo, output_img=blank_image, color=(255,0,0))
        jsonDict, img = detect.detectOuterContours(detect_img=gray, imgInfo=ogImgInfo, output_img=blank_image, color=(255,0,0))
        
        print(f"----------jsonobj: {jsonDict}")
        
        outContourImgName = f"bvim/resultsImg/POLYFLOOR-{os.path.basename(img_path)}"

        # #cv2 write method
        # cv2.imwrite(outcontourImgName, img)
        contourPilImg = Image.fromarray(img)
        contourPilImg = contourPilImg.save(outContourImgName)
        return jsonDict

    def detectRooms(img_path, timestamp, outputsFolder):
        img = cv2.imread(img_path)

            # grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # create verts (points 3d), points to use in mesh creations
        verts = []
        # create faces for each plane, describe order to create mesh points
        faces = []

        # Height of waLL
        height = 0.999

        # Scale pixel value to 3d pos
        scale = 100

        gray = detect.wall_filter(gray)

        gray = ~gray

        rooms, colored_rooms = detect.find_rooms(gray.copy())

        gray_rooms =  cv2.cvtColor(colored_rooms,cv2.COLOR_BGR2GRAY)

        # get box positions for rooms
        boxes, gray_rooms = detect.detectPreciseBoxes(gray_rooms, gray_rooms)

        # display(Image.fromarray(colored_rooms))
        outRoomImgName = f"{outputsFolder}/{timestamp}-RoomOutput-{os.path.basename(img_path)}"
        roomPilImg = Image.fromarray(colored_rooms)
        roomPilImg = roomPilImg.save(outRoomImgName)

        #Create verts
        room_count = 0
        for box in boxes:
            verts.extend([transform.scale_point_to_vector(box, scale, height)])
            room_count+= 1

        # create faces
        for room in verts:
            count = 0
            temp = ()
            for pos in room:
                temp = temp + (count,)
                count += 1
            faces.append([(temp)])

        vertOutputs = {
            "verts": verts
        }

        jsonName = f"{outputsFolder}/{timestamp}-RoomVerts-{os.path.basename(img_path).split('.')[0]}.json"
        RBGFloorPlanOpenCV.writeJSON(vertOutputs, jsonName)
    
    def detectWalls(img_path, timestamp, outputsFolder):
        img = cv2.imread(img_path)

        # grayscale image
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

        # create wall image (filter out small objects from image)
        wall_img = detect.wall_filter(gray)

        # detect walls
        boxes, img = detect.detectPreciseBoxes(wall_img)

        outWallImgName = f"{outputsFolder}/{timestamp}-WallOutput-{os.path.basename(img_path)}"
        wallPilImg = Image.fromarray(wall_img)
        wallPilImg = wallPilImg.save(outWallImgName)
        

        # create verts (points 3d), points to use in mesh creations
        verts = []
        # create faces for each plane, describe order to create mesh points
        faces = []

        # Height of waLL
        wall_height = 1

        # Scale pixel value to 3d pos
        scale = 100

        # Convert boxes to verts and faces
        verts, faces, wall_amount = transform.create_nx4_verts_and_faces(boxes, wall_height, scale)

        # Create top walls verts
        verts = []
        for box in boxes:
            verts.extend([transform.scale_point_to_vector(box, scale, 0)])

        # create faces
        faces = []
        for room in verts:
            count = 0
            temp = ()
            for _ in room:
                temp = temp + (count,)
                count += 1
            faces.append([(temp)])

        vertOutputs = {
            "verts": verts
        }

        jsonName = f"{outputsFolder}/{timestamp}-WallVerts-{os.path.basename(img_path).split('.')[0]}.json"
        RBGFloorPlanOpenCV.writeJSON(vertOutputs, jsonName)