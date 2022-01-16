import cv2
import dlib
import numpy as np
import bpy
import time
import math
from imutils import face_utils
from bpy.props import FloatProperty
from scipy.spatial import distance as dist

ARMATURE = "Armature"
EYEBONE = "eyecontrol"
HEADBONE = "headcontrol"
ACCBONE = "fangcontrol"
MODEL = "body"
EYES = "eyes"
BLINKSK = "blink"
JAWSK = "jaw_open"
togglecamwin = False

class FaceTrackerAddonPanel(bpy.types.Panel):
    """Creates a Panel in the Object properties window"""
    bl_label = "Face Tracker Addon"
    bl_idname = "FACE_TRK_PT_panel"
    bl_space_type = 'VIEW_3D'
    bl_region_type = 'UI'
    bl_category = "FaceTracker"

    def draw(self, context):
        layout = self.layout

        row = layout.row()
        row.label(text="Begin tracking face", icon='WORLD_DATA')

        row = layout.row()
        row.operator("wm.facetrackeraddon")
        
        row = layout.row()
        row.label(text = "Toggle alternate eyes")
        row.operator("wm.toggleeye")
        
        row = layout.row()
        row.label(text = "Press Escape or Right Click to Stop Tracking Face")
        
        row = layout.row()
        row.operator("wm.showcam")
        
        row = layout.row()
        row.label(text = "Reset positions")
        row.operator("wm.facetrackeraddonreset")    

class FaceTrackerReset(bpy.types.Operator):
    bl_idname = "wm.facetrackeraddonreset"
    bl_label = "Face Tracking Reset"

    def execute(self, context):
        armature = bpy.context.scene.objects.get(ARMATURE)
        eyebone = armature.pose.bones.get(EYEBONE)
        fangbone = armature.pose.bones.get(ACCBONE)
        eyebone.location = (0, 0, 0)
        headbone = armature.pose.bones.get(HEADBONE)
        headbone.rotation_euler = (0, 0, 0)
        fangbone.rotation_euler = (0, 0, 0)
        body_shapekeys = bpy.context.scene.objects.get(MODEL).data.shape_keys
        body_shapekeys.key_blocks[JAWSK].value = 0
        body_shapekeys.key_blocks[BLINKSK].value = 0    
        self.report({'INFO'}, "Reset positions")    
        return {'FINISHED'}

class DisplayCameraWindow(bpy.types.Operator):
    bl_idname = "wm.showcam"
    bl_label = "Display camera window"

    def execute(self, context):
        global togglecamwin
        togglecamwin = not togglecamwin 
        return {'FINISHED'}
    
class ToggleEye(bpy.types.Operator):
    bl_idname = "wm.toggleeye"
    bl_label = "Change Eyes by Toggle"
    
    def execute(self, context):
        eye = bpy.context.scene.objects.get(EYES)
        for slot in eye.material_slots:
            if slot.material is not None and slot.material.use_nodes and slot.material.node_tree is not None:
                for node in slot.material.node_tree.nodes:
                    if node.label == "Toggle" and node.type == "VALUE":
                        eyevalue = node.outputs["Value"].default_value
                        if eyevalue == 0:
                            node.outputs["Value"].default_value = 1
                        else:
                            node.outputs["Value"].default_value = 0
        return {'FINISHED'}
        
class FaceTrackerAddon (bpy.types.Operator):
    #blender info
    bl_idname = "wm.facetrackeraddon"
    bl_label = "Face Tracking Addon"

    _timer = None
    _cap = None
    _detector = dlib.get_frontal_face_detector()
    _predictor = dlib.shape_predictor('C:\Docs\scripts\shape_68.dat')
    
    #eye point numbers for dlib 68 shape
    _left = [36, 37, 38, 39, 40, 41]
    _right = [42, 43, 44, 45, 46, 47]
    
    #eyebrow point numbers for dlib
    _leftbrow = [17, 18, 19, 20, 21]
    _rightbrow = [22, 23, 24, 25, 26]
    
    #start and end indices for left and right eye points
    (_lStart, _lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (_rStart, _rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (_mStart, _mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

    #???
    _kernel = np.ones((9, 9), np.uint8)

    #eye armature controls
    armature = bpy.context.scene.objects.get(ARMATURE)
    _eyebone = armature.pose.bones.get(EYEBONE)
    _currentposx = _eyebone.location.x
    _currentposz = _eyebone.location.z
    
    #head armature controls
    _headbone = armature.pose.bones.get(HEADBONE)
    _headx = 0
    _heady = 0
    _headz = 0
    
    #accessory armature controls
    _fangbone = armature.pose.bones.get(ACCBONE)
    _fangx = 0.0
    
    #threshold value
    _thresholdval = 50
    
    #body shape key controls for eyes and mouth
    _blinking = False
    _blinkmin = 0.2
    _talkmin = 0.4
    _smilemax = 0.4
    _browmin = 1.6
    _browmax = 1.9
    _body_shapekeys = bpy.context.scene.objects.get(MODEL).data.shape_keys
  
    #3d points for head position
    _model_points = np.array([
        (0.0, 0.0, 0.0),  # Nose tip
        (0.0, -330.0, -65.0),  # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),  # Right eye right corne
        (-150.0, -150.0, -125.0),  # Left Mouth corner
        (150.0, -150.0, -125.0)  # Right mouth corner
    ])
    #literally nothing?
    def nothing(self, x):
        pass
    #calculate ratio between eyebrow length and distance between brow and eye
    def brow_aspect_ratio(self, eye, brow):
        A = dist.euclidean(eye[1], brow[2])
        B = dist.euclidean(brow[0], brow[4])
        return (A / B)
    
    #calculate eye aspect ratio
    def eye_aspect_ratio(self, eye):
        A = dist.euclidean(eye[1], eye[5])
        B = dist.euclidean(eye[2], eye[4])
        C = dist.euclidean(eye[0], eye[3])
        return (A + B) / (2.0 * C)
    def mouth_aspect_ratio(self, mouth):
        A = dist.euclidean(mouth[10], mouth[2])
        B = dist.euclidean(mouth[8], mouth[4])
        C = dist.euclidean(mouth[0], mouth[6])
        return (A + B) / (2.0 * C)
    #get facial landmarks for head pose
    def ref2dImagePoints(self, shape):
        imagePoints = [[shape.part(30).x, shape.part(30).y],
                       [shape.part(8).x, shape.part(8).y],
                       [shape.part(36).x, shape.part(36).y],
                       [shape.part(45).x, shape.part(45).y],
                       [shape.part(48).x, shape.part(48).y],
                       [shape.part(54).x, shape.part(54).y]]
        return np.array(imagePoints, dtype=np.float64)
    #create camera matrix
    def get_camera_matrix(self, focal_length, center):
        return np.array([[focal_length, 0, center[0]],
                         [0, focal_length, center[1]],
                         [0, 0, 1]], dtype="double")
    #extract xy coords of shape cv object 
    def shape_to_np(self, shape, dtype="int"):
        coords = np.zeros((68, 2), dtype=dtype)
        for i in range(0, 68):
            coords[i] = (shape.part(i).x, shape.part(i).y)
        return coords    
    #find distance between two points
    def dist_two_points(self, first, second):
        x_s = (first[0] - second[0]) ** 2
        y_s = (first[1] - second[1]) ** 2
        
        return math.sqrt(x_s + y_s)
    #draw eye on mask
    def eye_on_mask(self, mask, side, shape):
        points = [shape[i] for i in side]
        points = np.array(points, dtype=np.int32)
        return cv2.fillConvexPoly(mask, points, 255)
    #draw contours
    def contouring(self, thresh, mid, img, right=False):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            if right:
                cx += mid
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), 2)
        except:
            pass
    #find pupils in eye
    def find_pupil(self, thresh):
        cnts, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        try:
            cnt = max(cnts, key=cv2.contourArea)
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return [cx, cy]
        except:
            return None
    #modal function for blender
    def modal(self, context, event):
        global togglecamwin
        #to cancel, should improve with UI buttons
        if event.type in {'RIGHTMOUSE', 'ESC'}:
            self.cancel(context)
            return {'FINISHED'}
        #run based off timer
        if event.type == 'TIMER':
            
            #initialize camera for first run
            if self._cap == None:
                cv2.namedWindow('image')
                cv2.createTrackbar('threshold', 'image', self._thresholdval, 255, self.nothing)
                self.set_cam()
                
            #read camera input
            _, image = self._cap.read()
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            rects = self._detector(gray, 0)
            #analyze individual facial landmarks
            for rect in rects:
                shape = self._predictor(gray, rect)
                
                #detect head position
                image_points = self.ref2dImagePoints(shape)
                size = image.shape
                focal_length = size[1]
                center = (size[1] / 2, size[0] / 2)
                camera_matrix = self.get_camera_matrix(focal_length, center)
                dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
                (success, rotation_vector, translation_vector) = cv2.solvePnP(self._model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.cv2.SOLVEPNP_ITERATIVE)
                
                #find xy coordinates of facial landmarks?
                shape = self.shape_to_np(shape)
                
                #calculate eye aspect ratio for both eyes and find average
                lefteye = shape[self._lStart:self._lEnd]
                righteye = shape[self._rStart:self._rEnd]
                leftratio = self.eye_aspect_ratio(lefteye)
                rightratio = self.eye_aspect_ratio(righteye)
                eyeratio = (leftratio + rightratio) / 2.0
                
                #calculate ratio for eyebrows
                leftbrowratio = self.brow_aspect_ratio(lefteye, shape[self._leftbrow[0]:1 + self._leftbrow[4]])
                rightbrowratio = self.brow_aspect_ratio(righteye, shape[self._rightbrow[0]:1 + self._rightbrow[4]])
                browratio = (leftbrowratio + rightbrowratio) / 2.0

                
                #calculate mouth aspect ratio
                mouth = shape[self._mStart:self._mEnd]
                mouthratio = self.mouth_aspect_ratio(mouth)
                
                #control accessory movement
                browmove = ((browratio - self._browmin) / (self._browmax - self._browmin))
                browmove *= 0.255147
                browmove += 1.455605
                browmove -= 1.5708
                self._fangbone.rotation_euler = (browmove, 0, 0)
                
                #control mouth movement
                if mouthratio > self._talkmin:
                    #scale from 0 to 1
                    #mar scales from 0.5 to 1.0, cap at 1.0
                    if (mouthratio < 1.0):
                        mouthsize = (mouthratio - self._talkmin) * 1.666
                        self._body_shapekeys.key_blocks[JAWSK].value = mouthsize
                    else:
                        mouthsize = 1.0
                elif mouthratio < self._smilemax:
                    #smile
                    pass
                else:
                    #close mouth
                    self._body_shapekeys.key_blocks[JAWSK].value = 0
                
                #control eye movement+blinking
                if eyeratio > self._blinkmin:
                    #open eyes if closed
                    if self._blinking:
                        self._body_shapekeys.key_blocks[BLINKSK].value = 0
                        self._blinking = False
                    #calculate the center of the eye
                    leftcenterx = (shape[37][0] + shape[38][0] + shape[40][0] + shape[41][0]) / 4
                    leftcentery = (shape[37][1] + shape[38][1] + shape[40][1] + shape[41][1]) / 4
                    averagelefteye = [leftcenterx, leftcentery]
                    #calculate diameter of the left eye
                    diameterlefteye = self.dist_two_points(shape[38], shape[40])
                else:
                    #close eyes if open
                    self._body_shapekeys.key_blocks[BLINKSK].value = 1
                    self._blinking = True
                    pass
                #draw mask
                mask = np.zeros(image.shape[:2], dtype=np.uint8)
                mask = self.eye_on_mask(mask, self._left, shape)
                mask = self.eye_on_mask(mask, self._right, shape)
                mask = cv2.dilate(mask, self._kernel, 5)
                eyes = cv2.bitwise_and(image, image, mask=mask)
                mask = (eyes == [0, 0, 0]).all(axis=2)
                eyes[mask] = [255, 255, 255]

                mid = (shape[42][0] + shape[39][0]) // 2
            
                eyes_gray = cv2.cvtColor(eyes, cv2.COLOR_BGR2GRAY)
                self._thresholdval = cv2.getTrackbarPos('threshold', 'image')
                #threshold = cv2.getTrackbarPos('threshold', 'image')
                _, thresh = cv2.threshold(eyes_gray, self._thresholdval, 255, cv2.THRESH_BINARY)
                thresh = cv2.erode(thresh, None, iterations=2)  # 1
                thresh = cv2.dilate(thresh, None, iterations=4)  # 2
                thresh = cv2.medianBlur(thresh, 3)  # 3
                thresh = cv2.bitwise_not(thresh)
                #x y coords of left pupil
                pupillefteye = self.find_pupil(thresh[:, 0:mid])
                #move eye armature if not blinking
                if eyeratio > self._blinkmin:
                    try:
                        locx = ((pupillefteye[0] - averagelefteye[0]) / diameterlefteye) * 100
                        locy = ((pupillefteye[1] - averagelefteye[1]) / diameterlefteye) * 100
                        self._currentposx = locx * (-0.0002456)
                        self._currentposz = locy * 0.0002354
                        self._eyebone.location = (self._currentposx, self._currentposz, 0)
                    except:
                        pass
                #convert vectors into radians                
                rmat, jac = cv2.Rodrigues(rotation_vector)
                angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(rmat)
                self._headx = math.radians(angles[0] - 180)
                self._heady = math.radians(angles[1])
                if not (angles[2] < -50 or angles[2] > 50):
                    self._headz = math.radians(angles[2])
                self._headbone.rotation_euler = (self._headx, self._heady, self._headz)
                
                if togglecamwin:
                    #left eye
                    self.contouring(thresh[:, 0:mid], mid, image)
                    #right eye
                    self.contouring(thresh[:, mid:], mid, image, True)
                    #head position
                    (nose_end_point2D, jacobian) = cv2.projectPoints(np.array([(0.0, 0.0, 1000.0)]), rotation_vector, translation_vector, camera_matrix, dist_coeffs)
                    for p in image_points:
                      cv2.circle(image, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
                    p1 = (int(image_points[0][0]), int(image_points[0][1]))
                    p2 = (int(nose_end_point2D[0][0][0]), int(nose_end_point2D[0][0][1]))
                    cv2.line(image, p1, p2, (255, 0, 0), 2)                
            #display image
            try:
                if togglecamwin:
                    cv2.imshow("Output", image)
                else:
                    cv2.destroyWindow("Output")
                    cv2.imshow("image", thresh)

            except:
                pass
        return {'PASS_THROUGH'}
    #run from console?
    def execute(self, context):
        wm = context.window_manager
        self._timer = wm.event_timer_add(0.02, window=context.window)
        wm.modal_handler_add(self)
        return {'RUNNING_MODAL'}
    #stop running script, delete objects
    def cancel(self, context):
        wm = context.window_manager
        wm.event_timer_remove(self._timer)
        cv2.destroyAllWindows()
        self._cap.release()
        self._cap = None
        self.report({'INFO'}, "Stopped running")
    #initialize video feed object
    def set_cam(self):
        self._cap = cv2.VideoCapture(0)
#blender things
def register():
    bpy.utils.register_class(FaceTrackerAddon)
    bpy.utils.register_class(FaceTrackerAddonPanel)
    bpy.utils.register_class(FaceTrackerReset)
    bpy.utils.register_class(DisplayCameraWindow)
    bpy.utils.register_class(ToggleEye)

def unregister():
    bpy.utils.unregister_class(FaceTrackerAddon)
    bpy.utils.unregister_class(FaceTrackerAddonPanel)    
    bpy.utils.unregister_class(FaceTrackerReset)
    bpy.utils.unregister_class(DisplayCameraWindow)
    bpy.utils.unregister_class(ToggleEye)

if __name__ == "__main__":
    register()
    # test call
    #bpy.ops.wm.facetrackeraddon()