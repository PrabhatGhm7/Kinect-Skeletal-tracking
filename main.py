import cv2
import mediapipe as mp
import pygame
import numpy as np
from pygame import gfxdraw
import math

class GlowingSkeleton:
    def __init__(self):
        # Initialize MediaPipe
        self.mp_pose = mp.solutions.pose
        self.mp_face_mesh = mp.solutions.face_mesh
        self.pose = self.mp_pose.Pose()
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize Pygame
        pygame.init()
        self.width, self.height = 800, 600
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Animated Skeleton")
        
        # Colors
        self.GLOW_COLOR = (200, 255, 250)
        self.BACKGROUND = (0, 0, 0)
        
        # Joint sizes
        self.SIZES = {
            'head': 25,
            'body': 35,
            'shoulder': 20,
            'elbow': 15,
            'wrist': 12,
            'hip': 20,
            'knee': 15,
            'ankle': 12,
        }
        
        # Face landmark indices
        self.FACE_LANDMARKS = {
            'left_eye': [33, 159, 133, 145, 153, 157],  # Left eye 
            'right_eye': [362, 386, 263, 374, 380, 385],  # Right eye landmarks
            'mouth_outer': [61, 291, 39, 181, 84, 17, 314, 405, 321, 375, 291],  # Outer mouth landmark
            'left_eyebrow': [70, 63, 105, 66, 107],  # Left eyebrow 
            'right_eyebrow': [336, 296, 334, 293, 300]  # Right eyebrows
        }
        
        #Initializing our webcam
        self.cap =cv2.VideoCapture(0)
        
    def draw_glowing_circle(self, surface,  color,pos, radius):
        # Drawing the main circle
        pygame.draw.circle(surface,  color,pos,  radius)
        
        #Making the glow effect
        for i in range(3):
            glow_radius =radius + (i * 2)
            glow_color =  (*color, 100 - (i * 30))
            glow_surface =  pygame.Surface(( glow_radius *  2 + 4, glow_radius * 2 + 4),  pygame.SRCALPHA)
            pygame.draw.circle(glow_surface, glow_color, ( glow_radius  + 2,  glow_radius+ 2),  glow_radius)
            surface.blit(glow_surface, (pos[0] - glow_radius  - 2, pos[1] - glow_radius - 2), special_flags=pygame.BLEND_ALPHA_SDL2)

    def draw_glowing_line(self, surface,  color, start_pos,  end_pos, width):
        #Now we draw main line
        pygame.draw.line(surface, color,  start_pos, end_pos, width)
        
        
        for i in range(2):
            glow_color = (*color[:3], 50 - (i * 20))
            pygame.draw.line(surface, glow_color, start_pos, end_pos, width + (i * 2))

    def draw_face(self, face_landmarks):
        if not face_landmarks:
            return
        
        landmarks = face_landmarks.landmark
        
        def to_screen_coords(landmark):
            return (
                int(landmark.x * self.width),
                int(landmark.y * self.height)
            )
        
        # Draw eyes
        for eye_name in ['left_eye', 'right_eye']:
            points = [to_screen_coords(landmarks[idx]) for idx in self.FACE_LANDMARKS[eye_name]]
            pygame.draw.polygon(self.screen, self.GLOW_COLOR, points, 2)
            
            # Here adding the glowy in eyes
            center = (
                sum(p[0] for p in points) // len(points),
                sum(p[1] for p in points) // len(points)
            )
            self.draw_glowing_circle(self.screen,  self.GLOW_COLOR, center, 3)
        
        # Drawing the  mouth with exprssion
        mouth_points = [to_screen_coords(landmarks[idx]) for idx in self.FACE_LANDMARKS['mouth_outer']]
        
        # Calculating when we open mouth 
        mouth_top = landmarks[13].y
        mouth_bottom = landmarks[14].y
        mouth_openness = (mouth_bottom -mouth_top) * self.height
        
        # Drawing the pencil like mouth
        if mouth_openness > 20:  # Opening the mouth
            pygame.draw.polygon(self.screen,self.GLOW_COLOR, mouth_points, 2)
        else:  
            mouth_center_y = sum(p[1] for p in mouth_points) / len(mouth_points)
            mouth_points_adjusted = [(p[0], p[1] - 5 if i in [2, 3, 4] else p[1] + 5) 
                                   for i, p in enumerate(mouth_points)]
            pygame.draw.lines(self.screen,  self.GLOW_COLOR, False, mouth_points_adjusted, 2)
        
        # Drawing pencil like eyebrows
        for brow_name in ['left_eyebrow', 'right_eyebrow']:
            points = [to_screen_coords(landmarks[idx]) for idx in self.FACE_LANDMARKS[brow_name]]
            self.draw_glowing_line(self.screen, self.GLOW_COLOR, points[0], points[-1], 2)

    def draw_skeleton(self, pose_landmarks, face_landmarks):
        if not pose_landmarks.pose_landmarks:
            return
        
        # CLS
        self.screen.fill(self.BACKGROUND)
        
        lm = pose_landmarks.pose_landmarks.landmark
        
        def to_screen_coords(landmark):
            return (
                int(landmark.x * self.width),
                int(landmark.y * self.height)
            )
        
        # Now for body connection
        connections = [
            (11, 12), (12, 24), (24, 23), (23, 11),  # Main body
            (12, 14), (14, 16), (11, 13), (13, 15),  # Arm
            (24, 26), (26, 28), (23, 25), (25, 27)   # Legs
        ]
        
        # Draw connections with glow effect
        for start_idx, end_idx in connections:
            start_pos =  to_screen_coords(lm[start_idx])
            end_pos = to_screen_coords(lm[end_idx])
            self.draw_glowing_line(self.screen, self.GLOW_COLOR, start_pos, end_pos, 4)
        
        # Draw joints
        joint_mapping = {
            0: ('head',  lm[0]),
            11: ('shoulder', lm[11]), 12:('shoulder', lm[12]),
            13: ('elbow', lm[13]),  14: ('elbow', lm[14]),
            15: ('wrist',  lm[15]),  16: ('wrist', lm[16]),
            23: ('hip',lm[23]),  24: ('hip', lm[24]),
            25: ('knee',lm[25]),26: ('knee', lm[26]),
            27: ('ankle',  lm[27]),  28: ('ankle', lm[28]),
        }
        
        # Draw torso
        mid_shoulder = (
            (lm[11].x +  lm[12].x) / 2,
            (lm[11].y +lm[12].y) / 2
        )
        torso_pos = (int(mid_shoulder[0] * self.width), int(mid_shoulder[1] * self.height))
        self.draw_glowing_circle(self.screen, self.GLOW_COLOR, torso_pos, self.SIZES['body'])
        
        # Draw points
        for idx, (joint_type, landmark) in joint_mapping.items():
            if landmark.visibility > 0.5:
                pos = to_screen_coords(landmark)
                self.draw_glowing_circle(self.screen, self.GLOW_COLOR, pos, self.SIZES[joint_type])
        
        # Draw face if there is face in frame
        if face_landmarks and face_landmarks.multi_face_landmarks:
            self.draw_face(face_landmarks.multi_face_landmarks[0])
    
    def run(self):
        running = True
        clock = pygame.time.Clock()
        
        while running:
            for event in  pygame.event.get():
                if  event.type == pygame.QUIT:
                    running = False
            
            success,  image = self.cap.read()
            if not success:
                continue
            
            image =cv2.flip(image, 1)
            image_rgb =  cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        
            pose_results =  self.pose.process(image_rgb)
            face_results = self.face_mesh.process(image_rgb)
            
            self.draw_skeleton(pose_results, face_results)
            pygame.display.flip()
            clock.tick(30)
        
        self.cap.release()
        pygame.quit()

if __name__ == "__main__":
    app = GlowingSkeleton()
    app.run()