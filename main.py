from detector import *

#Sumario model_type
# OD = Detecção de objetos
# IS = Segmentação de instacias
# KP = Detecção Keypoint
# LVIS = Segmentação de Instacias LVIS
# PS = Segmentação Panóptica

detector = Detector(model_type='IS')

detector.onVideo()
