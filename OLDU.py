import cv2
import numpy as np
import torch
import dlib
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from skimage.color import rgb2lab



img_path=""



def load_image_and_get_dimensions(image_path):
    image = dlib.load_rgb_image(image_path)
    return image, image.shape[1], image.shape[0]  



def draw_landmarks(image, landmarks):
    image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  
    for (x, y) in landmarks:
        cv2.circle(image_bgr, (x, y), 1, (0, 255, 0), -1)
    cv2.imshow('Landmarks', image_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def get_landmarks(detector, predictor, image):
    detections = detector(image)
    if not detections:
        raise ValueError("Yuz algilanmadi")
    rect = detections[0]
    return np.array([[p.x, p.y] for p in predictor(image, rect).parts()])



def check_face_size_and_centering(landmarks, image_width, image_height):
    face_height = landmarks[8][1] - landmarks[19][1] 
    face_width = landmarks[16][0] - landmarks[0][0]  
    face_height_percentage = (face_height / image_height) * 100
    face_height_score = max(0, min(100, (face_height_percentage - 30) * 3.33)) if face_height_percentage <= 60 else max(0, (180 - face_height_percentage) * 3.33)
    is_correct_size = 30 <= face_height_percentage <= 60
    face_center_x = (landmarks[16][0] + landmarks[0][0]) / 2
    centering_deviation = abs(face_center_x - (image_width / 2))
    centering_percentage = max(0, 100 - (centering_deviation / (0.04 * image_width) * 100))
    is_centered = abs(face_center_x - (image_width / 2)) < (0.04 * image_width)
    combined_score = (face_height_score + centering_percentage) / 2
    print(f"kafa/genel yukseklik orani basarisi: %{face_height_score:.2f}")
    print(f"kafa/genel genislik orani basarisi: %{centering_percentage:.2f}")
    print (f"ortalama yukseklik-genislik basarisi: %{combined_score:.2f}")
    return is_correct_size and is_centered



def check_facial_orientation_and_expression(landmarks):
    nose_aligned_with_chin = abs(landmarks[33][0] - landmarks[8][0]) < 5  
    mouth_closed = abs(landmarks[62][1] - landmarks[66][1]) < 5 
    return nose_aligned_with_chin and mouth_closed



def check_sharpness_and_contrast(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = cv2.Laplacian(gray, cv2.CV_64F).var()
    return fm > 100 and np.std(gray) > 50



def check_lighting(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return np.mean(gray) > 80 and np.mean(gray) < 180



def check_image_quality(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    contrast = np.std(gray)
    return 50 <= brightness <= 200 and contrast > 15



face_cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(face_cascade_path)



def initialize_models():
    mtcnn = MTCNN(image_size=160, margin=0, min_face_size=20)
    facenet = InceptionResnetV1(pretrained='vggface2').eval()
    eyeglass_model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    eyeglass_model.eval()
    return mtcnn, facenet, eyeglass_model



def glasses_detector(path):
    detector = dlib.get_frontal_face_detector()
    predictor_path = 'C:/Users/tayla/Downloads/shape_predictor_81_face_landmarks (2).dat'
    predictor = dlib.shape_predictor(predictor_path)
    img = dlib.load_rgb_image(path)
    detections = detector(img)
    if not detections:
        return 'Yuz algilanmadi'
    rect = detections[0]
    sp = predictor(img, rect)
    landmarks = np.array([[p.x, p.y] for p in sp.parts()])
    x_min, x_max = min(landmarks[i][0] for i in range(28, 36)), max(landmarks[i][0] for i in range(28, 36))
    y_min, y_max = landmarks[20][1], landmarks[30][1]
    img2 = Image.open(path)
    img2 = img2.crop((x_min, y_min, x_max, y_max))
    img_blur = cv2.GaussianBlur(np.array(img2), (3, 3), sigmaX=0, sigmaY=0)
    edges = cv2.Canny(image=img_blur, threshold1=100, threshold2=200)
    edges_center = edges.T[len(edges.T) // 2]
    if 255 in edges_center:
        print("Gozluk takma yuzdesi: %100")
    else:
        print("Gozluk takma yuzdesi: %0")
    return 'Gozluk var' if 255 in edges_center else 'Gozluk yok'
    


def load_image_and_get_dimensions(image_path):
    image = dlib.load_rgb_image(image_path)
    return image, image.shape[1], image.shape[0]



def check_red_eye(image):
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=10)
    for (ex, ey, ew, eh) in eyes:
        eye_roi = image[ey:ey+eh, ex:ex+ew]
        eye_hsv = cv2.cvtColor(eye_roi, cv2.COLOR_BGR2HSV)
        lower_red = np.array([0, 50, 50])
        upper_red = np.array([10, 255, 255])
        mask1 = cv2.inRange(eye_hsv, lower_red, upper_red)
        lower_red = np.array([170, 50, 50])
        upper_red = np.array([180, 255, 255])
        mask2 = cv2.inRange(eye_hsv, lower_red, upper_red)
        red_eye_mask = mask1 + mask2
        if np.sum(red_eye_mask) > 50:  
            print("Kirmizi goz bulunmama basarisi: %0")
            return True
        else:
            print("Kirmizi goz bulunmama basarisi: %100")
   
    return False


import traceback

def is_head_and_shoulders_visible(image, face_rect):
    face_region = image[face_rect[1]:face_rect[1]+face_rect[3], face_rect[0]:face_rect[0]+face_rect[2]]
    total_pixels = face_rect[2] * face_rect[3]
    white_pixels = cv2.countNonZero(cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY))
    visible_area_percentage = white_pixels / total_pixels*100
    print (f"Omuz-yuz gorunurlugu basarisi: %{visible_area_percentage:.2f}")
    if visible_area_percentage >= 90:
        return True
    else:
        return False 
   



def is_background_valid(image):
    height, width, _ = image.shape
    bg_rect_width = width // 10
    bg_rect_height = height // 10

    top_left_rect = image[5:5+bg_rect_height, 5:5+bg_rect_width]
    top_right_rect = image[5:5+bg_rect_height, width-5-bg_rect_width:width-5]

    avg_color_top_left = np.mean(top_left_rect, axis=(0, 1))
    avg_color_top_right = np.mean(top_right_rect, axis=(0, 1))
    avg_color = (avg_color_top_left + avg_color_top_right) / 2

    return np.all(avg_color >= 200)



def is_proper_lighting(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    average_brightness = np.mean(gray_image)
    return 80 <= average_brightness <= 240



def check_for_accessories(image, landmarks):
    ear_regions = landmarks[17:21] + landmarks[22:26]
    accessories_detected = False
    for (x, y) in ear_regions:
        region = image[max(0, y-10):y+10, max(0, x-10):x+10]
        if np.std(region) > 15:  
            accessories_detected = True
            break
    return not accessories_detected



def percentage_based_evaluation(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Resim bulunamadi")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    height, width = gray.shape
    background_left = gray[:, :width // 10]
    background_right = gray[:, 9 * width // 10:]
    background_avg_left = np.mean(background_left)
    background_avg_right = np.mean(background_right)
    background_percentage = ((background_avg_left + background_avg_right) / 2) / 255 * 100
    print(f"Arka plan kontrolu basarisi: %{background_percentage:.2f}")

    average_brightness = np.mean(gray)
    proper_lighting_percentage = (average_brightness - 80) / (240 - 80) * 100
    proper_lighting_percentage = max(0, min(100, proper_lighting_percentage))
    print(f"Isik kosullari basarisi: %{proper_lighting_percentage:.2f}")

    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_percentage = (laplacian_var - 100) / 300 * 100  
    sharpness_percentage = max(0, min(100, sharpness_percentage))
    print(f"Resim keskinligi basarisi: %{sharpness_percentage:.2f}")


    std_dev = np.std(gray)
    contrast_percentage = (std_dev - 15) / 85 * 100  
    contrast_percentage = max(0, min(100, contrast_percentage))
    quality_percentage = (average_brightness - 50) / 150 * 100  
    quality_percentage = max(0, min(100, quality_percentage))
    print(f"Resim kalitesi basarisi: %{(contrast_percentage + quality_percentage) / 2:.2f}")        

percentage_based_evaluation(img_path)



def process_image(image_path, mtcnn, facenet, eyeglass_model):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Resim bulunamadi")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    glasses_present = glasses_detector(image_path) == 'Gozluk var'
    image_batch = np.expand_dims(image, axis=0)
    aligned_faces = mtcnn(image_batch)
    if aligned_faces is None or len(aligned_faces) == 0:
        raise ValueError("Yuz algilanmadi")
    embeddings = [facenet(face.unsqueeze(0)) for face in aligned_faces if isinstance(face, torch.Tensor)]
    return embeddings, glasses_present


def run_all_checks(image_path):
    mtcnn, facenet, eyeglass_model = initialize_models()
    embeddings, glasses_worn = process_image(image_path, mtcnn, facenet, eyeglass_model)
    image, image_width, image_height = load_image_and_get_dimensions(image_path)


    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    upper_body_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_upperbody.xml')
    gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3)
    upper_bodies = upper_body_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=3)

    face_bbox = [(x, y, w, h) for (x, y, w, h) in faces]
    upper_body_bbox = [(x, y, w, h) for (x, y, w, h) in upper_bodies]

    detector = dlib.get_frontal_face_detector()
    predictor_path = 'C:/Users/tayla/Downloads/shape_predictor_81_face_landmarks (2).dat'
    predictor = dlib.shape_predictor(predictor_path)
    landmarks = get_landmarks(detector, predictor, image)
    results = check_face_size_and_centering(landmarks, image_width, image_height)

    background_success = is_background_valid(image)
    proper_lighting_success = is_proper_lighting(image)
    sharpness_success = check_sharpness_and_contrast(image)
    quality_success = check_image_quality(image)
    glasses_success = 1 if not glasses_worn else 0
    dimension_success = check_face_size_and_centering(landmarks, image_width, image_height)
    red_eye_success = 1 if not check_red_eye(image) else 0
    head_and_shoulders_success = all(is_head_and_shoulders_visible(image, face) for face in faces)
    other_accessory_check_success = check_for_accessories(image, landmarks)
    expression_success = check_facial_orientation_and_expression(landmarks)

    total_checks = 10
    success_count = sum([background_success, proper_lighting_success, sharpness_success,
                         quality_success, glasses_success, dimension_success, red_eye_success,
                         head_and_shoulders_success, other_accessory_check_success, expression_success])
    success_rate = (success_count / total_checks) * 100
    print(f"Total Success Rate: {success_rate}%")
    

    max_bar_width = 600
    bar_height = 50
    img_height = 100
    text_space = 100 

    actual_bar_width = int((success_rate / 100.0) * max_bar_width)

    img_width = max_bar_width + text_space  
    img = np.zeros((img_height, img_width, 3), dtype=np.uint8)

    if success_rate >= 50:
        green_intensity = int(255 * ((success_rate - 50) / 50.0))  
        color = (0, green_intensity, 0)
    else:
        red_intensity = int(255 * (1 - (success_rate / 50.0)))  
        color = (0, 0, red_intensity)
    if actual_bar_width > 0:  
        img[25:25 + bar_height, 0:actual_bar_width] = color
    
    cv2.rectangle(img, (0, 25), (max_bar_width, 25 + bar_height), (255, 255, 255), 2)
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, f"{success_rate:.2f}%", (max_bar_width + 5, 50), font, 0.75, (255, 255, 255), 2, cv2.LINE_AA)

    cv2.imshow('Success Rate Bar', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



    red_eye_ok = not check_red_eye(image)
    quality_ok = check_image_quality(image)
    accessories_ok = not glasses_worn
    expression_ok = check_facial_orientation_and_expression(landmarks)
    keskinlik_ok = check_sharpness_and_contrast(image)
    lighting_ok = check_lighting(image)
    head_and_shoulders_ok = all(is_head_and_shoulders_visible(image, face) for face in faces)
    background_ok = is_background_valid(image)
    proper_lighting_ok = is_proper_lighting(image)
    other_accessory_check_ok = check_for_accessories(image, landmarks)



    checks_passed = results and quality_ok and accessories_ok and expression_ok and red_eye_ok and keskinlik_ok and lighting_ok and head_and_shoulders_ok and background_ok and proper_lighting_ok and other_accessory_check_ok
    
    print(f"Yuz analizi: {checks_passed}")
    if not checks_passed:
        print("Hata detaylari:")
        if not results:
            print("- Oranlar hatali.")
        if not quality_ok:
            print("- Resim kalitesi dusuk.")
        if not accessories_ok:
            print("- Gozluk algilandi.")
        if not expression_ok:
            print("- Mimikler notr degil.")   
        if not keskinlik_ok:
            print("- Keskinlikte yetersizlik var.")      
        if not lighting_ok:
            print("- Kotu isik kosullari.")
        if not red_eye_ok:
            print("- Kirmizi goz algilandi.")
        if not head_and_shoulders_ok:
            print("- Yuz ve omuz tam belli degil.")
        if not background_ok:
            print("- Arka plan cok uygun degil.")
        if not proper_lighting_ok:
            print("- Isik kosullari yeterli degil.")
        if not other_accessory_check_ok:
            print("- Sac, kupe veya basortusu gibi sebeplerden dolayi.")  

    

run_all_checks(img_path)
