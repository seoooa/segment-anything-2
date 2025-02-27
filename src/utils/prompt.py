import cv2
import numpy as np

def get_click_point(image):
    """
    Get the click point of the object in the image
    """
    points = []

    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.imshow("Select Object", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Select Object", mouse_callback)
    
    while len(points) == 0:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return np.array(points)

def get_bounding_box(image):
    """
    Get the bounding box of the object in the image
    """
    points = []
    temp_point = None
    drawing = False
    img_copy = image.copy()

    def mouse_callback(event, x, y, flags, param):
        nonlocal points, temp_point, drawing, img_copy
        
        if event == cv2.EVENT_LBUTTONDOWN:
            if not points:
                drawing = True
                points.append((x, y))
                temp_point = (x, y)
                img_copy = image.copy()
            elif len(points) == 1:
                points.append((x, y))
                cv2.rectangle(img_copy, points[0], points[1], (0, 255, 0), 2)
            
        elif event == cv2.EVENT_MOUSEMOVE and drawing and len(points) == 1:
            img_copy = image.copy()
            cv2.rectangle(img_copy, points[0], (x, y), (0, 255, 0), 2)

    cv2.imshow("Draw Bounding Box", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
    cv2.setMouseCallback("Draw Bounding Box", mouse_callback)
    
    while len(points) < 2:
        cv2.imshow("Draw Bounding Box", cv2.cvtColor(img_copy, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cv2.destroyAllWindows()
    return np.array(points)