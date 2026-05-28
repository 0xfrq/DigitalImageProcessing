import cv2
import numpy as np
import xml.etree.ElementTree as ET

class ManualCascadeClassifier:
    def __init__(self, xml_path):
        self.stages = []
        self.features = []
        self.base_window_size = (24, 24)
        self.load_from_xml(xml_path)

    def load_from_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        cascade = root.find('cascade')
        if cascade is None:
            raise ValueError("Invalid cascade XML")
            
        width = int(cascade.find('width').text)
        height = int(cascade.find('height').text)
        self.base_window_size = (width, height)
        
        # Load features first
        features_node = cascade.find('features')
        for feature_node in features_node.findall('_'):
            rects = []
            for rect_node in feature_node.find('rects').findall('_'):
                parts = rect_node.text.strip().split()
                rects.append((int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3]), float(parts[4])))
            self.features.append(rects)
            
        # Load stages
        stages_node = cascade.find('stages')
        for stage_node in stages_node.findall('_'):
            stage_threshold = float(stage_node.find('stageThreshold').text)
            weak_classifiers = []
            
            for weak_node in stage_node.find('weakClassifiers').findall('_'):
                internal_nodes = weak_node.find('internalNodes').text.strip().split()
                leaf_values = weak_node.find('leafValues').text.strip().split()
                
                feature_idx = int(internal_nodes[2])
                threshold = float(internal_nodes[3])
                leaf0 = float(leaf_values[0])
                leaf1 = float(leaf_values[1])
                
                weak_classifiers.append({
                    'feature_idx': feature_idx,
                    'threshold': threshold,
                    'leaf0': leaf0,
                    'leaf1': leaf1
                })
            
            self.stages.append({
                'threshold': stage_threshold,
                'weak_classifiers': weak_classifiers
            })

    def detectMultiScale(self, image, scaleFactor=1.2, minNeighbors=3, minSize=(30, 30)):
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        gray = gray.astype(np.float64)
        
        win_w, win_h = self.base_window_size
        img_h, img_w = gray.shape
        
        detected_rects = []
        
        scale = 1.0
        # Iterate over scales
        while True:
            scaled_w = int(img_w / scale)
            scaled_h = int(img_h / scale)
            
            if scaled_w < minSize[0] or scaled_h < minSize[1]:
                break
            if scaled_w < win_w or scaled_h < win_h:
                break
                
            scaled_img = cv2.resize(gray, (scaled_w, scaled_h), interpolation=cv2.INTER_LINEAR)
            
            # Compute integral images
            ii, ii_sq = cv2.integral2(scaled_img.astype(np.uint8))
            
            # ii and ii_sq returned by cv2.integral have dimensions (h+1, w+1)
            win_area = win_w * win_h
            inv_area = 1.0 / win_area
            
            step = 2
            
            # Sliding window
            for y in range(0, scaled_h - win_h, step):
                for x in range(0, scaled_w - win_w, step):
                    
                    win_sum = ii[y+win_h, x+win_w] - ii[y, x+win_w] - ii[y+win_h, x] + ii[y, x]
                    win_sq_sum = ii_sq[y+win_h, x+win_w] - ii_sq[y, x+win_w] - ii_sq[y+win_h, x] + ii_sq[y, x]
                    
                    mean = win_sum * inv_area
                    variance = win_sq_sum * inv_area - mean * mean
                    std = np.sqrt(max(0.0, variance))
                    if std == 0:
                        std = 1.0
                        
                    pass_all = True
                    for stage in self.stages:
                        stage_sum = 0.0
                        for weak in stage['weak_classifiers']:
                            feature = self.features[weak['feature_idx']]
                            feature_val = 0.0
                            for rect in feature:
                                rx, ry, rw, rh, rweight = rect
                                r_sum = ii[y+ry+rh, x+rx+rw] - ii[y+ry, x+rx+rw] - ii[y+ry+rh, x+rx] + ii[y+ry, x+rx]
                                feature_val += r_sum * rweight
                                
                            feature_val *= inv_area
                            
                            if feature_val < weak['threshold'] * std:
                                stage_sum += weak['leaf0']
                            else:
                                stage_sum += weak['leaf1']
                                
                        if stage_sum < stage['threshold']:
                            pass_all = False
                            break
                            
                    if pass_all:
                        # Convert to original image scale
                        orig_x = int(x * scale)
                        orig_y = int(y * scale)
                        orig_w = int(win_w * scale)
                        orig_h = int(win_h * scale)
                        detected_rects.append([orig_x, orig_y, orig_w, orig_h])
                        
            scale *= scaleFactor
            
        # Non-Maximum Suppression (groupRectangles)
        if len(detected_rects) == 0:
            return np.array([])
            
        rects = np.array(detected_rects).tolist()
        rects, weights = cv2.groupRectangles(rects, minNeighbors, 0.2)
        return rects
