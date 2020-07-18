# crop_faces_from_image
- <b>input:</b> input directory of images that include faces </br>
- <b>ouput:</b> output directory of cropped faces of the images from the input directory </br>
- <b>face_detector:</b> the face detection model <i>(using res10_300x300_ssd_iter_140000.caffemodel) </i></br>
- <b>crop_faces_from_image.py:</b> crops faces from the images of <b>input</b>, and put them in <b>output</b></br>

# how to use crop_faces_from_images.py
- for example, if you execute the program as: </br>
- <i><b> python crop_faces_from_image.py -n 10</b></i></br> 
- the cropped faces in the output directory will start from number <b>10</b> </br>
- the default is <b>1</b>, so if you just execute the program as:
- <i><b> python crop_faces_from_image.py</b></i>
- the cropped faces in the output directory will start from number <b>1</b> </br>
