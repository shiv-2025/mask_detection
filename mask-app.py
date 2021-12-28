import numpy as np
import cv2 #convert images into arrays
import streamlit as st
from PIL import Image, ImageOps
import matplotlib.pyplot as plt
from mtcnn.mtcnn import MTCNN
from matplotlib.patches import Rectangle
from matplotlib.patches import Circle
from scipy.spatial.distance import cosine
from keras_vggface.vggface import VGGFace
from keras_vggface.utils import preprocess_input
#import tensorflow as tf
from keras.models import load_model

st.write("""
# Face mask detection!

This app detects if a person is wearing a face mask or not.

""")

st.header('Input image')


st.markdown(
    """
    <style>
    .reportview-container {
        background: url("https://images.pexels.com/photos/3694711/pexels-photo-3694711.jpeg?auto=compress&cs=tinysrgb&dpr=3&h=750&w=1260")
    }
   .sidebar .sidebar-content {
        background: url("https://png.pngtree.com/background/20210712/original/pngtree-modern-double-color-futuristic-neon-background-picture-image_1181573.jpg")
    }
    </style>
    """,
    unsafe_allow_html=True
)


choice = st.selectbox("Select Option",[
    "Face Mask Detection",
    "Face Verification"
])


def main():
    fig = plt.figure()
    if choice == "Face Mask Detection":
        uploaded_file = st.file_uploader("Choose File", type=["jpg","png", "jpeg"])
        if uploaded_file is not None:
            data = np.asarray(Image.open(uploaded_file))
            image = ImageOps.fit(Image.open(uploaded_file), (128,128), Image.ANTIALIAS)
            image = np.asarray(image)
            img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            img_resize = (cv2.resize(img, dsize=(128,128), interpolation=cv2.INTER_CUBIC))/255.
            
            img_reshape = img_resize[np.newaxis,...]
        
            #model1 = tf.keras.models.load_model('model.h5')
            model1 = load_model('model.h5')

            prediction = model1.predict(img_reshape)
                              
            plt.axis("off")
            plt.imshow(data)
            ax = plt.gca()
          
            detector = MTCNN()
            faces = detector.detect_faces(data)
            for face in faces:
                x, y, width, height = face['box']
                rect = Rectangle((x, y), width, height, fill=False, color='maroon')
                ax.add_patch(rect)
                for _, value in face['keypoints'].items():
                    dot = Circle(value, radius=2, color='maroon')
                    ax.add_patch(dot)
            st.pyplot(fig)
                
            if prediction[0][0]<0.5:
                st.write("Great! You're wearing a mask")
            else:
                st.write("Please wear face mask")
           
            st.text("Probability of wearing a mask:") 
            st.write(prediction)
        else:
            st.text("Please upload an image file")

    elif choice == "Face Verification":
        column1, column2 = st.columns(2)
    
        with column1:
            image1 = st.file_uploader("Choose File", type=["jpg","png", "jpeg"])
           
        with column2:
            image2 = st.file_uploader("Select File", type=["jpg","png", "jpeg"])
        if (image1 is not None) & (image2  is not None):
            col1, col2 = st.columns(2)
            image1 =  Image.open(image1)
            image2 =  Image.open(image2)
            with col1:
                st.image(image1)
            with col2:
                st.image(image2)

            filenames = [image1,image2]

            faces = [extract_face(f) for f in filenames]
            samples = np.asarray(faces, "float32")
            samples = preprocess_input(samples, version=2)
            model2 = VGGFace(model= "resnet50" , include_top=False, input_shape=(224, 224, 3),
            pooling= "avg" )
            # perform prediction
            embeddings = model2.predict(samples)
            thresh = 0.5
            score = cosine(embeddings[0], embeddings[1])
            if score <= thresh:
                st.success( " >face is a match (%.3f <= %.3f) " % (score, thresh))
            else:
                st.error(" >face is NOT a match (%.3f > %.3f)" % (score, thresh))

def extract_face(file):
    pixels = np.asarray(file)
    plt.axis("off")
    plt.imshow(pixels)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]["box"]
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize((224, 224))
    face_array = np.asarray(image)
    return face_array

if __name__ == "__main__":
    main()                

