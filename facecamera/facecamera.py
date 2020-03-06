from .boundingbox import BoundingBox

from kivy.graphics import Translate, Fbo, ClearColor, ClearBuffers, Scale
from kivy.uix.camera import Camera
from kivy.lang import Builder
from kivy.properties import ListProperty, BooleanProperty

from PIL import Image
import numpy as np

import face_recognition as fr

class FaceCamera(Camera):
    detected_faces = ListProperty([])
    face_locations = ListProperty([])
    border_color = ListProperty((1, 0, 0, 1))
    label_color = ListProperty((1, 1, 1, 1))
    enable_face_detection = BooleanProperty(True)
    detected_count = 0

    def capture_image(self, filename):
        if self.parent is not None:
            canvas_parent_index = self.parent.canvas.indexof(self.canvas)
            if canvas_parent_index > -1:
                self.parent.canvas.remove(self.canvas)

        nw, nh = self.norm_image_size
        fbo = Fbo(size=(nw, nh), with_stencilbuffer=True)

        with fbo:
            ClearColor(0, 0, 0, 0)
            ClearBuffers()
            Scale(1, -1, 1)
            x = -self.x-(self.width-nw)/2
            y = -self.y-(self.height-nh)/2 - nh
            Translate(x, y, 0)

        fbo.add(self.canvas)
        fbo.draw()
        fbo.texture.save(filename, flipped=False)
        fbo.remove(self.canvas)

        if self.parent is not None and canvas_parent_index > -1:
            self.parent.canvas.insert(canvas_parent_index, self.canvas)

        return True

    def register_person(self, name, image):
        img = fr.load_image_file(image)
        enc = fr.face_encodings(img)[0]
        self._known_names.append(name)
        self._known_faces.append(enc)

    def __init__(self, *args, **kwargs):
        super(FaceCamera, self).__init__(*args, **kwargs)

        self._known_faces = []
        self._known_names = []

        # bounding boxes for each face
        self._bounding_boxes = []

    def on_tex(self, cam):
        super(FaceCamera, self).on_tex(cam)

        if not self.enable_face_detection:
            return
        scale = 6
        tex = cam.texture
        im = Image.frombytes('RGBA', tex.size, tex.pixels, 'raw')
        im = im.resize((tex.width//scale, tex.height//scale))
        # convert image to np.array without alpha channel
        arr = np.array(im)[:,:,:3]
        # get face locations from the resized image
        locations = fr.face_locations(arr, number_of_times_to_upsample=1,
                                      model="hog")
        # get face encodings for identification
        encodings = fr.face_encodings(arr, known_face_locations=locations,
                                      num_jitters=1)

        # update the face and location information
        faces = []
        for enc in encodings:
            # get name of the person
            matches = fr.compare_faces(self._known_faces, enc, tolerance=0.45)
            name = "Unknown"

            # use the first match which is found
            if True in matches:
                name = self._known_names[matches.index(True)]

            faces.append(name)

        # sort faces array and location array based on the name of face
        indices = np.argsort(faces)
        self.detected_faces = [f for f, i in sorted(zip(faces, indices),
                                                    key=lambda e: e[1])]
        self.face_locations = [(v*scale for v in l)
                               for l, i in sorted(zip(locations, indices),
                                                  key=lambda e: e[1])]

    def on_enable_face_detection(self, camera, enable):
        # reset faces and location arrays
        self.detected_faces = []
        self.face_locations = []

        # detect faces if this feature is activated
        if enable:
            self.on_tex(self._camera)

    def on_detected_faces(self, camera, faces):
        # remove old bounding boxes
        for bbox in self._bounding_boxes:
            self.remove_widget(bbox)
        self._bounding_boxes = []

        # if len(faces) != 1:
        #     return
            
        # add bounding boxes for each face
        for face_name in faces:
            
            # if self.detected_count == 10:
            #     get_running_app().stop()
            #     return

            bbox = BoundingBox(name=face_name, size_hint=(None, None))
            self._bounding_boxes.append(bbox)
            self.add_widget(bbox)
            self.detected_count += 1

    def on_face_locations(self, camera, locations):
        
        for loc, bbox in zip(locations, self._bounding_boxes):
            # calculate texture size and actual image size
            rw, rh = self.texture.size
            nw, nh = self.norm_image_size
            # calculate scale factor caused by allow_stretch=True and/or
            # keep_ratio = False
            sw, sh = nw/rw, nh/rh

            anchor_t = self.center[1]+nh/2
            anchor_l = self.center[0]-nw/2

            # calculate position of the face
            t, r, b, l = loc
            t = anchor_t - t*sh
            b = anchor_t - b*sh
            r = anchor_l + r*sw
            l = anchor_l + l*sw

            # update bounding box
            bbox.border_color = self.border_color
            bbox.label.color = self.label_color
            bbox.pos = (l, b)
            bbox.size = (r-l, t-b)