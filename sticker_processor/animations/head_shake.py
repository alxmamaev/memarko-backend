from .asbtract_animation import AbsAnimation
from lottie import Point
from lottie import objects
from lottie.utils.animation import rot_shake


class HeadShakeAnimation(AbsAnimation):
    def __init__(self, lottie_file):
        # self.hui = "pizda"
        super().__init__(["no_good"], 
                         lottie_file)

    def animate(self, face_sticker):
        animation = objects.Animation()

        layer = objects.ShapeLayer()
        animation.add_layer(layer)
        face_sticker_group = face_sticker.get_lottie()
        layer.add_shape(face_sticker_group)

        rot_shake(face_sticker_group.transform.rotation, Point(-10, 10), 0, 60, 10)

        return animation