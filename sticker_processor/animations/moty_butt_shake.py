from .asbtract_animation import AbsAnimation
from lottie import Point
from lottie.utils.animation import rot_shake


class MortyButtShakeAnimation(AbsAnimation):
    def __init__(self, lottie_file):
        # self.hui = "pizda"
        super().__init__(["peach"], 
                         lottie_file)

    def animate(self, face_sticker):
        animation = self.get_animation()

        face = animation.find("face")
        face.shapes = []

        face_sticker_group = face_sticker.get_lottie()
        face_sticker_group.transform.scale.value.x *= 0.35
        face_sticker_group.transform.scale.value.y *= 0.35
        face_sticker_group.transform.position.value = Point(70, 100)
        face.add_shape(face_sticker_group)

        rot_shake(face_sticker_group.transform.rotation, Point(-10, 10), 0, 60, 10)

        return animation