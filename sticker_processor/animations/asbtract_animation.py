from copy import deepcopy
from lottie.parsers.tgs import parse_tgs


class AbsAnimation:
    def __init__(self, emojis_names, lottie_file):
        self.emojis_names = emojis_names
        self.lottie_file = lottie_file

    def get_animation(self):
        return parse_tgs(self.lottie_file)

    def get_sticker_name(self):
        return "~".join(self.emojis_names)

    def animate(self, face_sticker):
        raise NotImplementedError()

    def __call__(self, face_sticker):
        return self.animate(face_sticker)