from lottie import objects
from lottie import Point, Color


class Segment:
    def __init__(self, polygon, texture=None, color=(0, 0, 0)):
        self.polygon = polygon
        self.texture = texture
        self.color = color

    def get_lottie(self):
        segment_group = objects.Group()

        if self.polygon is not None:
            polygon = segment_group.add_shape(objects.Path())
            polygon.shape.value.closed = True

            for (x, y) in self.polygon:
                polygon.shape.value.add_point(Point(int(x), int(y)))

            if self.texture is not None:
                for texture_line in self.texture:
                    texture = segment_group.add_shape(objects.Path())
                    for (x, y) in texture_line:
                        texture.shape.value.add_point(Point(int(x), int(y)))

            _color = (i/1.4 for i in self.color)
            segment_group.add_shape(objects.Stroke(Color(*_color), 2))
            segment_group.add_shape(objects.Fill(Color(*self.color)))

        return segment_group

class SegmentElipse:
    def __init__(self, position, size, color=(0, 0, 0)):
        self.size = size
        self.position = position
        self.color = color

    def get_lottie(self):
        segment_group = objects.Group()

        elipse = segment_group.add_shape(objects.Ellipse())
        elipse.size.value = Point(*self.size)
        elipse.position.value = Point(*self.position)

        segment_group.add_shape(objects.Stroke(Color(*self.color), 2))
        segment_group.add_shape(objects.Fill(Color(*self.color)))

        return segment_group


class Sticker:
    def __init__(self, segments={}):
        self.segments = segments

    def get_lottie(self):
        sricker_group = objects.Group()

        for segment in self.segments.values():
            sricker_group.add_shape(segment.get_lottie())

        sricker_group.transform.anchor_point.value = sricker_group.transform.position.value = Point(256, 256)

        return sricker_group

