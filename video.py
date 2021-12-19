class Video(object):
    def __init__(self, path):
        self.path = path

    def play(self):
        from os import startfile
        startfile(self.path)


class Movie_MP4(Video):
    type = 'MP4'


def videoOut(filename):
    class Video(object):
        def __init__(self, path):
            self.path = path

        def play(self):
            from os import startfile
            startfile(self.path)

    class Movie_MP4(Video):
        type = 'MP4'


def videoOut(filename):
    movie = Movie_MP4(filename)
    movie.play()
