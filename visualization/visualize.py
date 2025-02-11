import openslide


def main():
    slide = openslide.OpenSlide('/mnt/slides/normal_039.tif')
    w, h = slide.dimensions
    print(w, h)


if __name__ == "__main__":
    main()
