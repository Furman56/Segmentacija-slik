import cv2 as cv
import numpy as np

def kmeans(slika, k=3, iteracije=10):
    '''Izvede segmentacijo slike z uporabo metode k-means.'''
    
    # Pridobi dimenzije slike
    h, w, d = slika.shape  # h = height, w = width, d = depth (barvni razredi)
    
    # Pretvori sliko v 2D matriko kjer je vsaka vrstica en pixel
    pixels = slika.reshape(-1, d)  # (st_pixlov, depth)
    
    # Inicializira naključno st centrov iz pikslov
    centers = pixels[np.random.choice(pixels.shape[0], k, replace=False)] 
    

def meanshift(slika, velikost_okna, dimenzija):
    '''Izvede segmentacijo slike z uporabo metode mean-shift.'''
    pass

def izracunaj_centre(slika, izbira, dimenzija_centra, T):
    '''Izračuna centre za metodo kmeans.'''
    pass

if __name__ == "__main__":
    pass